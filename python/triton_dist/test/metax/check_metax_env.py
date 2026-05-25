#!/usr/bin/env python3
"""Check whether the local Metax runtime environment can run TritonDist tests.

This script intentionally does not import triton_dist or pymxshmem, so it can run
before TritonDist is built. It mirrors the topology checks used by
python/triton_dist/kernels/metax/utils.py closely enough to diagnose common
launch/topology mismatches.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def load_metax_utils():
    utils_path = Path(__file__).resolve().parents[2] / "kernels" / "metax" / "utils.py"
    spec = importlib.util.spec_from_file_location("metax_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load metax utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _metax_utils = load_metax_utils()
    _metax_utils_error: Exception | None = None
except Exception as exc:
    _metax_utils = None
    _metax_utils_error = exc

get_numa_world_size = getattr(_metax_utils, "get_numa_world_size", None)
has_fullmesh_mxlink_ngpus = getattr(_metax_utils, "has_fullmesh_mxlink_ngpus", None)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    warn_only: bool = False


def run(cmd: list[str], *, shell: bool = False) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd if not shell else " ".join(cmd), shell=shell, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=10)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return 127, "", str(exc)


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def print_kv(key: str, value: object) -> None:
    print(f"{key:28} {value}")


def status(check: Check) -> str:
    if check.ok:
        return "OK"
    return "WARN" if check.warn_only else "FAIL"


def parse_visible_devices() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return None
    devs: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            devs.append(int(item))
        except ValueError:
            return None
    return devs


def mx_smi_gpu_count() -> int | None:
    rc, out, _ = run(["mx-smi", "--list"])
    if rc != 0:
        return None
    return sum(1 for line in out.splitlines() if "GPU" in line and "UUID" in line)


def parse_mxlink_matrix() -> list[list[int]]:
    rc, out, err = run(["mx-smi", "topo", "-m"])
    if rc != 0:
        raise RuntimeError(err or "mx-smi topo -m failed")

    lines = [line.strip() for line in out.splitlines() if line.startswith("GPU")]
    matrix = [[-1 for _ in range(len(lines))] for _ in range(len(lines))]
    for i, line in enumerate(lines):
        parts = line.split()
        for j in range(1, len(parts)):
            if j - 1 >= len(lines):
                break
            if "MX" in parts[j]:
                matrix[i][j - 1] = 1
    return matrix


def parse_gpu_pci_addresses() -> dict[int, str]:
    rc, out, err = run(["mx-smi", "topo", "-t"])
    if rc != 0:
        raise RuntimeError(err or "mx-smi topo -t failed")

    gpu_to_addr: dict[int, str] = {}
    current_addr = None
    for line in out.splitlines():
        branch_line = line.strip()
        parts = branch_line.split("-+-")
        last_part = parts[-1].strip()
        if line.startswith("+-pci") or line.startswith("\\-pci"):
            if len(parts) > 1:
                current_addr = parts[1]
            if "GPU#" in last_part:
                comp = last_part.split()
                if len(comp) >= 2:
                    gpu_to_addr[int(comp[1][4:])] = current_addr or ""
        elif "GPU#" in parts[-1]:
            comp = last_part.split()
            if len(comp) >= 2:
                gpu_to_addr[int(comp[1][4:])] = current_addr or ""
    return gpu_to_addr


def numa_nodes_for_gpus(gpu_count: int) -> list[int | None]:
    try:
        gpu_to_addr = parse_gpu_pci_addresses()
    except RuntimeError:
        return [None] * gpu_count

    numa_nodes: list[int | None] = []
    for gpu_index in range(gpu_count):
        pci_address = gpu_to_addr.get(gpu_index)
        if not pci_address:
            numa_nodes.append(None)
            continue
        path = Path(f"/sys/bus/pci/devices/0000:{pci_address}/numa_node")
        try:
            numa_nodes.append(int(path.read_text().strip()))
        except Exception:
            numa_nodes.append(None)
    return numa_nodes


def infer_fullmesh_world_size(num_ranks: int, device_count: int, numa_world_size: int | None) -> int | None:
    if has_fullmesh_mxlink_ngpus is None:
        return None
    fullmesh_world_size = numa_world_size or num_ranks
    fullmesh_world_size = min(fullmesh_world_size, device_count)
    while fullmesh_world_size > 1:
        if device_count % fullmesh_world_size != 0:
            fullmesh_world_size //= 2
            continue
        try:
            if has_fullmesh_mxlink_ngpus(fullmesh_world_size):
                return fullmesh_world_size
        except Exception:
            return None
        fullmesh_world_size //= 2
    return fullmesh_world_size if fullmesh_world_size > 0 else None


def import_torch_info() -> tuple[bool, str, int | None, bool | None]:
    try:
        import torch
    except Exception as exc:
        return False, str(exc), None, None
    try:
        count = torch.cuda.device_count()
        available = torch.cuda.is_available()
    except Exception as exc:
        return True, f"{torch.__version__}; cuda query failed: {exc}", None, None
    return True, torch.__version__, count, available


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Metax environment without importing triton_dist.")
    parser.add_argument(
        "--num-ranks", type=int, default=None,
        help="World size to evaluate. Defaults to WORLD_SIZE, ARNOLD_WORKER_GPU, or detected GPU count.")
    parser.add_argument("--local-ranks", type=int, default=None,
                        help="Local ranks per node. Defaults to --num-ranks for this intra-node check.")
    args = parser.parse_args()

    print_section("Commands")
    checks: list[Check] = []
    checks.append(Check("mx-smi in PATH", shutil.which("mx-smi") is not None, shutil.which("mx-smi") or "missing"))
    checks.append(Check("python", True, sys.executable))
    for check in checks:
        print(f"[{status(check):4}] {check.name}: {check.detail}")

    print_section("Launch Environment")
    for key in [
            "CUDA_VISIBLE_DEVICES",
            "ARNOLD_WORKER_GPU",
            "ARNOLD_WORKER_NUM",
            "ARNOLD_ID",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "MXSHMEM_BOOTSTRAP",
            "MXSHMEM_IB_ENABLE_IBGDA",
            "MXSHMEM_IB_ENABLE_IBRC",
    ]:
        print_kv(key, os.environ.get(key, "<unset>"))

    detected_count = mx_smi_gpu_count()
    visible = parse_visible_devices()
    visible_count = len(visible) if visible is not None else detected_count
    env_num_ranks = os.environ.get("WORLD_SIZE")
    default_num_ranks = int(env_num_ranks) if env_num_ranks and env_num_ranks.isdigit() else visible_count
    num_ranks = args.num_ranks or default_num_ranks or 1
    local_ranks = args.local_ranks or int(os.environ.get("LOCAL_WORLD_SIZE") or num_ranks)

    print_section("GPU Runtime")
    torch_imported, torch_detail, torch_count, torch_cuda_available = import_torch_info()
    runtime_checks = [
        Check("mx-smi --list", detected_count is not None, f"gpu_count={detected_count}"),
        Check("torch import", torch_imported, torch_detail, warn_only=True),
        Check("metax utils importlib", _metax_utils is not None,
              "loaded" if _metax_utils is not None else str(_metax_utils_error), warn_only=True),
    ]
    if torch_imported:
        runtime_checks.append(
            Check("torch.cuda.is_available", bool(torch_cuda_available), f"available={torch_cuda_available}",
                  warn_only=True))
        runtime_checks.append(
            Check("torch cuda device count", torch_count is not None, f"device_count={torch_count}", warn_only=True))
    for check in runtime_checks:
        print(f"[{status(check):4}] {check.name}: {check.detail}")
    print_kv("evaluated num_ranks", num_ranks)
    print_kv("evaluated local_ranks", local_ranks)
    print_kv("visible device ids", visible if visible is not None else "not set")

    device_count = visible_count or torch_count or detected_count or 0
    print_section("Topology")
    matrix: list[list[int]] | None = None
    try:
        matrix = parse_mxlink_matrix()
        print(f"[OK  ] mx-smi topo -m: matrix_size={len(matrix)}")
    except RuntimeError as exc:
        print(f"[FAIL] mx-smi topo -m: {exc}")

    numa_nodes = numa_nodes_for_gpus(device_count) if device_count else []
    try:
        numa_world_size = get_numa_world_size() if get_numa_world_size is not None and device_count else None
    except Exception:
        numa_world_size = None
    print_kv("device_count used", device_count)
    print_kv("numa nodes", numa_nodes or "unknown")
    print_kv("numa_world_size", numa_world_size if numa_world_size is not None else "unknown")

    fullmesh_world_size = infer_fullmesh_world_size(local_ranks, device_count, numa_world_size)
    print_kv("fullmesh_world_size", fullmesh_world_size if fullmesh_world_size is not None else "unknown")

    print_section("AG GEMM Intra-node Readiness")
    final_checks = [
        Check("visible GPU count >= local ranks", device_count >= local_ranks,
              f"device_count={device_count}, local_ranks={local_ranks}"),
        Check("local ranks divide num ranks", num_ranks % local_ranks == 0,
              f"num_ranks={num_ranks}, local_ranks={local_ranks}"),
        Check("fullmesh_world_size >= 1", bool(fullmesh_world_size and fullmesh_world_size >= 1),
              f"fullmesh_world_size={fullmesh_world_size}"),
    ]
    failed = False
    for check in final_checks:
        print(f"[{status(check):4}] {check.name}: {check.detail}")
        failed = failed or (not check.ok and not check.warn_only)

    if failed:
        print("\nResult: environment is likely to fail the current Metax AG GEMM intra-node path.")
        return 1

    print("\nResult: basic environment checks passed for the current AG GEMM intra-node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
