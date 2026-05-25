#!/bin/bash
export MXSHMEM_BOOTSTRAP=UID
#export MXSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=en,eth0,em,bond
#export MACA_LAUNCH_BLOCKING=0
#export MCCL_DEBUG=INFO
#export MXSHMEM_SYMMETRIC_SIZE=${MXSHMEM_SYMMETRIC_SIZE:-1000000000}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-~/.triton}
rm -rf ${TRITON_CACHE_DIR}

nproc_per_node=${ARNOLD_WORKER_GPU:=$(mx-smi --list | grep "GPU.*UUID" | wc -l)}
nnodes=${ARNOLD_WORKER_NUM:=1}
node_rank=${ARNOLD_ID:=0}

if [ ${nnodes} >= 1 ]; then
  export MXSHMEM_IB_ENABLE_IBGDA=1
  export MXSHMEM_IB_ENABLE_IBRC=0
fi

master_addr=${ARNOLD_WORKER_0_HOST:="127.0.0.1"}
if [ -z ${ARNOLD_WORKER_0_PORT} ]; then
  master_port="23457"
else
  master_port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
fi

additional_args="--rdzv_endpoint=${master_addr}:${master_port}"

CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${additional_args} \
  $@"

echo ${CMD}
${CMD}

ret=$?
exit $ret
