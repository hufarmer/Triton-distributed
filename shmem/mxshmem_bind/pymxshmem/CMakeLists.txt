cmake_minimum_required(VERSION 3.12)

project(pvmxshmem LANGUAGES CXX CUDA)

message(FATAL_ERROR "camake start")
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c"
          "from __future__ import print_function; import os; import pybind11;
print(os.path.dirname(pybind11.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PYBIND11_DIR)
message("PYTHON_EXECUTABLE:${PYTHON_EXECUTABLE}")
if(NOT _PYTHON_SUCCESS MATCHES 0)
  message("PYBIND11_DIR: ${PYBIND11_DIR}")
  message(FATAL_ERROR "Pybind11 config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
find_package(pybind11 REQUIRED)

# find torch
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c"
          "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_DIR)
if(NOT _PYTHON_SUCCESS MATCHES 0)
  message("PY:${PYTHONPATH}")
  message(FATAL_ERROR "Torch config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_DIR}/lib")

if(TORCH_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

find_package(MXSHMEM REQUIRED)

set(SOURCES
    src/pvmxshmem.cc
    src/flush_l2c.cu
)

pybind11_add_module(pvmxshmem ${SOURCES})

include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")

set_target_properties(pvmxshmem PROPERTIES CXX_STANDARD 17
                                           CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#find_library(MXSHMEM_HOST_LIB
#    NAMES mxshmem_host  # 对应libmxshmem_host.so
#    PATHS /home/jjiang/code/tmp_triton/third_party/triton_dist/3rdparty/mxshmem/build  # 库所在的目录（若tmp是项目内子目录，用${CMAKE_SOURCE_DIR}/tmp）
#    NO_DEFAULT_PATH  # 仅搜索指定的PATHS，不搜索默认系统路径
#    REQUIRED  # 找不到库则直接报错
#)

# 2. 同理查找libmxshmem_device.so（若它也在tmp目录）
#find_library(MXSHMEM_DEVICE_LIB
#    NAMES mxshmem_device
#    PATHS /home/jjiang/code/tmp_triton/third_party/triton_dist/3rdparty/mxshmem/build
#    NO_DEFAULT_PATH
#    REQUIRED
#)
#message(FATAL_ERROR "MXSHMEM_HOST_LIB: ${MXSHMEM_HOST_LIB}")
#message(FATAL_ERROR "MXSHMEM_DEVICE_LIB: ${MXSHMEM_DEVICE_LIB}")
# 3. 修改target_link_libraries，替换原来的mxshmem::mxshmem_host为找到的库
#target_link_libraries(pvmxshmem PRIVATE
#    ${MXSHMEM_HOST_LIB}    # 用找到的tmp目录下的libmxshmem_host.so
#    ${MXSHMEM_DEVICE_LIB}  # 同理处理device库
#    torch
#    ${TORCH_PYTHON_LIBRARY}
#)

#set(MXSHMEM_INCLUDE_DIRS "/home/jjiang/code/tmp_triton/third_party/triton_dist/3rdparty/mxshmem/src/include")
#message("MXSHMEM_INCLUDE_DIRS:$MXSHMEM_INCLUDE_DIRS")

target_link_libraries(pvmxshmem PRIVATE mxshmem::mxshmem_host
                                        mxshmem::mxshmem_device
                                        torch
                                        ${TORCH_PYTHON_LIBRARY})

target_include_directories(pvmxshmem PRIVATE ${MXSHMEM_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_compile_options(pvmxshmem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
target_compile_options(pvmxshmem PRIVATE "-DUSE_MACA")
