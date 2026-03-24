cmake_minimum_required(VERSION 3.18)
project(tinyvllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architectures (A100 = 80)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(CUDAToolkit REQUIRED)

set(TVLLM_KERNEL_SOURCES
  cpp/src/kernels/rms_norm.cu
  cpp/src/kernels/rope.cu
  cpp/src/kernels/silu_mul.cu
  cpp/src/kernels/kv_cache.cu
  cpp/src/kernels/nf4_dequant.cu
)

# ===========================================================================
#  Main shared library (Python module)
# ===========================================================================
add_library(tinyvllm SHARED
  cpp/src/config.cpp
  cpp/src/block_manager.cpp
  cpp/src/kv_cache.cpp
  cpp/src/attention.cpp
  ${TVLLM_KERNEL_SOURCES}
  cpp/src/model.cpp
  cpp/src/sampler.cpp
  cpp/src/scheduler.cpp
  cpp/src/engine.cpp
  cpp/src/pybind.cpp
)

target_include_directories(tinyvllm PRIVATE
  cpp/include
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_INSTALL_PREFIX}/include
  ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include
)

target_link_libraries(tinyvllm PRIVATE
  ${TORCH_LIBRARIES}
  Python3::Python
  CUDA::cudart
)

set(TVLLM_TORCH_PYTHON "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")
if(EXISTS "${TVLLM_TORCH_PYTHON}")
  target_link_libraries(tinyvllm PRIVATE "${TVLLM_TORCH_PYTHON}")
endif()

set_target_properties(tinyvllm PROPERTIES PREFIX "" OUTPUT_NAME "tinyvllm")
