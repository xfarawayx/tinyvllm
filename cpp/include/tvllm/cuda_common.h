#pragma once
// ---------------------------------------------------------------------------
// cuda_common.h
//
// Shared CUDA utilities for tinyvllm kernels.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace tvllm {
namespace cuda {

// ===========================  error checking  ================================

#define TVLLM_CUDA_CHECK(expr)                                        \
  do {                                                                \
    cudaError_t __err = (expr);                                       \
    if (__err != cudaSuccess) {                                       \
      throw std::runtime_error(std::string("CUDA error: ") +         \
                               cudaGetErrorString(__err));            \
    }                                                                 \
  } while (0)

// ===========================  constants  =====================================

constexpr int kWarpSize = 32;

// ===========================  dtype dispatch  ================================

// Dispatch over fp16 and bf16 (the two dtypes supported by tinyvllm).
#define TVLLM_DISPATCH_FLOAT16(DTYPE, NAME, ...)                      \
  AT_DISPATCH_SWITCH(DTYPE, NAME,                                     \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)               \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

}  // namespace cuda
}  // namespace tvllm
