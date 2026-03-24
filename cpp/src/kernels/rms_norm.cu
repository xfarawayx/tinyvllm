// ---------------------------------------------------------------------------
// rms_norm.cu
//
// Fused RMSNorm CUDA kernel for tinyvllm.
//
// RMSNorm(x, weight, eps) = (x / sqrt(mean(x^2) + eps)) * weight
// ---------------------------------------------------------------------------

#include "tvllm/fused_kernels.h"
#include "tvllm/cuda_common.h"

namespace tvllm {
namespace cuda {

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ x,       // [num_rows, hidden_size]
    const scalar_t* __restrict__ weight,   // [hidden_size]
    scalar_t* __restrict__ output,         // [num_rows, hidden_size]
    int hidden_size,
    float eps) {
  const int row = blockIdx.x;
  const int warpId = threadIdx.x / kWarpSize;
  const int laneId = threadIdx.x % kWarpSize;

  float sum_sq = 0.0f;
  __shared__ float sum_sq_s[kWarpSize], rms_inv;

  for (int  i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(x[row * hidden_size + i]);
    sum_sq += val * val;
  }

  for (int s = kWarpSize >> 1; s > 0; s >>= 1) {
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, s);
  }

  if (laneId == 0) {
    sum_sq_s[warpId] = sum_sq;
  }
  __syncthreads();

  if (warpId == 0) {
    sum_sq = (threadIdx.x < (blockDim.x / kWarpSize)) ? sum_sq_s[laneId] : 0.0f;
    for (int s = kWarpSize >> 1; s > 0; s >>= 1) {
      sum_sq += __shfl_down_sync(0xffffffff, sum_sq, s);
    }
    if (laneId == 0) {
      rms_inv = rsqrtf(sum_sq / hidden_size + eps);
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float w = static_cast<float>(weight[i]);
    float val = static_cast<float>(x[row * hidden_size + i]);
    output[row * hidden_size + i] = static_cast<scalar_t>(val * rms_inv * w);
  }
}

// ============================================================================
//  Host entry point (launches fused CUDA kernel)
// ============================================================================
//
// --- libtorch fallback (kept for reference) ---
// auto x_f = x.to(torch::kFloat);
// auto variance = x_f.pow(2).mean(-1, true);
// auto normed = x_f * torch::rsqrt(variance + eps);
// auto y = normed * weight.to(torch::kFloat);
// return y.to(x.dtype());

torch::Tensor rms_norm(const torch::Tensor& x,
                       const torch::Tensor& weight,
                       double eps) {
  TORCH_CHECK(x.is_cuda(), "rms_norm: x must be on CUDA");
  TORCH_CHECK(weight.is_cuda(), "rms_norm: weight must be on CUDA");
  TORCH_CHECK(x.size(-1) == weight.size(0),
              "rms_norm: x last dim (", x.size(-1), ") != weight size (", weight.size(0), ")");
  TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kBFloat16,
              "rms_norm: x must be fp16 or bf16");
  
  auto x_flat = x.reshape({-1, x.size(-1)}).contiguous();
  int num_rows = x_flat.size(0);
  int hidden_size = x_flat.size(1);
  auto output = torch::empty_like(x_flat);

  constexpr int block_size = 256;
  dim3 grid(num_rows);
  dim3 block(block_size);

  TVLLM_DISPATCH_FLOAT16(x.scalar_type(), "rms_norm_kernel", [&] {
    rms_norm_kernel<scalar_t><<<grid, block>>>(
        x_flat.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        hidden_size,
        static_cast<float>(eps));
  });

  return output.reshape(x.sizes());
}

}  // namespace cuda
}  // namespace tvllm
