// ---------------------------------------------------------------------------
// kv_cache.cu
//
// Fused KV scatter kernel.  Given a batch of K/V vectors and a precomputed
// slot_mapping, writes all tokens into the paged KV pool in a single kernel
// launch.  Supports both prefill ([N, Hkv, D]) and decode ([B, 1, Hkv, D])
// inputs, and a -1 sentinel in slot_mapping to skip prefix-cached blocks.
// ---------------------------------------------------------------------------

#include "tvllm/fused_kernels.h"
#include "tvllm/cuda_common.h"

namespace tvllm {
namespace cuda {

template <typename scalar_t>
__global__ void scatter_kv_kernel(
    const scalar_t* __restrict__ k_in,         // [N, stride_hd]
    const scalar_t* __restrict__ v_in,         // [N, stride_hd]
    scalar_t* __restrict__ k_pool,             // [total_slots, stride_hd]
    scalar_t* __restrict__ v_pool,             // [total_slots, stride_hd]
    const int32_t* __restrict__ slot_mapping,  // [N]  (-1 = skip)
    int64_t N,
    int64_t stride_hd) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = N * stride_hd;
  if (idx >= total) return;

  int64_t n  = idx / stride_hd;
  int32_t slot = slot_mapping[n];
  if (slot < 0) return;  // skip sentinel

  int64_t hd  = idx % stride_hd;
  int64_t dst = static_cast<int64_t>(slot) * stride_hd + hd;

  k_pool[dst] = k_in[idx];
  v_pool[dst] = v_in[idx];
}

// ============================================================================
//  Host entry point
// ============================================================================

void scatter_kv(torch::Tensor& k_pool,
                torch::Tensor& v_pool,
                const torch::Tensor& k,
                const torch::Tensor& v,
                const torch::Tensor& slot_mapping) {
  TORCH_CHECK(k_pool.is_cuda(), "scatter_kv: k_pool must be on CUDA");
  TORCH_CHECK(k.is_cuda(), "scatter_kv: k must be on CUDA");
  TORCH_CHECK(slot_mapping.is_cuda(), "scatter_kv: slot_mapping must be on CUDA");
  TORCH_CHECK(slot_mapping.dtype() == torch::kInt32,
              "scatter_kv: slot_mapping must be int32");

  // Accept any rank: flatten leading dims → [N, Hkv * D].
  // e.g. [N, Hkv, D] or [B, 1, Hkv, D] both work.
  int64_t Hkv = k.size(-2);
  int64_t D   = k.size(-1);
  int64_t stride_hd = Hkv * D;
  int64_t N   = k.numel() / stride_hd;

  auto k_flat = k.reshape({N, stride_hd}).contiguous();
  auto v_flat = v.reshape({N, stride_hd}).contiguous();

  // k_pool/v_pool: [num_blocks, block_size, Hkv, D] -> [total_slots, Hkv * D]
  auto k_pool_flat = k_pool.reshape({-1, stride_hd});
  auto v_pool_flat = v_pool.reshape({-1, stride_hd});

  int64_t total = N * stride_hd;
  constexpr int threads = 256;
  int64_t grid_size = (total + threads - 1) / threads;

  TVLLM_DISPATCH_FLOAT16(k.scalar_type(), "scatter_kv_kernel", [&] {
    scatter_kv_kernel<scalar_t><<<grid_size, threads>>>(
        k_flat.data_ptr<scalar_t>(),
        v_flat.data_ptr<scalar_t>(),
        k_pool_flat.data_ptr<scalar_t>(),
        v_pool_flat.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int32_t>(),
        N,
        stride_hd);
  });
}

}  // namespace cuda
}  // namespace tvllm
