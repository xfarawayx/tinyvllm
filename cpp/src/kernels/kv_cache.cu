// ---------------------------------------------------------------------------
// kv_cache.cu
//
// Fused KV scatter kernel for decode.  Given a batch of K/V vectors and a
// precomputed slot_mapping, writes all sequences into the paged KV pool in
// a single kernel launch (replacing B * 2 tiny copy_() calls per layer).
// ---------------------------------------------------------------------------

#include "tvllm/fused_kernels.h"
#include "tvllm/cuda_common.h"

namespace tvllm {
namespace cuda {

template <typename scalar_t>
__global__ void scatter_kv_decode_kernel(
    const scalar_t* __restrict__ k_in,         // [B, stride_hd]
    const scalar_t* __restrict__ v_in,         // [B, stride_hd]
    scalar_t* __restrict__ k_pool,             // [total_slots, stride_hd]
    scalar_t* __restrict__ v_pool,             // [total_slots, stride_hd]
    const int32_t* __restrict__ slot_mapping,  // [B]
    int64_t B,
    int64_t stride_hd) {                       // num_kv_heads * head_dim

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * stride_hd;
  if (idx >= total) return;

  int64_t b  = idx / stride_hd;
  int64_t hd = idx % stride_hd;

  int64_t dst = static_cast<int64_t>(slot_mapping[b]) * stride_hd + hd;

  k_pool[dst] = k_in[idx];
  v_pool[dst] = v_in[idx];
}

// ============================================================================
//  Host entry point
// ============================================================================

void scatter_kv_decode(torch::Tensor& k_pool,
                       torch::Tensor& v_pool,
                       const torch::Tensor& k,
                       const torch::Tensor& v,
                       const torch::Tensor& slot_mapping) {
  TORCH_CHECK(k_pool.is_cuda(), "scatter_kv_decode: k_pool must be on CUDA");
  TORCH_CHECK(k.is_cuda(), "scatter_kv_decode: k must be on CUDA");
  TORCH_CHECK(slot_mapping.is_cuda(), "scatter_kv_decode: slot_mapping must be on CUDA");
  TORCH_CHECK(slot_mapping.dtype() == torch::kInt32,
              "scatter_kv_decode: slot_mapping must be int32");

  // k/v: [B, 1, Hkv, D] -> flatten to [B, Hkv * D]
  int64_t B   = k.size(0);
  int64_t Hkv = k.size(2);
  int64_t D   = k.size(3);
  int64_t stride_hd = Hkv * D;

  auto k_flat = k.reshape({B, stride_hd}).contiguous();
  auto v_flat = v.reshape({B, stride_hd}).contiguous();

  // k_pool/v_pool: [num_blocks, block_size, Hkv, D] -> [num_blocks * block_size, Hkv * D]
  auto k_pool_flat = k_pool.reshape({-1, stride_hd});
  auto v_pool_flat = v_pool.reshape({-1, stride_hd});

  int64_t total = B * stride_hd;
  constexpr int block_size = 256;
  int64_t grid_size = (total + block_size - 1) / block_size;

  TVLLM_DISPATCH_FLOAT16(k.scalar_type(), "scatter_kv_decode_kernel", [&] {
    scatter_kv_decode_kernel<scalar_t><<<grid_size, block_size>>>(
        k_flat.data_ptr<scalar_t>(),
        v_flat.data_ptr<scalar_t>(),
        k_pool_flat.data_ptr<scalar_t>(),
        v_pool_flat.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int32_t>(),
        B,
        stride_hd);
  });
}

}  // namespace cuda
}  // namespace tvllm
