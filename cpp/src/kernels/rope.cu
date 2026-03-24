// ---------------------------------------------------------------------------
// rope.cu
//
// Fused Rotary Position Embedding (RoPE) CUDA kernel for tinyvllm.
//
// Applies rotary embeddings to Q and K tensors in-place.
// For each position p and dimension pair (d, d + half):
//   x_new[d]        = x[d] * cos[p,d] - x[d+half] * sin[p,d]
//   x_new[d+half]   = x[d+half] * cos[p,d] + x[d] * sin[p,d]
// ---------------------------------------------------------------------------

#include "tvllm/fused_kernels.h"
#include "tvllm/cuda_common.h"

namespace tvllm {
namespace cuda {

// ============================================================================
//  CUDA kernel
// ============================================================================
//
// Grid:  (total_tokens, num_heads_q + num_heads_k)
//        First num_heads_q columns operate on Q, the rest on K.
// Block: (block_size,)  — threads stride over head_dim/2 pairs.

template <typename scalar_t>
__global__ void apply_rope_kernel(
    scalar_t* __restrict__ q,              // [total_tokens, num_heads_q, head_dim]
    scalar_t* __restrict__ k,              // [total_tokens, num_heads_k, head_dim]
    const int64_t* __restrict__ positions, // [total_tokens]
    const scalar_t* __restrict__ cos_table, // [max_pos, head_dim]
    const scalar_t* __restrict__ sin_table, // [max_pos, head_dim]
    int num_heads_q,
    int num_heads_k,
    int head_dim) {

  const int token_idx = blockIdx.x;
  const int head_idx  = blockIdx.y;
  const int half      = head_dim / 2;

  // Determine whether this block works on Q or K
  const bool is_q = (head_idx < num_heads_q);
  const int local_head = is_q ? head_idx : (head_idx - num_heads_q);
  const int num_heads  = is_q ? num_heads_q : num_heads_k;

  scalar_t* x = is_q ? q : k;

  // Base offset for this token + head
  const int64_t x_base = (int64_t)token_idx * num_heads * head_dim
                        + (int64_t)local_head * head_dim;

  // Position-based cos/sin offset
  const int64_t pos = positions[token_idx];
  const int64_t cs_base = pos * head_dim;

  // Each thread handles one or more dimension pairs via stride loop
  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    float cos_val = static_cast<float>(cos_table[cs_base + d]);
    float sin_val = static_cast<float>(sin_table[cs_base + d]);

    float x_d    = static_cast<float>(x[x_base + d]);
    float x_dhalf = static_cast<float>(x[x_base + d + half]);

    x[x_base + d]        = static_cast<scalar_t>(x_d * cos_val - x_dhalf * sin_val);
    x[x_base + d + half] = static_cast<scalar_t>(x_dhalf * cos_val + x_d * sin_val);
  }
}

// ============================================================================
//  Host entry point (launches fused CUDA kernel)
// ============================================================================
//
// --- libtorch fallback (kept for reference) ---
// auto pos_flat = positions.reshape({-1}).to(torch::kLong);
// auto cos = cos_table.index_select(0, pos_flat);
// auto sin = sin_table.index_select(0, pos_flat);
// auto num_tokens = pos_flat.size(0);
// auto head_dim = q.size(-1);
// cos = cos.view({num_tokens, 1, head_dim});
// sin = sin.view({num_tokens, 1, head_dim});
// auto rotate_half = [](const torch::Tensor& x) {
//   int64_t half = x.size(-1) / 2;
//   auto x1 = x.narrow(-1, 0, half);
//   auto x2 = x.narrow(-1, half, half);
//   return torch::cat({-x2, x1}, -1);
// };
// q = q * cos + rotate_half(q) * sin;
// k = k * cos + rotate_half(k) * sin;

void apply_rope(torch::Tensor& q,
                torch::Tensor& k,
                const torch::Tensor& positions,
                const torch::Tensor& cos_table,
                const torch::Tensor& sin_table) {
  TORCH_CHECK(q.is_cuda(), "apply_rope: q must be on CUDA");
  TORCH_CHECK(k.is_cuda(), "apply_rope: k must be on CUDA");
  TORCH_CHECK(positions.is_cuda(), "apply_rope: positions must be on CUDA");
  TORCH_CHECK(cos_table.is_cuda(), "apply_rope: cos_table must be on CUDA");
  TORCH_CHECK(sin_table.is_cuda(), "apply_rope: sin_table must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kHalf || q.dtype() == torch::kBFloat16,
              "apply_rope: q must be fp16 or bf16");
  TORCH_CHECK(q.size(-1) == k.size(-1),
              "apply_rope: q and k must have the same head_dim");

  // Save original shapes (may be 3D or 4D) for reshape-back
  auto q_sizes = q.sizes().vec();
  auto k_sizes = k.sizes().vec();

  int head_dim    = q.size(-1);
  int num_heads_q = q.size(-2);
  int num_heads_k = k.size(-2);

  // Flatten to 3D: [total_tokens, num_heads, head_dim]
  auto q_c = q.reshape({-1, num_heads_q, head_dim}).contiguous();
  auto k_c = k.reshape({-1, num_heads_k, head_dim}).contiguous();
  auto pos_flat = positions.reshape({-1}).contiguous();

  int total_tokens = q_c.size(0);

  constexpr int block_size = 128;
  dim3 grid(total_tokens, num_heads_q + num_heads_k);
  dim3 block(block_size);

  TVLLM_DISPATCH_FLOAT16(q.scalar_type(), "apply_rope_kernel", [&] {
    apply_rope_kernel<scalar_t><<<grid, block>>>(
        q_c.data_ptr<scalar_t>(),
        k_c.data_ptr<scalar_t>(),
        pos_flat.data_ptr<int64_t>(),
        cos_table.data_ptr<scalar_t>(),
        sin_table.data_ptr<scalar_t>(),
        num_heads_q,
        num_heads_k,
        head_dim);
  });
  TVLLM_CUDA_CHECK(cudaGetLastError());

  // Reshape back to original shape and rebind references
  q = q_c.reshape(q_sizes);
  k = k_c.reshape(k_sizes);
}

}  // namespace cuda
}  // namespace tvllm
