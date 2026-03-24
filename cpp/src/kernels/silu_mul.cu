// ---------------------------------------------------------------------------
// silu_mul.cu
//
// Fused SiLU·gate (SwiGLU activation) CUDA kernel for tinyvllm.
//
// Given a concatenated [gate, up] tensor along the last dimension:
//   output = silu(gate) * up
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ---------------------------------------------------------------------------

#include "tvllm/fused_kernels.h"
#include "tvllm/cuda_common.h"

namespace tvllm {
namespace cuda {

template <typename scalar_t>
__global__ void silu_mul_kernel(
    const scalar_t* __restrict__ gate_up,  // [num_rows, 2 * D]
    scalar_t* __restrict__ output,          // [num_rows, D]
    int64_t num_elements,
    int64_t last_dim_half) {

  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_elements) return;

  int64_t row = i / last_dim_half;
  int64_t col = i % last_dim_half;
  int64_t base = row * (2 * last_dim_half);

  float gate_val = static_cast<float>(gate_up[base + col]);
  float up_val = static_cast<float>(gate_up[base + col + last_dim_half]);
  float silu_val = gate_val / (1.0f + expf(-gate_val));
  output[i] = static_cast<scalar_t>(silu_val * up_val);
}

// ============================================================================
//  Host entry point (launches fused CUDA kernel)
// ============================================================================
//
// --- libtorch fallback (kept for reference) ---
// auto chunks = gate_up.chunk(2, -1);
// return torch::silu(chunks[0]) * chunks[1];

torch::Tensor silu_mul(const torch::Tensor& gate_up) {
  TORCH_CHECK(gate_up.is_cuda(), "silu_mul: gate_up must be on CUDA");
  TORCH_CHECK(gate_up.dtype() == torch::kHalf || gate_up.dtype() == torch::kBFloat16,
              "silu_mul: gate_up must be fp16 or bf16");
  TORCH_CHECK(gate_up.size(-1) % 2 == 0,
              "silu_mul: last dim must be even (gate + up concatenated)");

  auto input = gate_up.contiguous();
  int64_t last_dim_half = input.size(-1) / 2;

  // Output shape: same as input but with last dim halved
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = last_dim_half;
  auto output = torch::empty(out_sizes, input.options());

  int64_t num_elements = output.numel();

  constexpr int block_size = 256;
  int64_t grid_size = (num_elements + block_size - 1) / block_size;

  TVLLM_DISPATCH_FLOAT16(input.scalar_type(), "silu_mul_kernel", [&] {
    silu_mul_kernel<scalar_t><<<grid_size, block_size>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        num_elements,
        last_dim_half);
  });

  return output;
}

}  // namespace cuda
}  // namespace tvllm
