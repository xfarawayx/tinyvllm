#pragma once
// ---------------------------------------------------------------------------
// fused_kernels.h
//
// Host-side declarations for fused CUDA kernels used in the transformer
// forward pass.  These replace multi-op libtorch sequences with single
// kernel launches to reduce launch overhead and memory traffic.
//
//   rms_norm   – fused RMSNorm (cast + pow + mean + rsqrt + scale)
//   apply_rope – fused Rotary Position Embedding for Q and K
//   silu_mul   – fused SiLU(gate) * up  (aka SwiGLU activation)
// ---------------------------------------------------------------------------

#include <torch/torch.h>

namespace tvllm {
namespace cuda {

// ---------- Fused RMSNorm ------------------------------------------------
//
// Computes:  y = (x / sqrt(mean(x^2) + eps)) * weight
// Internal accumulation in fp32 for numerical stability.
//
// Parameters
// ----------
// x      : [*, hidden_size]  (fp16/bf16)  – input tensor (any leading dims)
// weight : [hidden_size]     (fp16/bf16)  – learnable scale
// eps    : RMSNorm epsilon (typically 1e-5 or 1e-6)
//
// Returns
// -------
// output : [*, hidden_size]  same dtype as x
//
torch::Tensor rms_norm(const torch::Tensor& x,
                       const torch::Tensor& weight,
                       double eps);

// ---------- Fused Residual-Add + RMSNorm ---------------------------------
//
// Computes in a single kernel:
//   residual += x            (in-place update)
//   output = rms_norm(residual, weight, eps)
//
// Parameters
// ----------
// residual : [*, hidden_size]  (fp16/bf16)  – updated in-place
// x        : [*, hidden_size]  (fp16/bf16)  – addend
// weight   : [hidden_size]     (fp16/bf16)  – RMSNorm scale
// eps      : RMSNorm epsilon
//
// Returns
// -------
// output : [*, hidden_size]  same dtype as residual
//
torch::Tensor fused_add_rms_norm(torch::Tensor& residual,
                                 const torch::Tensor& x,
                                 const torch::Tensor& weight,
                                 double eps);

// ---------- Fused Rotary Position Embedding (RoPE) -----------------------
//
// Applies rotary embeddings to Q and K tensors **in-place**.
//
// For each position p and dimension pair (d, d + head_dim/2):
//   x_new[d]              = x[d] * cos[p,d] - x[d + half] * sin[p,d]
//   x_new[d + half]       = x[d + half] * cos[p,d] + x[d] * sin[p,d]
//
// Parameters
// ----------
// q         : [total_tokens, num_heads, head_dim]     (fp16/bf16)  – modified in-place
// k         : [total_tokens, num_kv_heads, head_dim]  (fp16/bf16)  – modified in-place
// positions : [total_tokens] or [batch, seq_len]      (int64)
// cos_table : [max_position, head_dim]                (fp16/bf16)  – precomputed cos
// sin_table : [max_position, head_dim]                (fp16/bf16)  – precomputed sin
//
void apply_rope(torch::Tensor& q,
                torch::Tensor& k,
                const torch::Tensor& positions,
                const torch::Tensor& cos_table,
                const torch::Tensor& sin_table);

// ---------- Fused SiLU·gate (SwiGLU activation) -------------------------
//
// Given a concatenated [gate, up] tensor along the last dimension, computes:
//   output = silu(gate) * up
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
//
// Parameters
// ----------
// gate_up : [*, 2 * intermediate_size]  (fp16/bf16)
//           First half along last dim is gate, second half is up.
//
// Returns
// -------
// output : [*, intermediate_size]  same dtype as gate_up
//
torch::Tensor silu_mul(const torch::Tensor& gate_up);

// ---------- Fused KV scatter -----------------------------------------------
//
// Scatters K and V tokens into paged KV cache blocks using a precomputed
// slot_mapping.  Accepts any input rank whose last two dims are [Hkv, D]
// (e.g. [N, Hkv, D] for prefill or [B, 1, Hkv, D] for decode).
// A -1 sentinel in slot_mapping causes the corresponding token to be skipped
// (used for prefix-cached blocks).
//
// Parameters
// ----------
// k_pool      : [num_blocks, block_size, num_kv_heads, head_dim]  – destination (modified)
// v_pool      : [num_blocks, block_size, num_kv_heads, head_dim]  – destination (modified)
// k           : [*, num_kv_heads, head_dim]  (fp16/bf16)  – source
// v           : [*, num_kv_heads, head_dim]  (fp16/bf16)  – source
// slot_mapping: [N]  (int32)  – flat slot index per token (-1 = skip)
//
void scatter_kv(torch::Tensor& k_pool,
                torch::Tensor& v_pool,
                const torch::Tensor& k,
                const torch::Tensor& v,
                const torch::Tensor& slot_mapping);

// ---------- NF4 Dequantization -----------------------------------------------
//
// Dequantizes NF4 double-quantized weights on-the-fly.
//
// Parameters
// ----------
// packed_weights : [n_elements/2]  (uint8)  – 2 x 4-bit indices per byte
// absmax_q       : [num_blocks]    (uint8)  – level-1 quantized scaling indices
// absmax2        : [num_groups]    (fp16)   – level-2 scaling factors
// code2          : [256]           (fp16)   – level-2 codebook
// offset         : global quantization offset (float)
// blocksize      : quantization block size (must be power of 2)
// s2_blocksize   : secondary block size (must be power of 2)
// num_rows       : output rows (M)
// num_cols       : output cols (N)
// out_dtype      : desired output dtype (fp16 or bf16)
//
// Returns
// -------
// output : [num_rows, num_cols]  in out_dtype
//
torch::Tensor nf4_dequantize(
    const torch::Tensor& packed_weights,
    const torch::Tensor& absmax_q,
    const torch::Tensor& absmax2,
    const torch::Tensor& code2,
    float offset,
    int blocksize,
    int s2_blocksize,
    int64_t num_rows,
    int64_t num_cols,
    torch::Dtype out_dtype);

// ---------- NF4 Linear (dequant + GEMM) --------------------------------------
//
// Dequantizes the full [N, K] NF4 weight matrix and multiplies with input.
//
// Parameters
// ----------
// input          : [*, K]           (fp16/bf16) – activation tensor
// packed_weights : [n_elements/2]   (uint8)
// absmax_q       : [num_blocks]     (uint8)
// absmax2        : [num_groups]     (fp16)
// code2          : [256]            (fp16)
// offset         : global quantization offset
// blocksize      : quantization block size (power of 2)
// s2_blocksize   : secondary block size (power of 2)
// num_rows       : weight matrix rows  (N, i.e. out_features)
// num_cols       : weight matrix cols  (K, i.e. in_features)
//
// Returns
// -------
// output : [*, N]  same dtype as input
//
torch::Tensor nf4_linear(
    const torch::Tensor& input,
    const torch::Tensor& packed_weights,
    const torch::Tensor& absmax_q,
    const torch::Tensor& absmax2,
    const torch::Tensor& code2,
    float offset,
    int blocksize,
    int s2_blocksize,
    int64_t num_rows,
    int64_t num_cols);

}  // namespace cuda
}  // namespace tvllm
