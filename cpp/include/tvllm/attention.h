#pragma once
// ---------------------------------------------------------------------------
// attention.h
//
// Attention interfaces for tinyvllm.
//
// Runtime prefill/decode use FlashInfer-backed wrappers in cpp/src/attention.cpp.
// The original handwritten CUDA implementation lives in
// cpp/src/kernels/attention_fallback.cu and is kept as a test-only fallback.
//
// Two entry‑points modeled after Flash‑Attention:
//   1) flash_attn_varlen  – variable‑length batched prefill  (cf. flash_attn_varlen_func)
//   2) flash_attn_decode  – paged KV‑cache decode            (cf. flash_attn_with_kvcache)
// ---------------------------------------------------------------------------

#include <torch/torch.h>

namespace tvllm {
namespace cuda {

// ---------- Prefill (variable‑length, flash‑attention style) ----------------
//
// Computes multi‑head attention for a batch of variable‑length sequences that
// have been packed (concatenated) into 1‑D tensors.  GQA is handled natively.
//
// Parameters
// ----------
// q             : [total_q_tokens, num_heads, head_dim]        (fp16/bf16)
// k             : [total_kv_tokens, num_kv_heads, head_dim]    (fp16/bf16)
// v             : [total_kv_tokens, num_kv_heads, head_dim]    (fp16/bf16)
// cu_seqlens_q  : [batch_size + 1]  int32  – cumulative token counts for Q
// cu_seqlens_k  : [batch_size + 1]  int32  – cumulative token counts for K/V
// max_seqlen_q  : scalar – max sequence length in the Q side
// max_seqlen_k  : scalar – max sequence length in the K/V side
// softmax_scale : 1/sqrt(head_dim) if <= 0
// causal        : whether to apply causal masking
//
// Returns
// -------
// output : [total_q_tokens, num_heads, head_dim]  same dtype as q
//
torch::Tensor flash_attn_varlen(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float softmax_scale = 0.0f,
    bool causal = true);

// ---------- Decode (paged KV‑cache, single new token per seq) ---------------
//
// Split into plan + run so the caller can plan once outside the layer loop
// (avoiding 28× GPU→CPU syncs and FlashInfer plan() calls) and only run
// inside the per‑layer loop.
//
// flash_attn_decode_plan:
//   Converts block_tables + context_lens to FlashInfer metadata and calls
//   FlashInfer plan().  CPU int32 metadata tensors are preferred to avoid
//   unnecessary device round-trips. Must be called once before the layer loop.
//
// flash_attn_decode_run:
//   Executes FlashInfer decode kernel for one layer.  Assumes plan() was
//   already called.
//
// flash_attn_decode:
//   Convenience wrapper that calls plan + run in one shot (for tests / back‑compat).
//

void flash_attn_decode_plan(
    const torch::Tensor& q_any,       // only used for dtype/device inference
    const torch::Tensor& k_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const torch::Tensor& block_tables, // [batch_size, max_num_blocks_per_seq] int32, preferably CPU
    const torch::Tensor& context_lens, // [batch_size] int32, preferably CPU
    float softmax_scale = 0.0f,
    int64_t block_size = 16);

torch::Tensor flash_attn_decode_run(
    const torch::Tensor& q,           // [batch_size, 1, num_heads, head_dim]
    const torch::Tensor& k_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const torch::Tensor& v_cache);    // [num_blocks, block_size, num_kv_heads, head_dim]

torch::Tensor flash_attn_decode(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    float softmax_scale = 0.0f,
    int64_t block_size = 16);

// ---------- Prefill with paged KV cache (for prefix caching) -----------------
//
// New query tokens attend to both cached KV (via paged block tables) and
// themselves.  Uses FlashInfer BatchPrefillWithPagedKVCacheWrapper.
//
// Split into plan + run following the same pattern as decode.
//
// flash_attn_prefill_paged_plan:
//   Converts block_tables + context_lens to FlashInfer metadata and calls
//   plan().  Must be called once before the layer loop.
//
// flash_attn_prefill_paged_run:
//   Executes FlashInfer paged-prefill kernel for one layer.

void flash_attn_prefill_paged_plan(
    const torch::Tensor& q_any,       // only for dtype/device inference [total_q, num_heads, head_dim]
    const torch::Tensor& k_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const torch::Tensor& cu_seqlens_q, // [batch_size + 1] int32 — cumulative Q token counts
    const torch::Tensor& block_tables, // [batch_size, max_num_blocks] int32
    const torch::Tensor& context_lens, // [batch_size] int32 — total KV len per sequence
    float softmax_scale = 0.0f,
    int64_t block_size = 16,
    bool causal = true);

torch::Tensor flash_attn_prefill_paged_run(
    const torch::Tensor& q,           // [total_q_tokens, num_heads, head_dim]
    const torch::Tensor& k_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const torch::Tensor& v_cache);    // [num_blocks, block_size, num_kv_heads, head_dim]

}  // namespace cuda
}  // namespace tvllm
