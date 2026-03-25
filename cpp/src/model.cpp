#include "tvllm/model.h"

#include <stdexcept>

#include "tvllm/attention.h"
#include "tvllm/kv_cache.h"
#include "tvllm/fused_kernels.h"

namespace tvllm {

using torch::indexing::Slice;

// ---------------------------------------------------------------------------
//  Tensor helpers
// ---------------------------------------------------------------------------

static c10::optional<torch::Tensor> get_optional_tensor(
    const c10::impl::GenericDict& dict,
    const std::string& key,
    torch::Device device,
    torch::Dtype dtype) {
  auto it = dict.find(c10::IValue(key));
  if (it == dict.end()) {
    return c10::nullopt;
  }
  return it->value().toTensor().to(device, dtype);
}

static torch::Tensor get_tensor(const c10::impl::GenericDict& dict,
                                const std::string& key,
                                torch::Device device,
                                torch::Dtype dtype) {
  auto t = get_optional_tensor(dict, key, device, dtype);
  if (!t.has_value()) {
    throw std::runtime_error("missing weight: " + key);
  }
  return *t;
}

// Raw tensor without dtype conversion (for NF4 uint8 components)
static torch::Tensor get_tensor_raw(const c10::impl::GenericDict& dict,
                                    const std::string& key,
                                    torch::Device device) {
  auto it = dict.find(c10::IValue(key));
  if (it == dict.end()) {
    throw std::runtime_error("missing weight: " + key);
  }
  return it->value().toTensor().to(device);
}

// ---------------------------------------------------------------------------
//  Linear weight loading
// ---------------------------------------------------------------------------

static LinearWeight load_linear(const c10::impl::GenericDict& dict,
                                const std::string& prefix,
                                torch::Device device,
                                torch::Dtype dtype,
                                bool use_bias) {
  LinearWeight lw;
  lw.weight = get_tensor(dict, prefix + ".weight", device, dtype);
  if (use_bias) {
    lw.bias = get_optional_tensor(dict, prefix + ".bias", device, dtype);
  }
  return lw;
}

static LinearWeight load_nf4_linear(const c10::impl::GenericDict& dict,
                                    const std::string& prefix,
                                    torch::Device device,
                                    torch::Dtype dtype,
                                    bool use_bias) {
  LinearWeight lw;
  NF4Weight nf4;
  nf4.packed_weights = get_tensor_raw(dict, prefix + ".packed_weights", device);
  nf4.absmax_q = get_tensor_raw(dict, prefix + ".absmax_q", device);
  nf4.absmax2 = get_tensor_raw(dict, prefix + ".absmax2", device);
  nf4.code2 = get_tensor_raw(dict, prefix + ".code2", device);

  // Scalar metadata stored as 1-element tensors
  auto offset_t = get_tensor_raw(dict, prefix + ".offset", device);
  nf4.offset = offset_t.item<float>();
  auto bs_t = get_tensor_raw(dict, prefix + ".blocksize", device);
  nf4.blocksize = bs_t.item<int>();
  auto s2_t = get_tensor_raw(dict, prefix + ".s2_blocksize", device);
  nf4.s2_blocksize = s2_t.item<int>();
  auto nr_t = get_tensor_raw(dict, prefix + ".num_rows", device);
  nf4.num_rows = nr_t.item<int64_t>();
  auto nc_t = get_tensor_raw(dict, prefix + ".num_cols", device);
  nf4.num_cols = nc_t.item<int64_t>();

  lw.nf4 = std::move(nf4);

  if (use_bias) {
    lw.bias = get_optional_tensor(dict, prefix + ".bias", device, dtype);
  }
  return lw;
}

// ---------------------------------------------------------------------------
//  linear() — unified for fp16 and NF4
// ---------------------------------------------------------------------------

static torch::Tensor linear(const torch::Tensor& x, const LinearWeight& lw) {
  torch::Tensor y;
  if (lw.nf4.has_value()) {
    const auto& nf4 = *lw.nf4;
    y = cuda::nf4_linear_tiled(x, nf4.packed_weights, nf4.absmax_q,
                                nf4.absmax2, nf4.code2, nf4.offset,
                                nf4.blocksize, nf4.s2_blocksize,
                                nf4.num_rows, nf4.num_cols);
  } else {
    y = torch::matmul(x, lw.weight.t());
  }
  if (lw.bias.has_value()) {
    y = y + lw.bias.value();
  }
  return y;
}

// ---------------------------------------------------------------------------
//  QwenModel constructor
// ---------------------------------------------------------------------------

QwenModel::QwenModel(const ModelConfig& config,
                     const c10::impl::GenericDict& state_dict,
                     torch::Device device,
                     torch::Dtype dtype)
    : config_(config), device_(device), dtype_(dtype) {
  head_dim_ = (config_.head_dim > 0)
      ? config_.head_dim
      : config_.hidden_size / config_.num_attention_heads;

  tok_embeddings_ = get_tensor(state_dict, "tok_embeddings.weight", device_, dtype_);
  norm_weight_ = get_tensor(state_dict, "norm.weight", device_, dtype_);

  if (state_dict.find(c10::IValue("lm_head.weight")) != state_dict.end()) {
    lm_head_weight_ = get_tensor(state_dict, "lm_head.weight", device_, dtype_);
  } else {
    lm_head_weight_ = tok_embeddings_;
  }

  auto inv_freq = torch::arange(0, head_dim_, 2, torch::TensorOptions().device(device_).dtype(torch::kFloat));
  inv_freq = 1.0 / torch::pow(torch::tensor(config_.rope_theta, torch::TensorOptions().device(device_).dtype(torch::kFloat)),
                              inv_freq / static_cast<double>(head_dim_));

  auto pos_indices = torch::arange(0, config_.max_position_embeddings,
      torch::TensorOptions().device(device_).dtype(torch::kFloat));
  auto freqs = pos_indices.unsqueeze(1) * inv_freq.unsqueeze(0);
  auto emb = torch::cat({freqs, freqs}, -1);
  rope_cos_ = emb.cos().to(dtype_);
  rope_sin_ = emb.sin().to(dtype_);

  // Choose loader based on NF4 mode
  auto load_proj = [&](const std::string& prefix) -> LinearWeight {
    if (config_.use_nf4) {
      return load_nf4_linear(state_dict, prefix, device_, dtype_, config_.use_bias);
    } else {
      return load_linear(state_dict, prefix, device_, dtype_, config_.use_bias);
    }
  };

  layers_.reserve(config_.num_hidden_layers);
  for (int64_t i = 0; i < config_.num_hidden_layers; ++i) {
    QwenLayerWeights layer;
    std::string prefix = "layers." + std::to_string(i);

    layer.q_proj = load_proj(prefix + ".self_attn.q_proj");
    layer.k_proj = load_proj(prefix + ".self_attn.k_proj");
    layer.v_proj = load_proj(prefix + ".self_attn.v_proj");
    layer.o_proj = load_proj(prefix + ".self_attn.o_proj");

    layer.gate_proj = load_proj(prefix + ".mlp.gate_proj");
    layer.up_proj = load_proj(prefix + ".mlp.up_proj");
    layer.down_proj = load_proj(prefix + ".mlp.down_proj");

    // Build fused QKV and gate_up for fp16/bf16 (skip for NF4)
    if (!config_.use_nf4) {
      LinearWeight qkv;
      qkv.weight = torch::cat({layer.q_proj.weight,
                                layer.k_proj.weight,
                                layer.v_proj.weight}, 0);
      if (layer.q_proj.bias.has_value()) {
        qkv.bias = torch::cat({*layer.q_proj.bias,
                                *layer.k_proj.bias,
                                *layer.v_proj.bias}, 0);
      }
      layer.qkv_proj = std::move(qkv);

      LinearWeight gu;
      gu.weight = torch::cat({layer.gate_proj.weight,
                               layer.up_proj.weight}, 0);
      if (layer.gate_proj.bias.has_value()) {
        gu.bias = torch::cat({*layer.gate_proj.bias,
                               *layer.up_proj.bias}, 0);
      }
      layer.gate_up_proj = std::move(gu);

      // Free the now-redundant separate weights to reclaim GPU memory.
      layer.q_proj = {};
      layer.k_proj = {};
      layer.v_proj = {};
      layer.gate_proj = {};
      layer.up_proj = {};
    }

    layer.input_norm = get_tensor(state_dict, prefix + ".input_layernorm.weight", device_, dtype_);
    layer.post_norm = get_tensor(state_dict, prefix + ".post_attention_layernorm.weight", device_, dtype_);

    if (config_.use_qk_norm) {
      layer.q_norm = get_tensor(state_dict, prefix + ".self_attn.q_norm.weight", device_, dtype_);
      layer.k_norm = get_tensor(state_dict, prefix + ".self_attn.k_norm.weight", device_, dtype_);
    }

    layers_.push_back(std::move(layer));
  }
}

// ---------------------------------------------------------------------------
// Batch prefill: all sequences processed in a single packed forward pass
// ---------------------------------------------------------------------------
torch::Tensor QwenModel::forward_prefill(
    const std::vector<std::vector<int64_t>>& input_ids_list,
    const std::vector<int64_t>& seq_ids,
    PagedKVCache& cache,
    const std::vector<int64_t>& start_positions) {
  int64_t nseq = static_cast<int64_t>(input_ids_list.size());
  if (nseq != static_cast<int64_t>(seq_ids.size())) {
    throw std::runtime_error("input_ids_list and seq_ids size mismatch");
  }

  // Resolve per-sequence start positions (default to 0 for backward compat).
  std::vector<int64_t> start_pos(nseq, 0);
  if (!start_positions.empty()) {
    if (static_cast<int64_t>(start_positions.size()) != nseq) {
      throw std::runtime_error("start_positions size mismatch");
    }
    start_pos = start_positions;
  }

  // Calculate sequence lengths and total tokens for packing. Also find max seqlen for attention.
  std::vector<int64_t> seq_lens(nseq);
  int64_t total_tokens = 0;
  int64_t max_seqlen = 0;
  for (int64_t i = 0; i < nseq; ++i) {
    seq_lens[i] = static_cast<int64_t>(input_ids_list[i].size());
    total_tokens += seq_lens[i];
    max_seqlen = std::max(max_seqlen, seq_lens[i]);
  }

  std::vector<int64_t> all_ids;
  all_ids.reserve(total_tokens);
  for (const auto& ids : input_ids_list) {
    all_ids.insert(all_ids.end(), ids.begin(), ids.end());
  }
  auto flat_ids = torch::tensor(all_ids,
      torch::TensorOptions().device(device_).dtype(torch::kLong));

  auto x = torch::index_select(tok_embeddings_, 0, flat_ids);

  // Positions: start from start_pos[i] for each sequence.
  std::vector<int64_t> pos_data;
  pos_data.reserve(total_tokens);
  for (int64_t i = 0; i < nseq; ++i) {
    for (int64_t p = 0; p < seq_lens[i]; ++p) {
      pos_data.push_back(start_pos[i] + p);
    }
  }
  auto positions = torch::tensor(pos_data,
      torch::TensorOptions().device(device_).dtype(torch::kLong));

  // cu_seqlens for the Q side (new tokens only).
  std::vector<int32_t> cu_vals(nseq + 1);
  cu_vals[0] = 0;
  for (int64_t i = 0; i < nseq; ++i) {
    cu_vals[i + 1] = cu_vals[i] + static_cast<int32_t>(seq_lens[i]);
  }
  auto cu_seqlens = torch::tensor(cu_vals,
      torch::TensorOptions().dtype(torch::kInt32).device(device_));

  // Detect if any sequence has a cached prefix (start_pos > 0).
  bool has_prefix_cache = false;
  for (int64_t i = 0; !has_prefix_cache && i < nseq; ++i) {
    if (start_pos[i] > 0) has_prefix_cache = true; 
  }

  // If using paged prefill, plan once before the layer loop.
  // We need block_tables and context_lens from the cache.
  torch::Tensor block_tables_t, context_lens_t;
  if (has_prefix_cache) {
    block_tables_t = cache.block_tables_cpu_tensor(seq_ids);
    // context_lens = start_pos[i] + seq_lens[i] for each sequence
    // (total KV length: cached prefix + new tokens to be appended).
    // But at this point, only start_pos[i] tokens are in the cache.
    // After appending new K/V per layer, the total will be start_pos[i] + seq_lens[i].
    // We pass the final total KV length for attention.
    std::vector<int32_t> ctx_lens(nseq);
    for (int64_t i = 0; i < nseq; ++i) {
      ctx_lens[i] = static_cast<int32_t>(start_pos[i] + seq_lens[i]);
    }
    context_lens_t = torch::tensor(ctx_lens,
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  }

  int64_t q_dim = config_.num_attention_heads * head_dim_;
  int64_t kv_dim = config_.num_key_value_heads * head_dim_;
  bool use_fused = !config_.use_nf4;

  for (int64_t li = 0; li < config_.num_hidden_layers; ++li) {
    auto& layer = layers_[li];

    auto x_norm = cuda::rms_norm(x, layer.input_norm, config_.rms_norm_eps);

    torch::Tensor q, k, v;
    if (use_fused) {
      auto qkv = linear(x_norm, layer.qkv_proj);
      auto splits = qkv.split_with_sizes({q_dim, kv_dim, kv_dim}, -1);
      q = splits[0].view({total_tokens, config_.num_attention_heads, head_dim_});
      k = splits[1].view({total_tokens, config_.num_key_value_heads, head_dim_});
      v = splits[2].view({total_tokens, config_.num_key_value_heads, head_dim_});
    } else {
      q = linear(x_norm, layer.q_proj);
      k = linear(x_norm, layer.k_proj);
      v = linear(x_norm, layer.v_proj);
      q = q.view({total_tokens, config_.num_attention_heads, head_dim_});
      k = k.view({total_tokens, config_.num_key_value_heads, head_dim_});
      v = v.view({total_tokens, config_.num_key_value_heads, head_dim_});
    }

    if (config_.use_qk_norm) {
      auto q_flat = q.reshape({total_tokens * config_.num_attention_heads, head_dim_});
      q_flat = cuda::rms_norm(q_flat, layer.q_norm, config_.rms_norm_eps);
      q = q_flat.view({total_tokens, config_.num_attention_heads, head_dim_});
      auto k_flat = k.reshape({total_tokens * config_.num_key_value_heads, head_dim_});
      k_flat = cuda::rms_norm(k_flat, layer.k_norm, config_.rms_norm_eps);
      k = k_flat.view({total_tokens, config_.num_key_value_heads, head_dim_});
    }

    cuda::apply_rope(q, k, positions, rope_cos_, rope_sin_);

    // Append new K/V to cache (skip-write for shared blocks is handled inside).
    int64_t offset = 0;
    for (int64_t i = 0; i < nseq; ++i) {
      auto k_seq = k.narrow(0, offset, seq_lens[i]);
      auto v_seq = v.narrow(0, offset, seq_lens[i]);
      cache.append(seq_ids[i], li, k_seq, v_seq, start_pos[i]);
      offset += seq_lens[i];
    }

    torch::Tensor context;
    if (has_prefix_cache) {
      // Paged prefill: new tokens attend to full KV (cached + new) via page table.
      if (li == 0) {
        // Plan once, reuse across all layers.
        // Re-fetch block_tables after first layer's append to include newly
        // allocated blocks for the new tokens.
        block_tables_t = cache.block_tables_cpu_tensor(seq_ids);
        cuda::flash_attn_prefill_paged_plan(
            q, cache.k_pool(li), cu_seqlens,
            block_tables_t, context_lens_t,
            0.0f, cache.block_size(), /*causal=*/true);
      }
      context = cuda::flash_attn_prefill_paged_run(
          q.contiguous(), cache.k_pool(li), cache.v_pool(li));
    } else {
      // Standard path: self-attention on new tokens only.
      context = cuda::flash_attn_varlen(
          q.contiguous(), k.contiguous(), v.contiguous(),
          cu_seqlens, cu_seqlens,
          max_seqlen, max_seqlen, 0.0f, true);
    }

    context = context.reshape({total_tokens, config_.num_attention_heads * head_dim_});
    auto attn_out = linear(context, layer.o_proj);
    x = x + attn_out;

    auto ffn_norm = cuda::rms_norm(x, layer.post_norm, config_.rms_norm_eps);
    torch::Tensor gate_up;
    if (use_fused) {
      gate_up = linear(ffn_norm, layer.gate_up_proj);
    } else {
      auto gate = linear(ffn_norm, layer.gate_proj);
      auto up = linear(ffn_norm, layer.up_proj);
      gate_up = torch::cat({gate, up}, -1);
    }
    auto ff = cuda::silu_mul(gate_up);
    auto mlp_out = linear(ff, layer.down_proj);
    x = x + mlp_out;
  }

  auto x_norm_final = cuda::rms_norm(x, norm_weight_, config_.rms_norm_eps);

  std::vector<int64_t> last_indices;
  last_indices.reserve(nseq);
  int64_t offset = 0;
  for (int64_t i = 0; i < nseq; ++i) {
    last_indices.push_back(offset + seq_lens[i] - 1);
    offset += seq_lens[i];
  }
  auto indices = torch::tensor(last_indices,
      torch::TensorOptions().device(device_).dtype(torch::kLong));
  auto last_tokens = x_norm_final.index_select(0, indices);
  auto logits = torch::matmul(last_tokens, lm_head_weight_.t());
  return logits;
}

torch::Tensor QwenModel::forward_decode(const torch::Tensor& input_ids,
                                        const torch::Tensor& positions,
                                        const std::vector<int64_t>& seq_ids,
                                        PagedKVCache& cache,
                                        const torch::Tensor& positions_cpu) {
  auto input = input_ids.to(device_).to(torch::kLong);
  auto pos = positions.is_cuda() ? positions.to(torch::kLong)
                                 : positions.to(device_).to(torch::kLong);

  int64_t bsz = input.size(0);
  int64_t seq_len = input.size(1);
  if (seq_len != 1) {
    throw std::runtime_error("forward_decode expects seq_len == 1");
  }
  if (static_cast<int64_t>(seq_ids.size()) != bsz) {
    throw std::runtime_error("seq_ids size mismatch in forward_decode");
  }

  auto flat = input.reshape({-1});
  auto emb = torch::index_select(tok_embeddings_, 0, flat);
  auto x = emb.view({bsz, seq_len, config_.hidden_size});

  int64_t q_dim = config_.num_attention_heads * head_dim_;
  int64_t kv_dim = config_.num_key_value_heads * head_dim_;
  bool use_fused = !config_.use_nf4;

  torch::Tensor block_tables, context_lens;

  // Use caller-provided CPU positions to avoid GPU->CPU sync.
  auto start_pos_cpu = positions_cpu.defined()
      ? positions_cpu.to(torch::kLong).contiguous()
      : pos.squeeze(1).to(torch::kCPU);

  // Pre-compute slot_mapping once: allocates blocks, updates cur_len, builds
  // CUDA int32 [B] mapping. Reused across all 28 layers (same slots).
  auto slot_mapping = cache.prepare_decode_slots(seq_ids, start_pos_cpu);

  for (int64_t i = 0; i < config_.num_hidden_layers; ++i) {
    auto& layer = layers_[i];

    auto x_norm = cuda::rms_norm(x, layer.input_norm, config_.rms_norm_eps);

    torch::Tensor q, k, v;
    if (use_fused) {
      auto qkv = linear(x_norm, layer.qkv_proj);
      auto splits = qkv.split_with_sizes({q_dim, kv_dim, kv_dim}, -1);
      q = splits[0].view({bsz, seq_len, config_.num_attention_heads, head_dim_});
      k = splits[1].view({bsz, seq_len, config_.num_key_value_heads, head_dim_});
      v = splits[2].view({bsz, seq_len, config_.num_key_value_heads, head_dim_});
    } else {
      q = linear(x_norm, layer.q_proj);
      k = linear(x_norm, layer.k_proj);
      v = linear(x_norm, layer.v_proj);
      q = q.view({bsz, seq_len, config_.num_attention_heads, head_dim_});
      k = k.view({bsz, seq_len, config_.num_key_value_heads, head_dim_});
      v = v.view({bsz, seq_len, config_.num_key_value_heads, head_dim_});
    }

    if (config_.use_qk_norm) {
      auto q_flat = q.reshape({bsz * seq_len * config_.num_attention_heads, head_dim_});
      q_flat = cuda::rms_norm(q_flat, layer.q_norm, config_.rms_norm_eps);
      q = q_flat.view({bsz, seq_len, config_.num_attention_heads, head_dim_});
      auto k_flat = k.reshape({bsz * seq_len * config_.num_key_value_heads, head_dim_});
      k_flat = cuda::rms_norm(k_flat, layer.k_norm, config_.rms_norm_eps);
      k = k_flat.view({bsz, seq_len, config_.num_key_value_heads, head_dim_});
    }

    cuda::apply_rope(q, k, pos, rope_cos_, rope_sin_);

    cache.append_batch(i, k, v, slot_mapping);

    if (i == 0) {
      block_tables = cache.block_tables_cpu_tensor(seq_ids);
      context_lens = cache.context_lens_cpu_tensor(seq_ids);

      // Plan once: converts CPU metadata to FlashInfer decode state.
      cuda::flash_attn_decode_plan(
          q.contiguous(), cache.k_pool(0),
          block_tables, context_lens,
          0.0f, cache.block_size());
    }

    // Run only: no GPU->CPU sync, no plan() -- just the FlashInfer kernel.
    torch::Tensor context = cuda::flash_attn_decode_run(
        q.contiguous(),
        cache.k_pool(i),
        cache.v_pool(i));

    context = context.reshape({bsz, seq_len, config_.num_attention_heads * head_dim_});
    auto attn_out = linear(context, layer.o_proj);
    x = x + attn_out;

    auto ffn_norm = cuda::rms_norm(x, layer.post_norm, config_.rms_norm_eps);
    torch::Tensor gate_up;
    if (use_fused) {
      gate_up = linear(ffn_norm, layer.gate_up_proj);
    } else {
      auto gate = linear(ffn_norm, layer.gate_proj);
      auto up = linear(ffn_norm, layer.up_proj);
      gate_up = torch::cat({gate, up}, -1);
    }
    auto ff = cuda::silu_mul(gate_up);
    auto mlp_out = linear(ff, layer.down_proj);
    x = x + mlp_out;
  }

  auto x_norm = cuda::rms_norm(x, norm_weight_, config_.rms_norm_eps);
  auto last = x_norm.index({Slice(), 0, Slice()});
  auto logits = torch::matmul(last, lm_head_weight_.t());
  return logits;
}

}  // namespace tvllm
