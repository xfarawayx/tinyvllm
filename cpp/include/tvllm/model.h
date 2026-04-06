#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <vector>

#include "tvllm/config.h"

namespace tvllm {

class PagedKVCache;

struct NF4Weight {
  torch::Tensor packed_weights;  // uint8 [n_elements/2]
  torch::Tensor absmax_q;        // uint8 [num_blocks]
  torch::Tensor absmax2;         // fp16  [num_groups]
  torch::Tensor code2;           // fp16  [256]
  float offset;
  int blocksize;
  int s2_blocksize;
  int64_t num_rows;
  int64_t num_cols;
};

struct LinearWeight {
  torch::Tensor weight;                // fp16/bf16 [out, in] (used when not NF4)
  c10::optional<torch::Tensor> bias;
  c10::optional<NF4Weight> nf4;        // present when weight is NF4-quantized
};

struct QwenLayerWeights {
  // Separate projections (always populated; used for NF4 and qk_norm paths)
  LinearWeight q_proj;
  LinearWeight k_proj;
  LinearWeight v_proj;
  LinearWeight o_proj;
  LinearWeight gate_proj;
  LinearWeight up_proj;
  LinearWeight down_proj;

  // Fused projections for fp16/bf16 (empty for NF4). Populated at load time
  // by concatenating the separate weight matrices.
  LinearWeight qkv_proj;      // [q_dim+2*kv_dim, hidden] when fused
  LinearWeight gate_up_proj;  // [2*intermediate, hidden] when fused

  torch::Tensor input_norm;
  torch::Tensor post_norm;
  // Qwen3 per-head QK-norm (optional)
  torch::Tensor q_norm;
  torch::Tensor k_norm;
};

class QwenModel {
 public:
  QwenModel(const ModelConfig& config,
            const c10::impl::GenericDict& state_dict,
            torch::Device device,
            torch::Dtype dtype);

  /// Prefill and return logits.
  /// all_logits=false (default): last-token only [nseq, vocab].
  /// all_logits=true: all tokens [total_tokens, vocab].
  torch::Tensor forward_prefill(
      const std::vector<std::vector<int64_t>>& input_ids_list,
      const std::vector<int64_t>& seq_ids,
      PagedKVCache& cache,
      const std::vector<int64_t>& start_positions = {},
      bool all_logits = false);

  torch::Tensor forward_decode(const torch::Tensor& input_ids,
                               const torch::Tensor& positions,
                               const std::vector<int64_t>& seq_ids,
                               PagedKVCache& cache,
                               const std::vector<int64_t>& start_positions);

  const ModelConfig& config() const { return config_; }

 private:
  ModelConfig config_;
  torch::Device device_;
  torch::Dtype dtype_;
  int64_t head_dim_;

  torch::Tensor tok_embeddings_;
  torch::Tensor norm_weight_;
  torch::Tensor lm_head_weight_;
  torch::Tensor rope_cos_;
  torch::Tensor rope_sin_;

  std::vector<QwenLayerWeights> layers_;
};

}  // namespace tvllm
