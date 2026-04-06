#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tvllm/config.h"
#include "tvllm/kv_cache.h"
#include "tvllm/model.h"
#include "tvllm/sampler.h"

namespace tvllm {

class Engine {
 public:
  explicit Engine(const std::string& model_path);
  Engine(const std::string& model_dir, const c10::impl::GenericDict& state_dict);

  /// Continuous-batching generate: processes requests with at most
  /// max_batch_size concurrent sequences.  When a sequence finishes,
  /// its slot is immediately reused for a waiting request.
  ///
  /// @param sample_params  Per-request sampling parameters.
  /// @param max_batch_size 0 means no limit (all requests at once).
  std::vector<std::vector<int64_t>> generate(
      const std::vector<std::vector<int64_t>>& batch_input_ids,
      const std::vector<SampleParams>& sample_params,
      int64_t max_batch_size);

  /// Run prefill and return logits for all tokens [total_tokens, vocab].
  torch::Tensor forward_logits(
      const std::vector<std::vector<int64_t>>& batch_input_ids);

  void reset();

  int64_t eos_token_id() const { return config_.eos_token_id; }
  int64_t max_position_embeddings() const { return config_.max_position_embeddings; }

 private:
  ModelConfig config_;
  torch::Device device_;
  torch::Dtype dtype_;
  std::unique_ptr<QwenModel> model_;

  // Memory profiling results (computed once at construction).
  int64_t block_size_ = 16;
  int64_t max_num_batched_tokens_ = 8192;
  int64_t num_kv_blocks_ = 1;
  bool prefix_caching_ = false;

  // Persistent KV cache — survives across generate() calls so that
  // prefix caching can reuse KV blocks from previous conversations.
  std::unique_ptr<PagedKVCache> cache_;

  /// Run a warmup forward pass and profile peak activation memory,
  /// then compute the number of KV-cache blocks from remaining GPU memory.
  void profile_memory();
};

}  // namespace tvllm
