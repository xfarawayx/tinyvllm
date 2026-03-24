#pragma once

// ---------------------------------------------------------------------------
// kv_cache.h  —  Paged KV cache for transformer inference
//
// Manages a pool of fixed-size KV blocks shared across sequences.
// Each sequence has its own block table that maps logical positions
// to physical block ids, enabling efficient memory sharing and
// dynamic sequence management (foundation for continuous batching).
//
// Supports optional prefix caching: full blocks whose token content
// matches a previous sequence are shared via reference counting.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "tvllm/block_manager.h"
#include "tvllm/config.h"

namespace tvllm {

class PagedKVCache {
 public:
  // ---- Construction & reset ------------------------------------------------

  PagedKVCache(const ModelConfig& config,
               torch::Device device,
               torch::Dtype dtype,
               int64_t block_size = 16,
               int64_t num_blocks = 0,
               bool prefix_caching = false);

  /// Clear all sequences and return every block to the free pool.
  void reset();

  // ---- Sequence lifecycle --------------------------------------------------
  //  Support dynamic add/remove for future continuous batching.

  /// Register a new sequence.  Returns the assigned seq_id.
  int64_t add_sequence();

  /// Register a new sequence with prefix caching.
  /// Attempts to match full blocks of token_ids against cached blocks.
  /// Returns (seq_id, num_matched_tokens).
  std::pair<int64_t, int64_t> add_sequence_with_prefix(
      const std::vector<int64_t>& token_ids);

  /// Remove a sequence and free all of its blocks.
  void remove_sequence(int64_t seq_id);

  /// Check whether a seq_id is currently active.
  bool has_sequence(int64_t seq_id) const;

  /// Number of currently active sequences.
  int64_t num_sequences() const { return num_active_; }

  // ---- KV write ------------------------------------------------------------

  /// Append K/V for a single sequence at one layer.
  ///   k, v : [num_tokens, num_kv_heads, head_dim]
  /// Copies are batched per-block to minimize kernel launches.
  void append(int64_t seq_id, int64_t layer,
              const torch::Tensor& k, const torch::Tensor& v,
              int64_t start_pos);

  /// Decode batch variant backed by the fused scatter_kv_decode CUDA kernel.
  ///   k, v : [batch_size, num_tokens, num_kv_heads, head_dim]
  ///   start_positions : [batch_size]
  void append_batch(int64_t layer,
                    const std::vector<int64_t>& seq_ids,
                    const torch::Tensor& k, const torch::Tensor& v,
                    const torch::Tensor& start_positions);

  // ---- Metadata for kernel dispatch ----------------------------------------

  /// Build CPU int32 block_tables metadata, shape [len(seq_ids), max_num_blocks].
  /// Unused slots are filled with -1.
  torch::Tensor block_tables_cpu_tensor(const std::vector<int64_t>& seq_ids) const;

  /// Build CPU int32 context_lens metadata, shape [len(seq_ids)].
  torch::Tensor context_lens_cpu_tensor(const std::vector<int64_t>& seq_ids) const;

  // ---- Per-sequence query --------------------------------------------------

  int64_t seq_len(int64_t seq_id) const;

  // ---- Block pool access (for attention kernels) ---------------------------
  //  Shape: [num_blocks, block_size, num_kv_heads, head_dim]

  const torch::Tensor& k_pool(int64_t layer) const;
  const torch::Tensor& v_pool(int64_t layer) const;

  // ---- Properties ----------------------------------------------------------

  int64_t block_size()  const { return block_size_; }
  int64_t max_seq_len() const { return config_.max_position_embeddings; }
  int64_t num_layers()  const { return config_.num_hidden_layers; }
  torch::Device device() const { return device_; }
  int64_t num_blocks()  const { return block_manager_.num_blocks(); }
  int64_t free_block_count() const { return block_manager_.free_block_count(); }
  bool prefix_caching_enabled() const { return block_manager_.prefix_caching_enabled(); }

 private:
  struct SequenceState {
    bool active = false;
    int64_t cur_len = 0;
    std::vector<int32_t> block_table;
    std::vector<uint64_t> block_hashes;  // parallel to block_table, for prefix caching
  };

  /// Allocate physical blocks so that `pos` is covered.
  int32_t allocate_blocks_up_to(int64_t seq_id, int64_t pos);

  /// Validate and return a sequence (throws on invalid / inactive id).
  const SequenceState& get_seq(int64_t seq_id) const;
  SequenceState& get_seq(int64_t seq_id);

  ModelConfig config_;
  torch::Device device_;
  int64_t head_dim_;
  int64_t block_size_;
  int64_t num_active_ = 0;
  BlockManager block_manager_;
  std::vector<SequenceState> sequences_;
  std::vector<int64_t> free_seq_ids_;   // recycled sequence slot ids
};

}  // namespace tvllm
