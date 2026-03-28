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

  /// Pre-compute slot_mapping for a prefill batch (multiple tokens per sequence).
  /// Returns a CUDA int32 tensor [total_tokens] of flat slot indices.
  /// Slots for prefix-cached blocks are set to -1 (kernel will skip them).
  /// Also allocates needed blocks, registers block hashes, and updates cur_len.
  torch::Tensor prepare_prefill_slots(
      const std::vector<int64_t>& seq_ids,
      const std::vector<int64_t>& start_positions,
      const std::vector<int64_t>& seq_lens);

  /// Pre-compute slot_mapping for a decode batch (one token per sequence).
  /// Returns a CUDA int32 tensor [batch_size] of flat slot indices.
  /// Also allocates any needed blocks and updates cur_len.
  torch::Tensor prepare_decode_slots(const std::vector<int64_t>& seq_ids,
                                     const std::vector<int64_t>& start_positions);

  /// Scatter K/V into paged cache using a pre-computed slot_mapping.
  /// Accepts any k/v rank whose last two dims are [num_kv_heads, head_dim]
  /// (e.g. [N, Hkv, D] for prefill, [B, 1, Hkv, D] for decode).
  /// slot_mapping entries of -1 are skipped (prefix-cached blocks).
  void append_batch(int64_t layer,
                    const torch::Tensor& k, const torch::Tensor& v,
                    const torch::Tensor& slot_mapping);

  // ---- Metadata for kernel dispatch ----------------------------------------

  /// Return per-sequence block tables (physical block ids).
  std::vector<std::vector<int32_t>> block_tables(
      const std::vector<int64_t>& seq_ids) const;

  /// Return per-sequence context lengths (number of tokens stored).
  std::vector<int32_t> context_lens(
      const std::vector<int64_t>& seq_ids) const;

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
