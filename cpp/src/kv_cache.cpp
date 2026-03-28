#include "tvllm/kv_cache.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "tvllm/fused_kernels.h"

namespace tvllm {

// ===========================================================================
//  Construction & reset
// ===========================================================================

PagedKVCache::PagedKVCache(const ModelConfig& config,
                           torch::Device device,
                           torch::Dtype dtype,
                           int64_t block_size,
                           int64_t num_blocks,
                           bool prefix_caching)
    : config_(config),
      device_(device),
      head_dim_(config.head_dim > 0 ? config.head_dim : config.hidden_size / config.num_attention_heads),
      block_size_(block_size),
      block_manager_(config.num_hidden_layers,
                     num_blocks > 0
                         ? num_blocks
                         : (config.max_position_embeddings + block_size - 1) / block_size,
                     block_size,
                     config.num_key_value_heads,
                     head_dim_,
                     device,
                     dtype,
                     prefix_caching) {
  if (block_size_ <= 0) {
    throw std::runtime_error("block_size must be > 0");
  }
  if (!device_.is_cuda()) {
    throw std::runtime_error("PagedKVCache requires a CUDA device");
  }
}

void PagedKVCache::reset() {
  block_manager_.reset();
  sequences_.clear();
  free_seq_ids_.clear();
  num_active_ = 0;
}

// ===========================================================================
//  Sequence lifecycle
// ===========================================================================

int64_t PagedKVCache::add_sequence() {
  int64_t seq_id;
  if (!free_seq_ids_.empty()) {
    // Re-use a previously freed slot.
    seq_id = free_seq_ids_.back();
    free_seq_ids_.pop_back();
    sequences_[seq_id] = SequenceState{/*active=*/true, /*cur_len=*/0, {}};
  } else {
    seq_id = static_cast<int64_t>(sequences_.size());
    sequences_.push_back(SequenceState{/*active=*/true, /*cur_len=*/0, {}});
  }
  ++num_active_;
  return seq_id;
}

std::pair<int64_t, int64_t> PagedKVCache::add_sequence_with_prefix(
    const std::vector<int64_t>& token_ids) {
  // Allocate sequence slot (same logic as add_sequence).
  int64_t seq_id = add_sequence();
  auto& seq = get_seq(seq_id);

  if (!block_manager_.prefix_caching_enabled()) {
    return {seq_id, 0};
  }

  const int64_t num_tokens = static_cast<int64_t>(token_ids.size());
  const int64_t num_full_blocks = num_tokens / block_size_;

  // Compute chained hashes and attempt to match cached blocks.
  uint64_t prefix_hash = 0;
  int64_t matched_blocks = 0;

  for (int64_t b = 0; b < num_full_blocks; ++b) {
    uint64_t block_hash = hash_block_tokens(
        prefix_hash, token_ids.data() + b * block_size_, block_size_);

    int32_t cached_id = block_manager_.lookup_cached(block_hash);
    if (cached_id >= 0) {
      // Cache hit: reuse this block.
      seq.block_table.push_back(cached_id);
      seq.block_hashes.push_back(block_hash);
      matched_blocks++;
      prefix_hash = block_hash;
    } else {
      // Cache miss: stop matching. Allocate fresh block for this one.
      // Pre-compute hashes for all remaining blocks so we can register
      // them after the KV data is written.
      int32_t new_id = block_manager_.alloc_block();
      seq.block_table.push_back(new_id);
      seq.block_hashes.push_back(block_hash);
      prefix_hash = block_hash;

      // Allocate fresh blocks for remaining full blocks.
      for (int64_t b2 = b + 1; b2 < num_full_blocks; ++b2) {
        block_hash = hash_block_tokens(
            prefix_hash, token_ids.data() + b2 * block_size_, block_size_);
        new_id = block_manager_.alloc_block();
        seq.block_table.push_back(new_id);
        seq.block_hashes.push_back(block_hash);
        prefix_hash = block_hash;
      }
      break;
    }
  }

  int64_t matched_tokens = matched_blocks * block_size_;
  seq.cur_len = matched_tokens;  // cached tokens are already in KV
  return {seq_id, matched_tokens};
}

void PagedKVCache::remove_sequence(int64_t seq_id) {
  auto& seq = get_seq(seq_id);
  // Return every physical block to the free pool.
  for (int32_t block_id : seq.block_table) {
    block_manager_.free_block(block_id);
  }
  seq = SequenceState{};            // marks active = false
  free_seq_ids_.push_back(seq_id);
  --num_active_;
}

bool PagedKVCache::has_sequence(int64_t seq_id) const {
  if (seq_id < 0 || seq_id >= static_cast<int64_t>(sequences_.size())) {
    return false;
  }
  return sequences_[seq_id].active;
}

// ===========================================================================
//  Internal helpers
// ===========================================================================

const PagedKVCache::SequenceState& PagedKVCache::get_seq(int64_t seq_id) const {
  if (seq_id < 0 || seq_id >= static_cast<int64_t>(sequences_.size())) {
    throw std::runtime_error("invalid seq_id: " + std::to_string(seq_id));
  }
  if (!sequences_[seq_id].active) {
    throw std::runtime_error("seq_id " + std::to_string(seq_id) + " is not active");
  }
  return sequences_[seq_id];
}

PagedKVCache::SequenceState& PagedKVCache::get_seq(int64_t seq_id) {
  return const_cast<SequenceState&>(
      static_cast<const PagedKVCache&>(*this).get_seq(seq_id));
}

int32_t PagedKVCache::allocate_blocks_up_to(int64_t seq_id, int64_t pos) {
  auto& seq = get_seq(seq_id);
  int64_t logical_block = pos / block_size_;
  while (logical_block >= static_cast<int64_t>(seq.block_table.size())) {
    seq.block_table.push_back(block_manager_.alloc_block());
  }
  return seq.block_table[logical_block];
}

// ===========================================================================
//  KV write
// ===========================================================================

torch::Tensor PagedKVCache::prepare_prefill_slots(
    const std::vector<int64_t>& seq_ids,
    const std::vector<int64_t>& start_positions,
    const std::vector<int64_t>& seq_lens) {
  const int64_t nseq = static_cast<int64_t>(seq_ids.size());
  const bool prefix_caching = block_manager_.prefix_caching_enabled();

  int64_t total_tokens = 0;
  for (auto len : seq_lens) total_tokens += len;

  std::vector<int32_t> slot_vec;
  slot_vec.reserve(total_tokens);

  for (int64_t i = 0; i < nseq; ++i) {
    auto& seq = get_seq(seq_ids[i]);
    const int64_t start_pos = start_positions[i];
    const int64_t num_tokens = seq_lens[i];

    if (num_tokens <= 0) continue;

    // Pre-allocate all blocks this sequence needs in one go.
    allocate_blocks_up_to(seq_ids[i], start_pos + num_tokens - 1);

    for (int64_t j = 0; j < num_tokens; ++j) {
      const int64_t pos = start_pos + j;
      const int64_t logical_block = pos / block_size_;
      const int64_t slot_in_block = pos % block_size_;
      const int32_t block_id = seq.block_table[logical_block];

      // Skip writing to shared blocks (KV data already present).
      if (prefix_caching) {
        const auto& meta = block_manager_.block_meta(block_id);
        if (meta.refcount > 1 && meta.is_full) {
          slot_vec.push_back(-1);
          continue;
        }
      }

      slot_vec.push_back(
          static_cast<int32_t>(block_id * block_size_ + slot_in_block));

      // Register block hash when this token completes a block.
      if (prefix_caching &&
          slot_in_block == block_size_ - 1 &&
          logical_block < static_cast<int64_t>(seq.block_hashes.size()) &&
          seq.block_hashes[logical_block] != 0 &&
          !block_manager_.block_meta(block_id).is_full) {
        block_manager_.mark_block_full(block_id, seq.block_hashes[logical_block]);
      }
    }

    seq.cur_len = std::max(seq.cur_len, start_pos + num_tokens);
  }

  return torch::tensor(
      slot_vec,
      torch::TensorOptions().dtype(torch::kInt32).device(device_));
}

torch::Tensor PagedKVCache::prepare_decode_slots(
    const std::vector<int64_t>& seq_ids,
    const std::vector<int64_t>& start_positions) {
  const int64_t bsz = static_cast<int64_t>(seq_ids.size());

  std::vector<int32_t> slot_mapping_vec;
  slot_mapping_vec.reserve(bsz);

  for (int64_t b = 0; b < bsz; ++b) {
    const int64_t seq_id = seq_ids[b];
    const int64_t pos = start_positions[b];
    if (pos < 0 || pos + 1 > config_.max_position_embeddings) {
      throw std::runtime_error("position overflow in prepare_decode_slots (start_pos=" +
          std::to_string(pos) + ")");
    }

    auto& seq = get_seq(seq_id);
    const int32_t block_id = allocate_blocks_up_to(seq_id, pos);
    const int64_t slot = pos % block_size_;
    slot_mapping_vec.push_back(
        static_cast<int32_t>(block_id * block_size_ + slot));
    seq.cur_len = std::max(seq.cur_len, pos + 1);
  }

  return torch::tensor(
      slot_mapping_vec,
      torch::TensorOptions().dtype(torch::kInt32).device(device_));
}

void PagedKVCache::append_batch(int64_t layer,
                                const torch::Tensor& k,
                                const torch::Tensor& v,
                                const torch::Tensor& slot_mapping) {
  if (layer < 0 || layer >= config_.num_hidden_layers) {
    throw std::runtime_error("invalid layer index: " + std::to_string(layer));
  }
  auto& k_blk = block_manager_.k_blocks(layer);
  auto& v_blk = block_manager_.v_blocks(layer);
  cuda::scatter_kv(k_blk, v_blk, k, v, slot_mapping);
}

// ===========================================================================
//  Metadata for kernel dispatch
// ===========================================================================

std::vector<std::vector<int32_t>> PagedKVCache::block_tables(
    const std::vector<int64_t>& seq_ids) const {
  std::vector<std::vector<int32_t>> out;
  out.reserve(seq_ids.size());
  for (auto sid : seq_ids) {
    out.push_back(get_seq(sid).block_table);
  }
  return out;
}

std::vector<int32_t> PagedKVCache::context_lens(
    const std::vector<int64_t>& seq_ids) const {
  std::vector<int32_t> out;
  out.reserve(seq_ids.size());
  for (auto sid : seq_ids) {
    out.push_back(static_cast<int32_t>(get_seq(sid).cur_len));
  }
  return out;
}

// ===========================================================================
//  Per-sequence query & pool access
// ===========================================================================

int64_t PagedKVCache::seq_len(int64_t seq_id) const {
  return get_seq(seq_id).cur_len;
}

const torch::Tensor& PagedKVCache::k_pool(int64_t layer) const {
  return block_manager_.k_blocks(layer);
}

const torch::Tensor& PagedKVCache::v_pool(int64_t layer) const {
  return block_manager_.v_blocks(layer);
}

}  // namespace tvllm
