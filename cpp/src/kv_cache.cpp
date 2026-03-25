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

void PagedKVCache::append(int64_t seq_id, int64_t layer,
                          const torch::Tensor& k, const torch::Tensor& v,
                          int64_t start_pos) {
  auto& seq = get_seq(seq_id);
  if (layer < 0 || layer >= config_.num_hidden_layers) {
    throw std::runtime_error("invalid layer index: " + std::to_string(layer));
  }

  // Normalise shape to [num_tokens, num_kv_heads, head_dim].
  auto k_in = k.dim() == 2 ? k.unsqueeze(0) : k;
  auto v_in = v.dim() == 2 ? v.unsqueeze(0) : v;
  if (k_in.dim() != 3 || v_in.dim() != 3) {
    throw std::runtime_error("append expects k/v shape [num_tokens, num_kv_heads, head_dim]");
  }
  if (!k_in.sizes().equals(v_in.sizes())) {
    throw std::runtime_error("k and v shape mismatch in append");
  }

  const int64_t num_tokens = k_in.size(0);
  if (start_pos < 0 || start_pos + num_tokens > config_.max_position_embeddings) {
    throw std::runtime_error("position overflow in append (start_pos=" +
        std::to_string(start_pos) + ", num_tokens=" + std::to_string(num_tokens) + ")");
  }

  auto& k_blk = block_manager_.k_blocks(layer);
  auto& v_blk = block_manager_.v_blocks(layer);

  // ---- Block-aligned batch copy ------------------------------------------
  // Group contiguous tokens that land in the same physical block and copy
  // them in one narrow().copy_() call.  This reduces the number of CUDA
  // kernel launches from 2*T to 2*ceil(T / block_size).
  int64_t idx = 0;
  while (idx < num_tokens) {
    const int64_t pos      = start_pos + idx;
    const int32_t block_id = allocate_blocks_up_to(seq_id, pos);
    const int64_t slot     = pos % block_size_;
    const int64_t chunk    = std::min(block_size_ - slot, num_tokens - idx);
    const int64_t logical_block = pos / block_size_;

    // Skip writing to blocks that are shared (refcount > 1) — KV data
    // is already there from a previous sequence.
    bool skip_write = block_manager_.prefix_caching_enabled() &&
                      block_manager_.block_meta(block_id).refcount > 1 &&
                      block_manager_.block_meta(block_id).is_full;

    if (!skip_write) {
      k_blk.select(0, block_id).narrow(0, slot, chunk)
           .copy_(k_in.narrow(0, idx, chunk));
      v_blk.select(0, block_id).narrow(0, slot, chunk)
           .copy_(v_in.narrow(0, idx, chunk));
    }

    // Register block hash when block becomes full.
    if (block_manager_.prefix_caching_enabled() &&
        slot + chunk == block_size_ &&
        logical_block < static_cast<int64_t>(seq.block_hashes.size()) &&
        seq.block_hashes[logical_block] != 0 &&
        !block_manager_.block_meta(block_id).is_full) {
      block_manager_.mark_block_full(block_id, seq.block_hashes[logical_block]);
    }

    idx += chunk;
  }

  seq.cur_len = std::max(seq.cur_len, start_pos + num_tokens);
}

torch::Tensor PagedKVCache::prepare_decode_slots(
    const std::vector<int64_t>& seq_ids,
    const torch::Tensor& start_positions_cpu) {
  const int64_t bsz = static_cast<int64_t>(seq_ids.size());
  auto start_acc = start_positions_cpu.accessor<int64_t, 2>();

  std::vector<int32_t> slot_mapping_vec;
  slot_mapping_vec.reserve(bsz);

  for (int64_t b = 0; b < bsz; ++b) {
    const int64_t seq_id = seq_ids[b];
    const int64_t pos = start_acc[b][0];
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
  cuda::scatter_kv_decode(k_blk, v_blk, k, v, slot_mapping);
}

// ===========================================================================
//  Metadata for kernel dispatch
// ===========================================================================

torch::Tensor PagedKVCache::block_tables_cpu_tensor(
    const std::vector<int64_t>& seq_ids) const {
  const int64_t bsz = static_cast<int64_t>(seq_ids.size());

  // Determine the widest block table in the batch.
  int64_t max_blocks = 0;
  for (auto sid : seq_ids) {
    const auto& seq = get_seq(sid);
    max_blocks = std::max(max_blocks,
                          static_cast<int64_t>(seq.block_table.size()));
  }

  // Build on CPU with accessor; decode planning consumes host metadata.
  auto out = torch::full({bsz, max_blocks}, /*fill=*/-1,
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  auto acc = out.accessor<int32_t, 2>();

  for (int64_t b = 0; b < bsz; ++b) {
    const auto& table = get_seq(seq_ids[b]).block_table;
    for (int64_t i = 0; i < static_cast<int64_t>(table.size()); ++i) {
      acc[b][i] = table[i];
    }
  }

  return out;
}

torch::Tensor PagedKVCache::context_lens_cpu_tensor(
    const std::vector<int64_t>& seq_ids) const {
  std::vector<int32_t> lens;
  lens.reserve(seq_ids.size());
  for (auto sid : seq_ids) {
    lens.push_back(static_cast<int32_t>(get_seq(sid).cur_len));
  }
  return torch::tensor(lens,
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
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
