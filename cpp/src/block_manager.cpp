#include "tvllm/block_manager.h"

#include <stdexcept>

namespace tvllm {

BlockManager::BlockManager(int64_t num_layers,
                           int64_t num_blocks,
                           int64_t block_size,
                           int64_t num_key_value_heads,
                           int64_t head_dim,
                           torch::Device device,
                           torch::Dtype dtype,
                           bool prefix_caching)
    : num_layers_(num_layers),
      num_blocks_(num_blocks),
      block_size_(block_size),
      prefix_caching_enabled_(prefix_caching) {
  if (num_layers_ <= 0 || num_blocks_ <= 0 || block_size_ <= 0) {
    throw std::runtime_error("invalid BlockManager shape");
  }
  if (!device.is_cuda()) {
    throw std::runtime_error("BlockManager requires CUDA tensors");
  }

  auto options = torch::TensorOptions().device(device).dtype(dtype);

  k_blocks_.reserve(num_layers_);
  v_blocks_.reserve(num_layers_);
  for (int64_t layer = 0; layer < num_layers_; ++layer) {
    k_blocks_.push_back(torch::zeros({num_blocks_, block_size_, num_key_value_heads, head_dim}, options));
    v_blocks_.push_back(torch::zeros({num_blocks_, block_size_, num_key_value_heads, head_dim}, options));
  }

  free_list_.reserve(num_blocks_);
  for (int32_t block_id = static_cast<int32_t>(num_blocks_ - 1); block_id >= 0; --block_id) {
    free_list_.push_back(block_id);
  }

  if (prefix_caching_enabled_) {
    block_meta_.resize(num_blocks_);
  }
}

void BlockManager::reset() {
  free_list_.clear();
  for (int32_t block_id = static_cast<int32_t>(num_blocks_ - 1); block_id >= 0; --block_id) {
    free_list_.push_back(block_id);
  }

  if (prefix_caching_enabled_) {
    block_meta_.assign(num_blocks_, BlockMeta{});
    hash_to_block_.clear();
    evictable_list_.clear();
    evictable_index_.clear();
  }
}

int32_t BlockManager::alloc_block() {
  if (!free_list_.empty()) {
    int32_t block_id = free_list_.back();
    free_list_.pop_back();
    if (prefix_caching_enabled_) {
      auto& meta = block_meta_[block_id];
      meta = BlockMeta{};
      meta.refcount = 1;
    }
    return block_id;
  }

  // If prefix caching is on, try to evict the LRU cached block.
  if (prefix_caching_enabled_ && !evictable_list_.empty()) {
    int32_t block_id = evictable_list_.front();
    evictable_remove(block_id);

    // Remove from hash table.
    auto& meta = block_meta_[block_id];
    if (meta.content_hash != 0) {
      hash_to_block_.erase(meta.content_hash);
    }
    meta = BlockMeta{};
    meta.refcount = 1;
    return block_id;
  }

  throw std::runtime_error("BlockManager out of free blocks");
}

void BlockManager::free_block(int32_t block_id) {
  if (block_id < 0 || block_id >= num_blocks_) {
    throw std::runtime_error("invalid block id in BlockManager::free_block");
  }
  if (prefix_caching_enabled_) {
    unref_block(block_id);
  } else {
    free_list_.push_back(block_id);
  }
}

int32_t BlockManager::lookup_cached(uint64_t content_hash) {
  if (!prefix_caching_enabled_ || content_hash == 0) {
    return -1;
  }
  auto it = hash_to_block_.find(content_hash);
  if (it == hash_to_block_.end()) {
    return -1;
  }
  int32_t block_id = it->second;
  auto& meta = block_meta_[block_id];

  // If refcount was 0, it's in the evictable list — remove it.
  if (meta.refcount == 0) {
    evictable_remove(block_id);
  }
  meta.refcount++;
  return block_id;
}

void BlockManager::ref_block(int32_t block_id) {
  if (!prefix_caching_enabled_) return;
  auto& meta = block_meta_[block_id];
  if (meta.refcount == 0) {
    evictable_remove(block_id);
  }
  meta.refcount++;
}

void BlockManager::unref_block(int32_t block_id) {
  if (!prefix_caching_enabled_) {
    free_list_.push_back(block_id);
    return;
  }
  auto& meta = block_meta_[block_id];
  if (meta.refcount <= 0) {
    throw std::runtime_error("unref_block: refcount already 0");
  }
  meta.refcount--;

  if (meta.refcount == 0) {
    if (meta.content_hash != 0 && meta.is_full) {
      // Keep in cache for potential reuse.
      evictable_push_back(block_id);
    } else {
      // No cached content — return to free list.
      if (meta.content_hash != 0) {
        hash_to_block_.erase(meta.content_hash);
      }
      meta = BlockMeta{};
      free_list_.push_back(block_id);
    }
  }
}

void BlockManager::mark_block_full(int32_t block_id, uint64_t content_hash) {
  if (!prefix_caching_enabled_ || content_hash == 0) return;
  auto& meta = block_meta_[block_id];
  meta.content_hash = content_hash;
  meta.is_full = true;
  hash_to_block_[content_hash] = block_id;
}

int64_t BlockManager::free_block_count() const {
  if (prefix_caching_enabled_) {
    return static_cast<int64_t>(free_list_.size()) +
           static_cast<int64_t>(evictable_list_.size());
  }
  return static_cast<int64_t>(free_list_.size());
}

void BlockManager::evictable_remove(int32_t block_id) {
  auto it = evictable_index_.find(block_id);
  if (it != evictable_index_.end()) {
    evictable_list_.erase(it->second);
    evictable_index_.erase(it);
  }
}

void BlockManager::evictable_push_back(int32_t block_id) {
  evictable_list_.push_back(block_id);
  evictable_index_[block_id] = std::prev(evictable_list_.end());
}

torch::Tensor& BlockManager::k_blocks(int64_t layer) {
  return k_blocks_.at(layer);
}

torch::Tensor& BlockManager::v_blocks(int64_t layer) {
  return v_blocks_.at(layer);
}

const torch::Tensor& BlockManager::k_blocks(int64_t layer) const {
  return k_blocks_.at(layer);
}

const torch::Tensor& BlockManager::v_blocks(int64_t layer) const {
  return v_blocks_.at(layer);
}

}  // namespace tvllm
