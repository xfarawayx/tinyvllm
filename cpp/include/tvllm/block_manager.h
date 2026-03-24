#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace tvllm {

/// Per-block metadata for prefix caching.
struct BlockMeta {
  int32_t refcount = 0;        // number of sequences using this block
  uint64_t content_hash = 0;   // 0 = no cached content
  bool is_full = false;        // only full blocks are cacheable
};

/// Compute a chained hash over a block of token IDs.
/// prefix_hash chains to all preceding blocks, ensuring block N is only
/// reusable when blocks 0..N-1 also matched.
static inline uint64_t hash_block_tokens(uint64_t prefix_hash,
                                          const int64_t* tokens,
                                          int64_t count) {
  uint64_t h = prefix_hash;
  for (int64_t i = 0; i < count; ++i) {
    h ^= std::hash<int64_t>{}(tokens[i]) +
         0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  }
  return h;
}

class BlockManager {
 public:
  BlockManager(int64_t num_layers,
               int64_t num_blocks,
               int64_t block_size,
               int64_t num_key_value_heads,
               int64_t head_dim,
               torch::Device device,
               torch::Dtype dtype,
               bool prefix_caching = false);

  void reset();

  /// Allocate a fresh block (refcount set to 1 when prefix caching enabled).
  /// When the free list is empty and prefix caching is on, evicts the LRU
  /// cached block.
  int32_t alloc_block();

  /// Free a block. When prefix caching is disabled, returns it to the free
  /// list directly. When enabled, delegates to unref_block().
  void free_block(int32_t block_id);

  /// Look up a cached block by content hash.  Returns block_id if found
  /// (increments refcount), or -1 if not cached.
  int32_t lookup_cached(uint64_t content_hash);

  /// Increment the reference count of a block.
  void ref_block(int32_t block_id);

  /// Decrement the reference count.  If it reaches 0 and the block has a
  /// valid content hash, the block enters the evictable list.  Otherwise it
  /// goes back to the free list.
  void unref_block(int32_t block_id);

  /// Mark a block as full and register its content hash for future reuse.
  void mark_block_full(int32_t block_id, uint64_t content_hash);

  torch::Tensor& k_blocks(int64_t layer);
  torch::Tensor& v_blocks(int64_t layer);
  const torch::Tensor& k_blocks(int64_t layer) const;
  const torch::Tensor& v_blocks(int64_t layer) const;

  int64_t block_size() const { return block_size_; }
  int64_t num_blocks() const { return num_blocks_; }
  bool prefix_caching_enabled() const { return prefix_caching_enabled_; }

  /// Number of available blocks (free list + evictable when prefix caching).
  int64_t free_block_count() const;

  /// Access per-block metadata (for skip-write checks in KV cache).
  const BlockMeta& block_meta(int32_t block_id) const { return block_meta_[block_id]; }

 private:
  int64_t num_layers_;
  int64_t num_blocks_;
  int64_t block_size_;
  bool prefix_caching_enabled_;

  std::vector<torch::Tensor> k_blocks_;
  std::vector<torch::Tensor> v_blocks_;
  std::vector<int32_t> free_list_;

  // --- Prefix caching structures ---
  std::vector<BlockMeta> block_meta_;
  std::unordered_map<uint64_t, int32_t> hash_to_block_;
  std::list<int32_t> evictable_list_;   // LRU order: front = oldest
  std::unordered_map<int32_t, std::list<int32_t>::iterator> evictable_index_;

  void evictable_remove(int32_t block_id);
  void evictable_push_back(int32_t block_id);
};

}  // namespace tvllm
