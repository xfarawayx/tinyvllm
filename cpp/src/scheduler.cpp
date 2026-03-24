#include "tvllm/scheduler.h"

#include <algorithm>
#include <stdexcept>

namespace tvllm {

int64_t Scheduler::blocks_for_tokens(int64_t num_tokens) const {
  if (block_size_ <= 0 || num_tokens <= 0) {
    return 0;
  }
  return (num_tokens + block_size_ - 1) / block_size_;
}

int64_t Scheduler::reserved_blocks_for_request(const SequenceRequest& req) const {
  int64_t max_new_tokens = std::max<int64_t>(req.sample.max_new_tokens, 0);
  int64_t total_tokens = static_cast<int64_t>(req.prompt_ids.size()) + max_new_tokens;
  int64_t total_blocks = blocks_for_tokens(total_tokens);
  // Cached prefix blocks are already allocated (shared) — don't count them
  // as new blocks needed from the free pool.
  int64_t cached_blocks = blocks_for_tokens(req.prefix_cached_tokens);
  return std::max<int64_t>(total_blocks - cached_blocks, 0);
}

int64_t Scheduler::reserved_future_blocks_for_request(const SequenceRequest& req) const {
  int64_t reserved_blocks = reserved_blocks_for_request(req);
  int64_t allocated_blocks = blocks_for_tokens(req.tokens_in_cache);
  return std::max<int64_t>(reserved_blocks - allocated_blocks, 0);
}

Scheduler::Scheduler(int64_t max_batch_size,
                     int64_t block_size,
                     int64_t max_num_batched_tokens)
    : max_batch_size_(max_batch_size),
      block_size_(block_size),
      max_num_batched_tokens_(max_num_batched_tokens) {
  if (max_batch_size_ <= 0) {
    throw std::runtime_error("max_batch_size must be positive");
  }
}

int64_t Scheduler::add_request(std::vector<int64_t> prompt_ids,
                                SampleParams sample) {
  int64_t id = next_id_++;
  SequenceRequest req;
  req.request_id = id;
  req.prompt_ids = std::move(prompt_ids);
  req.sample = sample;
  requests_.emplace(id, std::move(req));
  waiting_.push_back(id);
  return id;
}

SchedulerOutput Scheduler::schedule(int64_t free_blocks) {
  SchedulerOutput out;

  // Existing running sequences need a decode step.
  for (auto id : running_) {
    out.to_decode.push_back(&requests_.at(id));
  }

  // Keep enough free blocks in reserve for already-running sequences to
  // continue decoding up to their configured max_new_tokens.
  if (free_blocks >= 0 && block_size_ > 0) {
    int64_t reserved_future_blocks = 0;
    for (auto id : running_) {
      reserved_future_blocks +=
          reserved_future_blocks_for_request(requests_.at(id));
    }
    free_blocks = std::max<int64_t>(free_blocks - reserved_future_blocks, 0);
  }

  // Admit new sequences from the waiting queue into free slots.
  int64_t free_slots = max_batch_size_ - static_cast<int64_t>(running_.size());
  int64_t batched_tokens = 0;

  while (free_slots > 0 && !waiting_.empty()) {
    int64_t req_id = waiting_.front();
    const auto& req = requests_.at(req_id);
    int64_t prompt_len = static_cast<int64_t>(req.prompt_ids.size());

    // Token budget: prevent activation OOM by limiting total prefill tokens.
    // Only non-cached tokens go through the prefill forward pass.
    int64_t effective_prompt_len = prompt_len - req.prefix_cached_tokens;
    if (max_num_batched_tokens_ > 0 && batched_tokens > 0) {
      if (batched_tokens + effective_prompt_len > max_num_batched_tokens_) {
        break;  // would exceed token budget; defer to next step
      }
    }

    // Block-aware admission: reserve enough space for the request's
    // worst-case KV-cache footprint so decode growth cannot exhaust the pool.
    if (free_blocks >= 0 && block_size_ > 0) {
      int64_t blocks_needed = reserved_blocks_for_request(req);
      if (blocks_needed > free_blocks) {
        break;  // not enough blocks — wait for running requests to finish
      }
      free_blocks -= blocks_needed;
    }

    waiting_.pop_front();
    running_.push_back(req_id);
    out.to_prefill.push_back(&requests_.at(req_id));
    batched_tokens += effective_prompt_len;
    --free_slots;
  }

  return out;
}

void Scheduler::finish_request(int64_t request_id) {
  running_.erase(
      std::remove(running_.begin(), running_.end(), request_id),
      running_.end());
}

bool Scheduler::has_unfinished() const {
  return !waiting_.empty() || !running_.empty();
}

}  // namespace tvllm
