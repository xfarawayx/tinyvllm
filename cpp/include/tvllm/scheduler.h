#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

#include "tvllm/sampler.h"

namespace tvllm {

/// Per-request state tracked by the scheduler.
struct SequenceRequest {
  int64_t request_id = -1;
  std::vector<int64_t> prompt_ids;
  SampleParams sample;
  std::vector<int64_t> output_ids;   // prompt + generated tokens
  int64_t seq_cache_id = -1;         // KV-cache sequence id
  int64_t tokens_in_cache = 0;       // tokens already materialized in KV cache
  int64_t generated_tokens = 0;
  int64_t last_token = -1;           // last sampled token (for next decode input)
  int64_t prefix_cached_tokens = 0;  // tokens matched from prefix cache
};

/// Output of a single scheduling step.
struct SchedulerOutput {
  std::vector<SequenceRequest*> to_prefill;  // newly admitted from waiting
  std::vector<SequenceRequest*> to_decode;   // already running
};

/// Simple FCFS continuous-batching scheduler.
///
/// Maintains a waiting queue and a running set.  Each call to schedule()
/// admits as many waiting requests as there are free slots (up to
/// max_batch_size), and returns them for prefill.  Already-running
/// sequences are returned for decode.
///
/// Admission is gated by three constraints:
///   1. max_batch_size  — at most this many concurrent running sequences.
///   2. free KV-cache blocks — enough blocks to cover the request's
///      worst-case KV footprint (prompt + max_new_tokens), while reserving
///      future growth for sequences that are already running.
///   3. max_num_batched_tokens — total tokens across all prefills in one
///      step must not exceed this limit (prevents activation OOM).
class Scheduler {
 public:
  Scheduler(int64_t max_batch_size,
            int64_t block_size = 0,
            int64_t max_num_batched_tokens = 0);

  /// Enqueue a new request.  Returns the assigned request id.
  int64_t add_request(std::vector<int64_t> prompt_ids, SampleParams sample);

  /// Decide what to prefill and what to decode this step.
  /// @param free_blocks  Number of currently free KV-cache blocks.
  ///                     Pass -1 to disable block-aware admission control.
  SchedulerOutput schedule(int64_t free_blocks = -1);

  /// Mark a request as finished and free its running slot.
  void finish_request(int64_t request_id);

  bool has_unfinished() const;

 private:
  int64_t blocks_for_tokens(int64_t num_tokens) const;
  int64_t reserved_blocks_for_request(const SequenceRequest& req) const;
  int64_t reserved_future_blocks_for_request(const SequenceRequest& req) const;

  int64_t max_batch_size_;
  int64_t block_size_;
  int64_t max_num_batched_tokens_;
  int64_t next_id_ = 0;
  std::deque<int64_t> waiting_;       // request ids awaiting prefill
  std::vector<int64_t> running_;      // request ids currently decoding
  std::unordered_map<int64_t, SequenceRequest> requests_;
};

}  // namespace tvllm
