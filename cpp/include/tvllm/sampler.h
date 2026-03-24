#pragma once

#include <cstdint>
#include <vector>
#include <torch/torch.h>

namespace tvllm {

/// Per-request sampling parameters.
struct SampleParams {
  int64_t max_new_tokens = 0;
  double temperature = 0.0;
  bool ignore_eos = false;
};

/// Batched greedy sampling: argmax over [batch_size, vocab_size] logits,
/// returns all token IDs in one CPU vector with a single GPU→CPU sync.
std::vector<int64_t> greedy_sample_batch(const torch::Tensor& logits);

/// Batched sampling with per-request temperature.
/// Requests with temperature <= 0 fall back to greedy sampling.
std::vector<int64_t> sample_batch(const torch::Tensor& logits,
                                  const std::vector<SampleParams>& sample_params);

}  // namespace tvllm
