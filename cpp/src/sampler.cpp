#include "tvllm/sampler.h"

#include <stdexcept>

namespace tvllm {

std::vector<int64_t> greedy_sample_batch(const torch::Tensor& logits) {
  // logits: [batch_size, vocab_size]
  // Single argmax + single GPU→CPU transfer instead of N .item() calls
  auto tokens = std::get<1>(logits.max(-1));        // [batch_size] on GPU
  auto tokens_cpu = tokens.to(torch::kCPU);         // one sync
  auto accessor = tokens_cpu.accessor<int64_t, 1>();
  std::vector<int64_t> result(accessor.size(0));
  for (int64_t i = 0; i < accessor.size(0); ++i) {
    result[i] = accessor[i];
  }
  return result;
}

std::vector<int64_t> sample_batch(const torch::Tensor& logits,
                                  const std::vector<SampleParams>& sample_params) {
  if (logits.dim() != 2) {
    throw std::runtime_error("sample_batch expects logits with shape [batch_size, vocab_size]");
  }

  int64_t batch_size = logits.size(0);
  if (static_cast<int64_t>(sample_params.size()) != batch_size) {
    throw std::runtime_error("sample_batch sample_params size must match batch size");
  }
  if (batch_size == 0) {
    return {};
  }

  bool all_greedy = true;
  bool any_greedy = false;
  std::vector<float> temperatures(batch_size, 1.0f);
  for (int64_t i = 0; i < batch_size; ++i) {
    double temperature = sample_params[i].temperature;
    if (temperature > 0.0) {
      all_greedy = false;
      temperatures[i] = static_cast<float>(temperature);
    } else {
      any_greedy = true;
    }
  }

  if (all_greedy) {
    return greedy_sample_batch(logits);
  }

  auto temp_tensor = torch::tensor(
      temperatures,
      torch::TensorOptions().device(logits.device()).dtype(torch::kFloat32));
  auto scaled_logits = logits.to(torch::kFloat32) / temp_tensor.unsqueeze(1);
  auto probs = torch::softmax(scaled_logits, -1);
  auto sampled = torch::multinomial(probs, 1).squeeze(1);
  auto sampled_cpu = sampled.to(torch::kCPU);
  auto accessor = sampled_cpu.accessor<int64_t, 1>();
  std::vector<int64_t> result(accessor.size(0));
  for (int64_t i = 0; i < accessor.size(0); ++i) {
    result[i] = accessor[i];
  }

  if (any_greedy) {
    auto greedy = greedy_sample_batch(logits);
    for (int64_t i = 0; i < batch_size; ++i) {
      if (sample_params[i].temperature <= 0.0) {
        result[i] = greedy[i];
      }
    }
  }

  return result;
}

}  // namespace tvllm
