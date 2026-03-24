#pragma once

#include <string>

namespace tvllm {

struct ModelConfig {
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t max_position_embeddings = 0;
  double rope_theta = 10000.0;
  double rms_norm_eps = 1e-5;
  bool tie_word_embeddings = false;
  int64_t bos_token_id = -1;
  int64_t eos_token_id = -1;
  bool use_bias = false;
  int64_t head_dim = 0;   // 0 = derive as hidden_size / num_attention_heads
  bool use_qk_norm = false;
  bool use_nf4 = false;

  static ModelConfig from_file(const std::string& path);
};

}  // namespace tvllm
