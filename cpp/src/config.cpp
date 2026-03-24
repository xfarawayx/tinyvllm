#include "tvllm/config.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tvllm {

static std::unordered_map<std::string, std::string> parse_kv_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open config file: " + path);
  }
  std::unordered_map<std::string, std::string> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    auto key = line.substr(0, pos);
    auto val = line.substr(pos + 1);
    out[key] = val;
  }
  return out;
}

static int64_t get_int(const std::unordered_map<std::string, std::string>& dict,
                       const std::string& key,
                       int64_t def_val) {
  auto it = dict.find(key);
  if (it == dict.end()) {
    return def_val;
  }
  return std::stoll(it->second);
}

static double get_double(const std::unordered_map<std::string, std::string>& dict,
                         const std::string& key,
                         double def_val) {
  auto it = dict.find(key);
  if (it == dict.end()) {
    return def_val;
  }
  return std::stod(it->second);
}

static bool get_bool(const std::unordered_map<std::string, std::string>& dict,
                     const std::string& key,
                     bool def_val) {
  auto it = dict.find(key);
  if (it == dict.end()) {
    return def_val;
  }
  auto val = it->second;
  for (auto& ch : val) {
    ch = static_cast<char>(std::tolower(ch));
  }
  return val == "true" || val == "1";
}

ModelConfig ModelConfig::from_file(const std::string& path) {
  auto dict = parse_kv_file(path);

  ModelConfig cfg;
  cfg.vocab_size = get_int(dict, "vocab_size", 0);
  cfg.hidden_size = get_int(dict, "hidden_size", 0);
  cfg.intermediate_size = get_int(dict, "intermediate_size", 0);
  cfg.num_hidden_layers = get_int(dict, "num_hidden_layers", 0);
  cfg.num_attention_heads = get_int(dict, "num_attention_heads", 0);
  cfg.num_key_value_heads = get_int(dict, "num_key_value_heads", cfg.num_attention_heads);
  cfg.max_position_embeddings = get_int(dict, "max_position_embeddings", 0);
  cfg.rope_theta = get_double(dict, "rope_theta", 10000.0);
  cfg.rms_norm_eps = get_double(dict, "rms_norm_eps", 1e-5);
  cfg.tie_word_embeddings = get_bool(dict, "tie_word_embeddings", false);
  cfg.bos_token_id = get_int(dict, "bos_token_id", -1);
  cfg.eos_token_id = get_int(dict, "eos_token_id", -1);
  cfg.use_bias = get_bool(dict, "use_bias", false);
  cfg.head_dim = get_int(dict, "head_dim", 0);
  cfg.use_qk_norm = get_bool(dict, "use_qk_norm", false);
  cfg.use_nf4 = get_bool(dict, "use_nf4", false);

  if (cfg.hidden_size <= 0 || cfg.num_attention_heads <= 0 ||
      cfg.num_hidden_layers <= 0 || cfg.vocab_size <= 0) {
    throw std::runtime_error("invalid model config");
  }
  return cfg;
}

}  // namespace tvllm
