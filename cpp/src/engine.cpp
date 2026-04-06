#include "tvllm/engine.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <torch/torch.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Allow compilation without nvcc — the header is still available when
// linking against CUDA runtime.
#include <cuda_runtime_api.h>
#endif
#include <c10/cuda/CUDACachingAllocator.h>

#include "tvllm/kv_cache.h"
#include "tvllm/sampler.h"
#include "tvllm/scheduler.h"

namespace tvllm {

namespace {

template <typename T, typename = void>
struct HasStatMember : std::false_type {};

template <typename T>
struct HasStatMember<T, std::void_t<decltype(std::declval<const T&>().stat)>>
    : std::true_type {};

template <typename T>
const c10::cuda::CUDACachingAllocator::Stat& aggregate_allocator_stat(
    const T& stats) {
  if constexpr (HasStatMember<T>::value) {
    return stats.stat[0];
  }
  return stats[0];
}

}  // namespace

static torch::Device resolve_device() {
  if (!torch::cuda::is_available()) {
    throw std::runtime_error("tinyvllm requires CUDA, but no CUDA device is available");
  }
  return torch::kCUDA;
}

static torch::Dtype resolve_dtype() {
  return torch::kBFloat16;
}

static int64_t resolve_block_size() {
  const char* env = std::getenv("TVLLM_BLOCK_SIZE");
  if (!env) {
    return 32;
  }
  try {
    int64_t value = std::stoll(env);
    return value > 0 ? value : 32;
  } catch (...) {
    return 32;
  }
}

static bool resolve_prefix_caching() {
  const char* env = std::getenv("TVLLM_PREFIX_CACHING");
  return env && std::string(env) == "1";
}

static int64_t resolve_max_num_batched_tokens() {
  const char* env = std::getenv("TVLLM_MAX_NUM_BATCHED_TOKENS");
  if (!env) {
    return 8192;
  }
  try {
    int64_t val = std::stoll(env);
    return val > 0 ? val : 8192;
  } catch (...) {
    return 8192;
  }
}

// ---------------------------------------------------------------------------
// Profile-based memory budgeting (following nanovllm / vLLM approach)
//
// Instead of estimating activation memory with a formula, we run an actual
// warmup forward pass at max_num_batched_tokens and measure peak memory
// via PyTorch's CUDACachingAllocator stats.  This naturally captures every
// cost that formula estimation misses: FlashInfer workspace, allocator
// fragmentation, NF4 dequantization temporaries, etc.
//
// Flow:
//   1. Create a temporary KV cache (just enough blocks for profiling).
//   2. Reset peak memory stats (temp cache + model weights = baseline).
//   3. Run forward_prefill with max_num_batched_tokens dummy tokens.
//   4. activation_overhead = peak - current  (peak above steady state).
//   5. Destroy temp cache, release memory.
//   6. num_kv_blocks = (free - activation_overhead) / per_block_bytes.
//
// Activation is a hard constraint (OOM crash if exceeded).  KV cache is a
// soft constraint (scheduler defers requests if blocks run out).  By
// measuring activation first and giving the rest to KV cache, we guarantee
// that OOM cannot happen at the profiled batch size.
// ---------------------------------------------------------------------------

void Engine::profile_memory() {
  block_size_ = resolve_block_size();
  max_num_batched_tokens_ = resolve_max_num_batched_tokens();

  int64_t head_dim = config_.head_dim > 0
      ? config_.head_dim
      : config_.hidden_size / config_.num_attention_heads;
  int64_t dtype_bytes = static_cast<int64_t>(torch::elementSize(dtype_));

  c10::DeviceIndex dev_idx = device_.has_index() ? device_.index() : 0;

  // --- Step 1: warmup forward to measure peak activation memory ----------

  // Temporary KV cache with just enough blocks for the profiling run.
  int64_t profile_blocks =
      (max_num_batched_tokens_ + block_size_ - 1) / block_size_ + 1;

  int64_t activation_peak = 0;
  {
    PagedKVCache temp_cache(config_, device_, dtype_,
                            block_size_, profile_blocks, /*prefix_caching=*/false);
    int64_t seq_id = temp_cache.add_sequence();

    // Reset peak stats AFTER the temp cache allocation so that the cache
    // memory becomes part of the baseline and does not inflate the
    // activation measurement.
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::resetPeakStats(dev_idx);

    // Record baseline reserved memory.  After emptyCache the allocator
    // holds no cached blocks, so reserved == allocated == only live
    // tensors (model weights + temp KV cache).
    int64_t reserved_baseline;
    {
      auto bs = c10::cuda::CUDACachingAllocator::getDeviceStats(dev_idx);
      reserved_baseline =
          aggregate_allocator_stat(bs.reserved_bytes).current;
    }

    // Forward pass with dummy tokens.
    std::vector<std::vector<int64_t>> dummy_ids = {
        std::vector<int64_t>(max_num_batched_tokens_, 0)
    };
    std::vector<int64_t> seq_ids = {seq_id};
    std::vector<int64_t> start_pos = {0};

    model_->forward_prefill(dummy_ids, seq_ids, temp_cache, start_pos);
    cudaDeviceSynchronize();

    // Use reserved_bytes (actual CUDA memory claimed by the caching
    // allocator) instead of allocated_bytes (what callers requested).
    // For NF4 models the allocation churn from repeated
    // dequant-temp → free → matmul cycles causes the allocator to
    // reserve significantly more CUDA memory than the peak
    // outstanding allocation.  Using reserved_bytes captures this
    // overhead and prevents over-allocating KV-cache blocks.
    auto stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(dev_idx);
    int64_t reserved_peak =
        aggregate_allocator_stat(stats.reserved_bytes).peak;
    activation_peak = reserved_peak - reserved_baseline;

    temp_cache.remove_sequence(seq_id);
    // ~PagedKVCache frees the block pool tensors when this scope exits.
  }

  // --- Step 2: compute KV-cache blocks from remaining memory -------------

  c10::cuda::CUDACachingAllocator::emptyCache();

  size_t free_bytes = 0, total_bytes = 0;
  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("profile_memory: cudaMemGetInfo failed: ") +
        cudaGetErrorString(err));
  }

  // All free GPU memory (after model weights, FlashInfer workspace, etc.)
  // minus the profiled activation peak goes to KV cache.
  int64_t per_block_bytes = 2 * config_.num_hidden_layers * block_size_ *
                            config_.num_key_value_heads * head_dim * dtype_bytes;

  int64_t available_for_kv = static_cast<int64_t>(free_bytes) - activation_peak;
  num_kv_blocks_ = available_for_kv / per_block_bytes;
  num_kv_blocks_ = std::max(num_kv_blocks_, int64_t{1});

  fprintf(stderr,
      "[profile_memory] max_num_batched_tokens=%ld\n"
      "  GPU total=%.0f MiB, free=%.0f MiB\n"
      "  activation peak=%.0f MiB (profiled)\n"
      "  kv_cache: %ld blocks (%.0f MiB)\n",
      max_num_batched_tokens_,
      total_bytes / (1024.0 * 1024.0),
      free_bytes / (1024.0 * 1024.0),
      activation_peak / (1024.0 * 1024.0),
      num_kv_blocks_,
      (num_kv_blocks_ * per_block_bytes) / (1024.0 * 1024.0));

  // Construct the persistent KV cache once.
  prefix_caching_ = resolve_prefix_caching();
  cache_ = std::make_unique<PagedKVCache>(
      config_, device_, dtype_, block_size_, num_kv_blocks_, prefix_caching_);
}

Engine::Engine(const std::string& model_path)
    : device_(resolve_device()), dtype_(resolve_dtype()) {
  torch::NoGradGuard no_grad;

  auto config_path = model_path + "/config.txt";
  auto weights_path = model_path + "/state_dict.pt";

  config_ = ModelConfig::from_file(config_path);

  std::ifstream in(weights_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open weights file: " + weights_path);
  }
  in.seekg(0, std::ios::end);
  auto file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<char> data(file_size);
  in.read(data.data(), file_size);
  auto iv = torch::pickle_load(data);
  if (!iv.isGenericDict()) {
    throw std::runtime_error("state_dict pickle must be a dict");
  }
  auto state_dict = iv.toGenericDict();

  model_ = std::make_unique<QwenModel>(config_, state_dict, device_, dtype_);
  profile_memory();
}

Engine::Engine(const std::string& model_dir, const c10::impl::GenericDict& state_dict)
  : device_(resolve_device()), dtype_(resolve_dtype()) {
  torch::NoGradGuard no_grad;

  auto config_path = model_dir + "/config.txt";
  config_ = ModelConfig::from_file(config_path);

  model_ = std::make_unique<QwenModel>(config_, state_dict, device_, dtype_);
  profile_memory();
}

// ---------------------------------------------------------------------------
// Continuous-batching generate
// ---------------------------------------------------------------------------
std::vector<std::vector<int64_t>> Engine::generate(
    const std::vector<std::vector<int64_t>>& batch_input_ids,
    const std::vector<SampleParams>& sample_params,
    int64_t max_batch_size) {
  torch::NoGradGuard no_grad;

  if (batch_input_ids.empty()) {
    throw std::runtime_error("batch_input_ids cannot be empty");
  }

  int64_t num_requests = static_cast<int64_t>(batch_input_ids.size());
  if (static_cast<int64_t>(sample_params.size()) != num_requests) {
    throw std::runtime_error(
        "sample_params length must match batch_input_ids length");
  }
  for (const auto& seq : batch_input_ids) {
    if (seq.empty()) {
      throw std::runtime_error("each sequence in batch_input_ids must be non-empty");
    }
  }

  if (max_batch_size <= 0) {
    max_batch_size = num_requests;
  }

  auto options = torch::TensorOptions().device(device_).dtype(torch::kLong);

  std::vector<std::vector<int64_t>> outputs(num_requests);

  PagedKVCache& cache = *cache_;
  Scheduler scheduler(max_batch_size, block_size_, max_num_batched_tokens_);

  std::unordered_map<int64_t, int64_t> rid_to_idx;
  for (int64_t i = 0; i < num_requests; ++i) {
    int64_t rid = scheduler.add_request(batch_input_ids[i], sample_params[i]);
    rid_to_idx[rid] = i;
  }

  fprintf(stderr, "[continuous batching] %ld requests, max_batch_size=%ld, "
          "num_blocks=%ld, block_size=%ld, max_batched_tokens=%ld\n",
          num_requests, max_batch_size, num_kv_blocks_, block_size_,
          max_num_batched_tokens_);

  int64_t iteration = 0;
  while (scheduler.has_unfinished()) {
    auto step = scheduler.schedule(cache.free_block_count());
    ++iteration;

    // ---- Prefill newly admitted sequences --------------------------------
    if (!step.to_prefill.empty()) {
      std::vector<std::vector<int64_t>> prefill_ids_list;
      std::vector<int64_t> prefill_cache_ids;
      std::vector<int64_t> prefill_start_positions;

      for (auto* req : step.to_prefill) {
        req->output_ids = req->prompt_ids;

        if (prefix_caching_) {
          auto [seq_id, matched] = cache.add_sequence_with_prefix(req->prompt_ids);
          req->seq_cache_id = seq_id;
          req->prefix_cached_tokens = matched;

          if (matched >= static_cast<int64_t>(req->prompt_ids.size())) {
            // Entire prompt is cached. We still need to run a minimal prefill
            // of the last token to obtain logits for sampling the first
            // generated token.  The KV append will skip writing (shared block).
            int64_t last_pos = static_cast<int64_t>(req->prompt_ids.size()) - 1;
            prefill_ids_list.push_back({req->prompt_ids.back()});
            prefill_start_positions.push_back(last_pos);
            prefill_cache_ids.push_back(req->seq_cache_id);
            continue;
          }

          // Only send non-cached suffix to prefill.
          auto suffix_ids = std::vector<int64_t>(
              req->prompt_ids.begin() + matched, req->prompt_ids.end());
          prefill_ids_list.push_back(std::move(suffix_ids));
          prefill_start_positions.push_back(matched);
        } else {
          req->seq_cache_id = cache.add_sequence();
          prefill_ids_list.push_back(req->prompt_ids);
          prefill_start_positions.push_back(0);
        }
        prefill_cache_ids.push_back(req->seq_cache_id);
      }

      // Run prefill for sequences that have non-cached tokens.
      if (!prefill_ids_list.empty()) {
        auto logits_batch = model_->forward_prefill(
            prefill_ids_list, prefill_cache_ids, cache, prefill_start_positions);

        // Sample first generated token for each newly prefilled sequence.
        std::vector<SampleParams> step_sample_params;
        step_sample_params.reserve(step.to_prefill.size());
        for (auto* req : step.to_prefill) {
          step_sample_params.push_back(req->sample);
        }
        auto sampled_tokens = sample_batch(logits_batch, step_sample_params);
        for (int64_t i = 0; i < static_cast<int64_t>(step.to_prefill.size()); ++i) {
          auto* req = step.to_prefill[i];
          int64_t token = sampled_tokens[i];
          req->tokens_in_cache = static_cast<int64_t>(req->prompt_ids.size());
          bool hit_eos = token == config_.eos_token_id && !req->sample.ignore_eos;

          if (hit_eos || req->sample.max_new_tokens <= 0) {
            outputs[rid_to_idx[req->request_id]] = req->output_ids;
            cache.remove_sequence(req->seq_cache_id);
            scheduler.finish_request(req->request_id);
            fprintf(stderr, "  [iter %ld] req %ld finished at prefill (EOS)\n",
                    iteration, req->request_id);
          } else {
            req->output_ids.push_back(token);
            req->generated_tokens = 1;
            req->last_token = token;
          }
        }
      }
    }

    // ---- Decode running sequences ----------------------------------------
    // Each running sequence has a last_token to feed into the decode step.
    if (!step.to_decode.empty()) {
      std::vector<SequenceRequest*> active;
      std::vector<int64_t> tokens, positions, cache_ids;

      for (auto* req : step.to_decode) {
        if (req->last_token < 0) continue;  // shouldn't happen
        active.push_back(req);
        tokens.push_back(req->last_token);
        positions.push_back(cache.seq_len(req->seq_cache_id));
        cache_ids.push_back(req->seq_cache_id);
      }

      if (!active.empty()) {
        int64_t bsz = static_cast<int64_t>(active.size());
        auto input_ids = torch::tensor(tokens, options).view({bsz, 1});
        auto pos_tensor = torch::tensor(positions, options).view({bsz, 1});

        auto logits_batch = model_->forward_decode(
            input_ids, pos_tensor, cache_ids, cache, positions);

        std::vector<SampleParams> step_sample_params;
        step_sample_params.reserve(active.size());
        for (auto* req : active) {
          step_sample_params.push_back(req->sample);
        }
        auto sampled_tokens = sample_batch(logits_batch, step_sample_params);
        for (int64_t i = 0; i < bsz; ++i) {
          auto* req = active[i];
          int64_t token = sampled_tokens[i];
          int64_t cached_len = cache.seq_len(req->seq_cache_id);
          req->tokens_in_cache = cached_len;
          bool hit_eos = token == config_.eos_token_id && !req->sample.ignore_eos;

          if (hit_eos ||
              req->generated_tokens >= req->sample.max_new_tokens ||
              cached_len >= config_.max_position_embeddings) {
            outputs[rid_to_idx[req->request_id]] = req->output_ids;
            cache.remove_sequence(req->seq_cache_id);
            scheduler.finish_request(req->request_id);
            // fprintf(stderr, "  [iter %ld] req %ld finished (%ld tokens generated)\n",
            //         iteration, req->request_id, req->generated_tokens);
          } else {
            req->output_ids.push_back(token);
            req->generated_tokens++;
            req->last_token = token;
          }
        }
      }
    }
  }

  fprintf(stderr, "[continuous batching] done in %ld iterations\n", iteration);
  return outputs;
}

torch::Tensor Engine::forward_logits(
    const std::vector<std::vector<int64_t>>& batch_input_ids) {
  torch::NoGradGuard no_grad;

  if (batch_input_ids.empty()) {
    throw std::runtime_error("batch_input_ids cannot be empty");
  }
  for (const auto& seq : batch_input_ids) {
    if (seq.empty()) {
      throw std::runtime_error("each sequence must be non-empty");
    }
  }

  PagedKVCache& cache = *cache_;

  // Allocate temporary sequences in the KV cache.
  std::vector<int64_t> seq_ids;
  seq_ids.reserve(batch_input_ids.size());
  for (size_t i = 0; i < batch_input_ids.size(); ++i) {
    seq_ids.push_back(cache.add_sequence());
  }

  std::vector<int64_t> start_pos(batch_input_ids.size(), 0);
  auto logits = model_->forward_prefill(
      batch_input_ids, seq_ids, cache, start_pos, /*all_logits=*/true);

  // Clean up temporary sequences.
  for (auto sid : seq_ids) {
    cache.remove_sequence(sid);
  }

  return logits;
}

void Engine::reset() {
  if (cache_) {
    cache_->reset();
  }
}

}  // namespace tvllm
