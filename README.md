# tinyvllm

A minimal LLM inference engine for Qwen2.5 / Qwen3 models, built from scratch in C++/CUDA with a Python API via pybind11. Implements core techniques from the vLLM paper — paged KV-cache, continuous batching, prefix caching, and custom CUDA kernels — as a learning-oriented codebase.

## Features

- **Qwen2.5 & Qwen3 support** — including optional per-head QK-norm for Qwen3
- **FlashInfer-backed attention** — runtime prefill and decode use FlashInfer wrappers, with plan/run split to minimize GPU→CPU syncs
- **Fused CUDA kernels** — RMSNorm (fp32 accumulation), RoPE (in-place on Q+K), SwiGLU activation, KV scatter for decode — each replacing multi-op libtorch calls with a single kernel launch
- **Paged KV-cache** — fixed-size block pool (default 16 tokens/block), dynamic allocation, automatic GPU memory budgeting via `cudaMemGetInfo`
- **Prefix caching** — content-addressable block sharing with chained hashing and LRU eviction
- **NF4 quantization** — double-quantized 4-bit weights with on-the-fly CUDA dequantization (bitsandbytes-compatible)
- **Continuous batching** — FCFS scheduler with three admission constraints: max batch size, free KV-cache blocks, and max batched tokens
- **Per-request sampling** — `temperature` sampling and `ignore_eos`; `temperature <= 0` falls back to greedy decoding
- **fp16 / bf16** — both data types supported across all kernels
- **Head dim 64 / 128** — compile-time dispatch for both common head dimensions

## Prerequisites

- Python 3.9+
- PyTorch (with CUDA) and `transformers`
- FlashInfer
- CUDA toolkit + NVIDIA GPU
- CMake >= 3.18, C++17 compiler

## Build

```bash
# Using the build script (auto-detects Python, CUDA, PyTorch paths):
./make.sh

# Or manually:
mkdir -p build && cd build
cmake .. && make -j
```

Set `CMAKE_CUDA_ARCHITECTURES` to target a different GPU (default `80` / A100):

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. && make -j
# or via the build script:
CUDA_ARCH=89 ./make.sh
```

The build produces:
- `build/tinyvllm.*.so` — Python extension module

## Convert weights

Convert HuggingFace Qwen2.5/Qwen3 weights to tinyvllm format:

```bash
# fp16/bf16
python python/convert_qwen25.py \
  --model /path/to/Qwen2.5-7B \
  --output /path/to/converted_weights \
  --dtype bf16

# NF4 quantized (requires bitsandbytes>=0.43)
python python/convert_qwen25_nf4.py \
  --model /path/to/Qwen2.5-7B \
  --output /path/to/converted_weights_nf4 \
  --dtype bf16
```

This creates `config.txt` (model hyperparameters) and `state_dict.pt` (weight tensors) in the output directory.

## Run inference

```bash
# Single-turn generation
PYTHONPATH=build \
python python/run_generate.py \
  --model_dir /path/to/converted_weights \
  --hf_model /path/to/Qwen2.5-7B \
  --prompts "Hello" "How are you" \
  --max_new_tokens 32 \
  --max_batch_size 4

# Interactive multi-turn chat
PYTHONPATH=build \
python python/chat.py \
  --model_dir /path/to/converted_weights \
  --hf_model /path/to/Qwen2.5-7B \
  --max_new_tokens 512 \
  --temperature 0.7
```

The `--hf_model` path is used only for the tokenizer.

## Benchmark

```bash
PYTHONPATH=build \
python python/benchmark.py \
  --model_dir /path/to/converted_weights \
  --num_requests 256 \
  --input_len_min 100 --input_len_max 1024 \
  --output_len_min 100 --output_len_max 1024
```

## Python API

```python
import tinyvllm

engine = tinyvllm.Engine("/path/to/converted_weights")
params = tinyvllm.SampleParams(
    max_new_tokens=64,
    temperature=0.6,
    ignore_eos=False,
)

outputs = engine.generate(
    batch_input_ids=[[1, 2, 3], [4, 5, 6]],
    sample_params=[params, params],
    max_batch_size=4,
)
# outputs: list[list[int]] — generated token IDs per sequence
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `TVLLM_BLOCK_SIZE` | `16` | KV cache block size (tokens per block) |
| `TVLLM_GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of free GPU memory allocated to KV cache |
| `TVLLM_PREFIX_CACHING` | `0` | Set to `1` to enable prefix caching |

## Architecture

```
Python (tokenizer, CLI, chat)
  │
  ▼
Engine ─── Scheduler (FCFS continuous batching)
  │            │
  ▼            ▼
QwenModel ─── PagedKVCache ─── BlockManager (prefix caching, LRU eviction)
  │
  ▼
CUDA Kernels (RMSNorm, RoPE, SwiGLU, KV scatter, NF4 dequant, decode attention)
  │
  ▼
FlashInfer (prefill & decode attention wrappers)
```

## Project structure

```
cpp/
  include/tvllm/     # Headers (config, engine, model, attention, kv_cache, etc.)
  src/               # C++ implementations + pybind11 bindings
  src/kernels/       # CUDA kernels (rms_norm, rope, silu_mul, kv_cache, nf4_dequant, decode_v2)
python/
  convert_qwen25.py      # HF → tinyvllm weight conversion (fp16/bf16/fp32)
  convert_qwen25_nf4.py  # HF → tinyvllm NF4 quantized conversion
  run_generate.py         # Inference CLI
  chat.py                 # Interactive multi-turn chat
  benchmark.py            # Throughput benchmark
CMakeLists.txt
make.sh                   # Build convenience script
```

## Limitations

- Sampling supports temperature / greedy only (no top-k / top-p)
- Qwen2.5 / Qwen3 architecture only
- No tensor parallelism or multi-GPU support
- CUDA-only runtime; requires FlashInfer and an NVIDIA GPU
