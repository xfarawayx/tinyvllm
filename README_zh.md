# tinyvllm

一个从零实现的轻量级 LLM 推理引擎，支持 Qwen2.5 / Qwen3 模型。使用 C++/CUDA 编写核心逻辑，通过 pybind11 提供 Python API。实现了 vLLM 论文中的核心技术——分页 KV 缓存、连续批处理、前缀缓存，以及自定义 CUDA 算子——作为一个学习导向的代码库。

## 功能特性

- **Qwen2.5 & Qwen3 支持** — 包括 Qwen3 的可选逐头 QK-norm
- **FlashInfer 注意力后端** — prefill 和 decode 均使用 FlashInfer，采用 plan/run 分离以减少 GPU→CPU 同步
- **融合 CUDA 算子** — RMSNorm（fp32 累加）、融合 residual-add + RMSNorm、RoPE（Q+K 原地计算）、SwiGLU 激活、decode KV scatter——每个算子将多个 libtorch 操作替换为单次 kernel 调用
- **分页 KV 缓存** — 固定大小的块池（默认 16 token/块），动态分配，通过 `cudaMemGetInfo` 自动计算 GPU 显存预算
- **前缀缓存** — 基于内容哈希的块共享，链式哈希，LRU 淘汰
- **NF4 量化** — 双重量化的 4-bit 权重，CUDA kernel 实时反量化（兼容 bitsandbytes）
- **连续批处理** — FCFS 调度器，三重准入约束：最大批大小、空闲 KV 块数、最大批处理 token 数
- **逐请求采样控制** — 支持 `temperature` 采样和 `ignore_eos`；`temperature <= 0` 回退到贪心解码
- **fp16 / bf16** — 所有算子均支持两种数据类型
- **Head dim 64 / 128** — 编译期分派两种常见的注意力头维度

## 环境要求

- Python 3.9+
- PyTorch（含 CUDA）及 `transformers`
- FlashInfer
- CUDA 工具链 + NVIDIA GPU
- CMake >= 3.18，C++17 编译器

## 构建

```bash
# 使用构建脚本（自动检测 Python、CUDA、PyTorch 路径）：
./make.sh

# 或手动构建：
mkdir -p build && cd build
cmake .. && make -j
```

设置 `CMAKE_CUDA_ARCHITECTURES` 以适配不同的 GPU（默认 `80` / A100）：

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. && make -j
# 或通过构建脚本：
CUDA_ARCH=89 ./make.sh
```

构建产物：
- `build/tinyvllm.*.so` — Python 扩展模块

## 权重转换

将 HuggingFace Qwen2.5/Qwen3 权重转换为 tinyvllm 格式：

```bash
# fp16/bf16
python python/convert_qwen25.py \
  --model /path/to/Qwen2.5-7B \
  --output /path/to/converted_weights \
  --dtype bf16

# NF4 量化（需要 bitsandbytes>=0.43）
python python/convert_qwen25_nf4.py \
  --model /path/to/Qwen2.5-7B \
  --output /path/to/converted_weights_nf4 \
  --dtype bf16
```

输出 `config.txt`（模型超参数）和 `state_dict.pt`（权重张量）到目标目录。

## 运行推理

```bash
# 单轮生成
PYTHONPATH=build \
python python/run_generate.py \
  --model_dir /path/to/converted_weights \
  --hf_model /path/to/Qwen2.5-7B \
  --prompts "你好" "今天天气怎么样" \
  --max_new_tokens 32 \
  --max_batch_size 4

# 交互式多轮对话
PYTHONPATH=build \
python python/chat.py \
  --model_dir /path/to/converted_weights \
  --hf_model /path/to/Qwen2.5-7B \
  --max_new_tokens 512 \
  --temperature 0.7
```

`--hf_model` 路径仅用于加载 tokenizer。

## 性能测试

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
# outputs: list[list[int]] — 每个序列生成的 token ID 列表

# 前向传播返回 logits（用于困惑度评估）
logits = engine.forward_logits(batch_input_ids=[[1, 2, 3]])
# logits: torch.Tensor [total_tokens, vocab_size]
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TVLLM_BLOCK_SIZE` | `16` | KV 缓存块大小（每块的 token 数） |
| `TVLLM_PREFIX_CACHING` | `0` | 设为 `1` 启用前缀缓存 |

## 架构

```
Python（tokenizer、CLI、chat）
  │
  ▼
Engine ─── Scheduler（FCFS 连续批处理）
  │            │
  ▼            ▼
QwenModel ─── PagedKVCache ─── BlockManager（前缀缓存、LRU 淘汰）
  │
  ▼
CUDA Kernels（RMSNorm、融合 add+RMSNorm、RoPE、SwiGLU、KV scatter、NF4 反量化）
  │
  ▼
FlashInfer（prefill & decode 注意力包装器）
```

## 项目结构

```
cpp/
  include/tvllm/     # 头文件（config、engine、model、attention、kv_cache 等）
  src/               # C++ 实现 + pybind11 绑定
  src/kernels/       # CUDA 算子（rms_norm、rope、silu_mul、kv_cache、nf4_dequant）
python/
  convert_qwen25.py      # HF → tinyvllm 权重转换（fp16/bf16/fp32）
  convert_qwen25_nf4.py  # HF → tinyvllm NF4 量化转换
  run_generate.py         # 推理 CLI
  chat.py                 # 交互式多轮对话
  benchmark.py            # 吞吐量测试
  eval_perplexity.py      # WikiText-2 困惑度评估
CMakeLists.txt
make.sh                   # 构建便捷脚本
```

## 已知限制

- 采样仅支持 temperature / 贪心解码（暂无 top-k / top-p）
- 仅支持 Qwen2.5 / Qwen3 架构
- 不支持张量并行 / 多 GPU
- 仅支持 CUDA 运行时；需要 FlashInfer 和 NVIDIA GPU
