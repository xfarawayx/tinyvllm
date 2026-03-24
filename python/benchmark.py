#!/usr/bin/env python3
"""Throughput benchmark for tinyvllm.

Scenario:
  - Total Requests : 256 sequences
  - Input  Length  : Randomly sampled between 100-1024 tokens
  - Output Length  : Randomly sampled between 100-1024 tokens
  - No max_batch_size limit (all requests scheduled freely)

Usage:
  CUDA_VISIBLE_DEVICES=5 python benchmark.py --model_dir <converted-model-dir>
"""

import argparse
import os
import random
import time
import sys

import torch
import tinyvllm


def main() -> None:
    parser = argparse.ArgumentParser(description="tinyvllm throughput benchmark")
    parser.add_argument("--model_dir", required=True,
                        help="Path to converted tinyvllm model directory")
    parser.add_argument("--num_requests", type=int, default=256)
    parser.add_argument("--input_len_min", type=int, default=100)
    parser.add_argument("--input_len_max", type=int, default=1024)
    parser.add_argument("--output_len_min", type=int, default=100)
    parser.add_argument("--output_len_max", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Cap concurrent running requests; 0 lets the engine decide")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ignore_eos", action="store_true",
                        help="Ignore EOS and continue until max_new_tokens")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Read vocab_size from config.txt ---
    vocab_size = None
    config_path = f"{args.model_dir}/config.txt"
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("vocab_size"):
                vocab_size = int(line.split("=", 1)[1].strip())
                break
    if vocab_size is None:
        print("[ERROR] Could not read vocab_size from config.txt", file=sys.stderr)
        sys.exit(1)

    # --- Generate random input sequences ---
    input_lens = [random.randint(args.input_len_min, args.input_len_max)
                  for _ in range(args.num_requests)]

    # Use token ids in [1, vocab_size-1] to avoid special tokens (0 is often pad).
    batch_input_ids = [
        [random.randint(1, vocab_size - 1) for _ in range(length)]
        for length in input_lens
    ]

    total_input_tokens = sum(input_lens)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

    print("=" * 65)
    print("tinyvllm Throughput Benchmark")
    print("=" * 65)
    print(f"  GPU                : {gpu_name}  (CUDA_VISIBLE_DEVICES={gpu_index})")
    print(f"  Num requests       : {args.num_requests}")
    print(f"  Input length range : [{args.input_len_min}, {args.input_len_max}]")
    print(f"  Output length range: [{args.output_len_min}, {args.output_len_max}]")
    print(f"  Temperature        : {args.temperature}")
    print(f"  Ignore EOS         : {args.ignore_eos}")
    print(f"  Max batch size     : {args.max_batch_size if args.max_batch_size > 0 else 'auto'}")
    print(f"  Seed               : {args.seed}")
    print(f"  Total input tokens : {total_input_tokens}")
    print("-" * 65)

    # --- Load model ---
    print("Loading model...", flush=True)
    load_start = time.perf_counter()
    state_dict = torch.load(f"{args.model_dir}/state_dict.pt", map_location="cpu")
    engine = tinyvllm.Engine(args.model_dir, state_dict)
    load_elapsed = time.perf_counter() - load_start
    print(f"Model loaded in {load_elapsed:.2f}s")

    # --- Warmup (small batch) ---
    warmup_ids = [batch_input_ids[0][:100]]
    _ = engine.generate(
        warmup_ids,
        [tinyvllm.SampleParams(4, temperature=args.temperature, ignore_eos=args.ignore_eos)],
    )
    torch.cuda.synchronize()

    # --- Benchmark ---
    print(f"Running benchmark ({args.num_requests} requests, "
          f"output_len=[{args.output_len_min}, {args.output_len_max}], "
          f"max_batch_size={args.max_batch_size if args.max_batch_size > 0 else 'auto'})...",
          flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    output_lens = [random.randint(args.output_len_min, args.output_len_max)
                   for _ in range(args.num_requests)]
    sample_params = [
        tinyvllm.SampleParams(
            ol,
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
        )
        for ol in output_lens
    ]
    all_outputs = engine.generate(
        batch_input_ids,
        sample_params,
        args.max_batch_size,
    )

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # --- Compute statistics ---
    actual_output_lens = [len(out) - input_lens[i]
                          for i, out in enumerate(all_outputs)]
    requested_output_tokens = sum(output_lens)
    total_output_tokens = sum(actual_output_lens)
    total_tokens = total_input_tokens + total_output_tokens

    print("=" * 65)
    print("Results")
    print("=" * 65)
    print(f"  Elapsed time          : {elapsed:.3f} s")
    print(f"  Total input  tokens   : {total_input_tokens}")
    print(f"  Requested output toks : {requested_output_tokens}")
    print(f"  Total output tokens   : {total_output_tokens}")
    print(f"  Total tokens          : {total_tokens}")
    print(f"  Requested throughput  : {requested_output_tokens / elapsed:.2f} tok/s")
    print(f"  Input  throughput     : {total_input_tokens / elapsed:.2f} tok/s")
    print(f"  Output throughput     : {total_output_tokens / elapsed:.2f} tok/s")
    print(f"  Total  throughput     : {total_tokens / elapsed:.2f} tok/s")
    print(f"  Request throughput    : {args.num_requests / elapsed:.2f} req/s")
    print("-" * 65)
    print(f"  Avg input  length     : {total_input_tokens / args.num_requests:.1f}")
    print(f"  Avg actual output len : {total_output_tokens / args.num_requests:.1f}")
    print(f"  Min actual output len : {min(actual_output_lens)}")
    print(f"  Max actual output len : {max(actual_output_lens)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
