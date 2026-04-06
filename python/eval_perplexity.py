#!/usr/bin/env python3
"""WikiText-2 perplexity evaluation for tinyvllm and HuggingFace models.

Usage:
    # tinyvllm engine
    python python/eval_perplexity.py \
        --model_dir /path/to/converted_model \
        --hf_model Qwen/Qwen2.5-1.5B

    # HuggingFace reference (no tinyvllm)
    python python/eval_perplexity.py \
        --hf_model Qwen/Qwen2.5-1.5B --hf_only
"""
import argparse
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


def eval_tinyvllm(model_dir: str, tokenizer, input_ids: torch.Tensor,
                  max_length: int, stride: int) -> float:
    import tinyvllm

    state_dict = torch.load(
        f"{model_dir}/state_dict.pt", map_location="cpu", weights_only=True)
    engine = tinyvllm.Engine(model_dir, state_dict)

    max_length = min(max_length, engine.max_position_embeddings())
    stride = stride or max_length
    total_length = input_ids.size(0)

    print(f"Context length: {max_length}, stride: {stride}")

    nlls: list[float] = []
    num_scored = 0
    prev_end = 0

    for begin in range(0, total_length, stride):
        end = min(begin + max_length, total_length)
        chunk_ids = input_ids[begin:end].tolist()

        if len(chunk_ids) < 2:
            break

        engine.reset()
        logits = engine.forward_logits([chunk_ids])  # [seq_len, vocab]

        targets = input_ids[begin:end].to(logits.device)

        target_start = max(1, prev_end - begin)
        target_end = end - begin

        if target_start >= target_end:
            break

        shift_logits = logits[target_start - 1 : target_end - 1]
        shift_labels = targets[target_start : target_end]

        nll = F.cross_entropy(
            shift_logits.float(), shift_labels, reduction="sum"
        ).item()

        count = target_end - target_start
        nlls.append(nll)
        num_scored += count
        prev_end = end

        ppl_so_far = math.exp(sum(nlls) / num_scored)
        print(f"  [{begin}:{end}] scored {count} tokens, running ppl = {ppl_so_far:.2f}")

        if end >= total_length:
            break

    avg_nll = sum(nlls) / num_scored
    perplexity = math.exp(avg_nll)
    print(f"\nTokens scored: {num_scored}")
    print(f"Perplexity: {perplexity:.4f}")
    return perplexity


def eval_hf(hf_model: str, input_ids: torch.Tensor,
            max_length: int, stride: int) -> float:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    device = model.device

    stride = stride or max_length
    total_length = input_ids.size(0)

    print(f"Context length: {max_length}, stride: {stride}")

    nlls: list[float] = []
    num_scored = 0
    prev_end = 0

    for begin in range(0, total_length, stride):
        end = min(begin + max_length, total_length)
        chunk = input_ids[begin:end].unsqueeze(0).to(device)

        if chunk.size(1) < 2:
            break

        with torch.no_grad():
            logits = model(chunk).logits[0]  # [seq_len, vocab]

        targets = input_ids[begin:end].to(device)

        target_start = max(1, prev_end - begin)
        target_end = end - begin

        if target_start >= target_end:
            break

        shift_logits = logits[target_start - 1 : target_end - 1]
        shift_labels = targets[target_start : target_end]

        nll = F.cross_entropy(
            shift_logits.float(), shift_labels, reduction="sum"
        ).item()

        count = target_end - target_start
        nlls.append(nll)
        num_scored += count
        prev_end = end

        ppl_so_far = math.exp(sum(nlls) / num_scored)
        print(f"  [{begin}:{end}] scored {count} tokens, running ppl = {ppl_so_far:.2f}")

        if end >= total_length:
            break

    avg_nll = sum(nlls) / num_scored
    perplexity = math.exp(avg_nll)
    print(f"\nTokens scored: {num_scored}")
    print(f"Perplexity: {perplexity:.4f}")
    return perplexity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None,
                        help="tinyvllm converted model dir")
    parser.add_argument("--hf_model", required=True, help="HF model name/path")
    parser.add_argument("--hf_only", action="store_true",
                        help="Run HuggingFace reference only (no tinyvllm)")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=0)
    args = parser.parse_args()

    if not args.hf_only and args.model_dir is None:
        parser.error("--model_dir is required unless --hf_only is set")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    print(f"WikiText-2 test: {input_ids.size(0)} tokens")

    if args.hf_only:
        eval_hf(args.hf_model, input_ids, args.max_length, args.stride)
    else:
        eval_tinyvllm(args.model_dir, tokenizer, input_ids,
                      args.max_length, args.stride)


if __name__ == "__main__":
    main()
