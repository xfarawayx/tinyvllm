#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoTokenizer

import tinyvllm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hf_model", required=True)
    parser.add_argument("--prompts", nargs="+", required=True, help="batch prompts")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore_eos", action="store_true",
                        help="Ignore EOS and continue until max_new_tokens")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Max concurrent sequences for continuous batching (0 = all at once)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)

    batch_input_ids = []
    for prompt in args.prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        batch_input_ids.append(ids)

    state_dict = torch.load(f"{args.model_dir}/state_dict.pt", map_location="cpu", weights_only=True)
    engine = tinyvllm.Engine(args.model_dir, state_dict)

    sample_params = [tinyvllm.SampleParams(
                        args.max_new_tokens,
                        temperature=args.temperature,
                        ignore_eos=args.ignore_eos)
                     for _ in range(len(batch_input_ids))]
    all_output_ids = engine.generate(batch_input_ids, sample_params, args.max_batch_size)
    for i, ids in enumerate(all_output_ids):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"[{i}] {text}")


if __name__ == "__main__":
    main()
