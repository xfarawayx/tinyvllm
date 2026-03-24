"""
Convert HuggingFace Qwen2.5/Qwen3 model weights to tinyvllm NF4 format.

Uses bitsandbytes to quantize each linear layer weight to NF4 (4-bit NormalFloat)
with double quantization. Non-linear weights (embeddings, norms, lm_head) remain
in fp16/bf16.

Usage:
    python convert_qwen25_nf4.py --model <hf_model_dir> --output <output_dir> [--dtype bf16]

Requirements:
    pip install bitsandbytes>=0.43
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import bitsandbytes.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


def _dtype_from_str(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _collect_config(cfg, use_bias: bool, use_qk_norm: bool) -> Dict[str, object]:
    default_head_dim = cfg.hidden_size // cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", default_head_dim)
    return {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": head_dim,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_theta": getattr(cfg, "rope_theta", 10000.0),
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", 1e-5),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", False),
        "bos_token_id": getattr(cfg, "bos_token_id", -1),
        "eos_token_id": getattr(cfg, "eos_token_id", -1),
        "use_bias": use_bias,
        "use_qk_norm": use_qk_norm,
        "use_nf4": True,
    }


def _quantize_nf4(
    weight: torch.Tensor, blocksize: int = 64
) -> Dict[str, torch.Tensor]:
    """Quantize a weight tensor to NF4 with double quantization.

    Returns a dict of component tensors suitable for saving in state_dict.
    """
    num_rows, num_cols = weight.shape

    # bitsandbytes requires fp16 input on CUDA
    w = weight.to(dtype=torch.float16, device="cuda")
    quant, state = F.quantize_4bit(
        w, quant_type="nf4", blocksize=blocksize, compress_statistics=True
    )

    assert state.state2 is not None, "double quantization must be enabled"
    s2 = state.state2

    packed_weights = quant.cpu().to(torch.uint8).flatten()
    absmax_q = state.absmax.cpu().to(torch.uint8)
    absmax2 = s2.absmax.cpu().to(torch.float16)
    code2 = s2.code.cpu().to(torch.float16)
    offset_val = state.offset.cpu().item() if state.offset is not None else 0.0

    return {
        "packed_weights": packed_weights,
        "absmax_q": absmax_q,
        "absmax2": absmax2,
        "code2": code2,
        "offset": torch.tensor([offset_val], dtype=torch.float32),
        "blocksize": torch.tensor([blocksize], dtype=torch.int32),
        "s2_blocksize": torch.tensor([s2.blocksize], dtype=torch.int32),
        "num_rows": torch.tensor([num_rows], dtype=torch.int64),
        "num_cols": torch.tensor([num_cols], dtype=torch.int64),
    }


# Linear projection keys in the Qwen model that should be NF4-quantized
_LINEAR_PROJ_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def _convert_state_dict_nf4(
    state: Dict[str, torch.Tensor], cfg, blocksize: int
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    # Embeddings (not quantized)
    out["tok_embeddings.weight"] = state["model.embed_tokens.weight"]

    for i in range(cfg.num_hidden_layers):
        hf_prefix = f"model.layers.{i}."
        tv_prefix = f"layers.{i}."

        for suffix in _LINEAR_PROJ_SUFFIXES:
            hf_key = hf_prefix + suffix
            # Remove ".weight" to get the projection prefix
            proj_name = suffix.replace(".weight", "")
            tv_proj_prefix = tv_prefix + proj_name

            w = state[hf_key]
            print(f"  quantizing {hf_key} [{w.shape}] ...", end=" ", flush=True)
            nf4_components = _quantize_nf4(w, blocksize)
            for comp_name, comp_tensor in nf4_components.items():
                out[f"{tv_proj_prefix}.{comp_name}"] = comp_tensor
            print("done")

        # Biases (not quantized, stored as-is)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            bias_key = hf_prefix + f"self_attn.{proj}.bias"
            if bias_key in state:
                out[f"{tv_prefix}self_attn.{proj}.bias"] = state[bias_key]

        # Qwen3 per-head QK-norm weights
        for norm in ["q_norm", "k_norm"]:
            norm_key = hf_prefix + f"self_attn.{norm}.weight"
            if norm_key in state:
                out[f"{tv_prefix}self_attn.{norm}.weight"] = state[norm_key]

        # LayerNorm weights (not quantized)
        out[f"{tv_prefix}input_layernorm.weight"] = state[
            hf_prefix + "input_layernorm.weight"
        ]
        out[f"{tv_prefix}post_attention_layernorm.weight"] = state[
            hf_prefix + "post_attention_layernorm.weight"
        ]

    # Final norm (not quantized)
    out["norm.weight"] = state["model.norm.weight"]

    # LM head (not quantized)
    if "lm_head.weight" in state:
        out["lm_head.weight"] = state["lm_head.weight"]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Qwen model to tinyvllm NF4 format"
    )
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--output", required=True, help="output directory path")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="dtype for non-quantized weights (embeddings, norms)")
    parser.add_argument("--blocksize", type=int, default=64,
                        choices=[64, 128, 256, 512, 1024, 2048, 4096],
                        help="NF4 quantization block size (must be power of 2)")
    args = parser.parse_args()

    dtype = _dtype_from_str(args.dtype)

    print(f"Loading model from {args.model} ...")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,  # load as fp16 for quantization
        trust_remote_code=True,
    )

    state = model.state_dict()
    del model  # free memory

    print(f"Quantizing with NF4 (blocksize={args.blocksize}) ...")
    converted = _convert_state_dict_nf4(state, cfg, args.blocksize)
    del state

    # Convert non-NF4 weights to the requested dtype while keeping the
    # NF4 helper tensors in their required storage format.
    for k in list(converted.keys()):
        t = converted[k]
        # Skip NF4 component tensors and scalar metadata:
        # - packed_weights / absmax_q stay uint8
        # - absmax2 / code2 must remain fp16 for the dequant CUDA kernel
        # - metadata tensors keep their original integer / float32 dtypes
        if (
            t.dtype in (torch.uint8, torch.int32, torch.int64, torch.float32)
            or k.endswith(".absmax2")
            or k.endswith(".code2")
        ):
            continue
        converted[k] = t.to(dtype)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_bias = "model.layers.0.self_attn.q_proj.bias" in state if hasattr(main, '_state_ref') else False
    # Re-check bias from converted keys
    use_bias = any(k.endswith(".q_proj.bias") for k in converted)
    use_qk_norm = any(k.endswith(".q_norm.weight") for k in converted)
    config = _collect_config(cfg, use_bias, use_qk_norm)

    config_path = out_dir / "config.txt"
    with config_path.open("w", encoding="utf-8") as f:
        for key, val in config.items():
            f.write(f"{key}={val}\n")

    weights_path = out_dir / "state_dict.pt"
    torch.save(converted, weights_path, _use_new_zipfile_serialization=False)

    print(f"saved: {config_path}")
    print(f"saved: {weights_path}")

    # Print size comparison
    total_nf4_bytes = sum(
        t.numel() * t.element_size()
        for k, t in converted.items()
        if "packed_weights" in k
    )
    num_linear = sum(1 for k in converted if "packed_weights" in k)
    print(f"\nNF4 quantized {num_linear} linear layers")
    print(f"Packed weight size: {total_nf4_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
