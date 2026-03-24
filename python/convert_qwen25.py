import argparse
from pathlib import Path
from typing import Dict

import torch
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
    }


def _convert_state_dict(state: Dict[str, torch.Tensor], cfg) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    out["tok_embeddings.weight"] = state["model.embed_tokens.weight"]

    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}."
        out[f"layers.{i}.self_attn.q_proj.weight"] = state[prefix + "self_attn.q_proj.weight"]
        out[f"layers.{i}.self_attn.k_proj.weight"] = state[prefix + "self_attn.k_proj.weight"]
        out[f"layers.{i}.self_attn.v_proj.weight"] = state[prefix + "self_attn.v_proj.weight"]
        out[f"layers.{i}.self_attn.o_proj.weight"] = state[prefix + "self_attn.o_proj.weight"]

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            bias_key = prefix + f"self_attn.{proj}.bias"
            if bias_key in state:
                out[f"layers.{i}.self_attn.{proj}.bias"] = state[bias_key]

        # Qwen3 per-head QK-norm weights (absent in Qwen2.5)
        for norm in ["q_norm", "k_norm"]:
            norm_key = prefix + f"self_attn.{norm}.weight"
            if norm_key in state:
                out[f"layers.{i}.self_attn.{norm}.weight"] = state[norm_key]

        out[f"layers.{i}.mlp.gate_proj.weight"] = state[prefix + "mlp.gate_proj.weight"]
        out[f"layers.{i}.mlp.up_proj.weight"] = state[prefix + "mlp.up_proj.weight"]
        out[f"layers.{i}.mlp.down_proj.weight"] = state[prefix + "mlp.down_proj.weight"]

        out[f"layers.{i}.input_layernorm.weight"] = state[prefix + "input_layernorm.weight"]
        out[f"layers.{i}.post_attention_layernorm.weight"] = state[
            prefix + "post_attention_layernorm.weight"
        ]

    out["norm.weight"] = state["model.norm.weight"]
    if "lm_head.weight" in state:
        out["lm_head.weight"] = state["lm_head.weight"]

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--output", required=True, help="output directory path")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype = _dtype_from_str(args.dtype)

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    state = model.state_dict()
    converted = _convert_state_dict(state, cfg)
    for k in list(converted.keys()):
        converted[k] = converted[k].to(dtype)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_bias = "model.layers.0.self_attn.q_proj.bias" in state
    use_qk_norm = "model.layers.0.self_attn.q_norm.weight" in state
    config = _collect_config(cfg, use_bias, use_qk_norm)
    config_path = out_dir / "config.txt"
    with config_path.open("w", encoding="utf-8") as f:
        for key, val in config.items():
            f.write(f"{key}={val}\n")

    weights_path = out_dir / "state_dict.pt"
    torch.save(converted, weights_path, _use_new_zipfile_serialization=False)

    print(f"saved: {config_path}")
    print(f"saved: {weights_path}")


if __name__ == "__main__":
    main()
