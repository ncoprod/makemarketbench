"""Export MLX LoRALinear weights to PEFT, preserving the low-rank update.

MLX computes scale * x @ A @ B; PEFT computes (alpha/r) * x @ A.T @ B.T.
This converts the adapter only. MLX affine and bitsandbytes NF4 base weights
are different quantizations; equivalent adapter deltas do not imply identical logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def export(source: Path, config_path: Path, out: Path) -> dict:
    from safetensors.numpy import load_file, save_file

    config = json.loads(config_path.read_text())
    rank = config["lora_parameters"]["rank"]
    scale = config["lora_parameters"]["scale"]
    weights = load_file(source)
    converted = {}
    layers, modules = set(), set()
    for key, weight in weights.items():
        stem, suffix = key.rsplit(".", 1)
        if suffix not in ("lora_a", "lora_b"):
            raise ValueError(f"unsupported adapter tensor: {key}")
        layers.add(int(stem.split(".")[2]))
        modules.add(stem.rsplit(".", 1)[1])
        peft_suffix = "lora_A.weight" if suffix == "lora_a" else "lora_B.weight"
        converted[f"base_model.model.{stem}.{peft_suffix}"] = np.ascontiguousarray(weight.T)
    # Check the actual forward update for every projection, without a base model.
    rng = np.random.default_rng(0)
    max_error = 0.0
    for key, a in weights.items():
        if not key.endswith(".lora_a"):
            continue
        stem = key.removesuffix(".lora_a")
        b = weights[stem + ".lora_b"]
        x = rng.normal(size=(2, a.shape[0])).astype(np.float32)
        pa = converted[f"base_model.model.{stem}.lora_A.weight"]
        pb = converted[f"base_model.model.{stem}.lora_B.weight"]
        original = scale * (x @ a @ b)
        restored = (scale * rank / rank) * (x @ pa.T @ pb.T)
        np.testing.assert_allclose(original, restored, rtol=1e-5, atol=1e-5)
        max_error = max(max_error, float(np.max(np.abs(original - restored))))
    out.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(out / "adapter_model.safetensors"))
    peft = dict(base_model_name_or_path="Qwen/Qwen3-1.7B", peft_type="LORA",
                task_type="CAUSAL_LM", r=rank, lora_alpha=scale * rank,
                lora_dropout=0.0, bias="none", target_modules=sorted(modules),
                layers_to_transform=sorted(layers), inference_mode=False)
    (out / "adapter_config.json").write_text(json.dumps(peft, indent=2) + "\n")
    manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    tensors=len(converted), layers=sorted(layers), rank=rank, mlx_scale=scale,
                    max_forward_delta_error=max_error,
                    caveat="Adapter delta preserved; MLX affine and CUDA NF4 base quantizations differ.")
    (out / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(export(args.source, args.config, args.out), indent=2))


if __name__ == "__main__":
    main()
