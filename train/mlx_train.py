"""MLX LoRA with explicit non-thinking targets, validation stopping and receipts.

Uses mlx-lm's trainer directly so a callback can save the lowest-validation-loss
weights. The callback stops after three unsuccessful validation checks; test
episodes never influence training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from makemarket.provenance import file_sha256, receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--data", default="train/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accumulation", type=int, default=2)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    np.random.seed(args.seed)
    record = receipt(vars(args), args.revision)
    record["data_sha256"] = {n: file_sha256(Path(args.data) / f"{n}.jsonl") for n in ("train", "valid")}
    record["source_sha256"] = file_sha256(__file__)
    record["status"] = "initializing"
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import lora
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.tuner.trainer import TrainingCallback
    from mlx_lm.utils import load

    config = dict(lora.CONFIG_DEFAULTS)
    config.update(model=args.model, train=True, data=args.data, adapter_path=args.out,
                  resume_adapter_file=args.resume, iters=args.iters, batch_size=args.batch_size,
                  grad_accumulation_steps=args.accumulation, learning_rate=args.lr,
                  num_layers=args.layers, seed=args.seed, grad_checkpoint=True,
                  max_seq_length=1536, mask_prompt=True, steps_per_eval=100,
                  steps_per_report=10, save_every=100, val_batches=80)
    model, tokenizer = load(args.model)
    tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, enable_thinking=False)
    train_set, valid_set, _ = load_dataset(SimpleNamespace(**config), tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best.safetensors"

    class StopTraining(Exception):
        pass

    class Validation(TrainingCallback):
        best, stale, best_step = float("inf"), 0, 0
        history: list[dict] = []

        def on_val_loss_report(self, info):
            self.history.append(info)
            if info["val_loss"] < self.best - 0.001:
                self.best, self.stale, self.best_step = info["val_loss"], 0, info["iteration"]
                mx.save_safetensors(str(best_path), dict(tree_flatten(model.trainable_parameters())))
            else:
                self.stale += 1
            record.update(validation=self.history, best_validation_loss=self.best,
                          best_step=self.best_step, peak_memory_bytes=mx.get_peak_memory(),
                          wall_seconds=time.monotonic() - started, status="running")
            Path(args.result).write_text(json.dumps(record, indent=2) + "\n")
            if self.stale >= 3:
                raise StopTraining

    validation = Validation()
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    try:
        lora.train_model(SimpleNamespace(**config), model, train_set, valid_set, validation)
        record["status"] = "completed"
    except StopTraining:
        record["status"] = "validation_plateau"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(wall_seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory())
        if best_path.exists():
            record["best_adapter_sha256"] = file_sha256(best_path)
        Path(args.result).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
