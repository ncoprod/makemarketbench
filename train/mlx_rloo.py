"""Small local REINFORCE run with a leave-one-out baseline (RLOO-style).

Four categorical samples per training state, exact quote-EV rewards, one update
per group. No PPO clipping, reference KL or critic. This is a local fallback,
not a claim to have executed TRL's GRPO trainer. Test seeds are never sampled.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from grpo_unsloth_kaggle import build_prompt_set, make_reward_fns

from makemarket.provenance import file_sha256, receipt


def leave_one_out(rewards):
    rewards = np.asarray(rewards, dtype=float)
    return rewards - (rewards.sum() - rewards) / (len(rewards) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    record = receipt(vars(args), args.revision)
    record["source_sha256"] = file_sha256(__file__)
    record["initial_adapter_sha256"] = file_sha256(Path(args.adapter) / "adapters.safetensors")
    result, out = Path(args.result), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(result.read_text()) if result.exists() else {}
    history = previous.get("history", [])
    started = time.monotonic()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import load_adapters
    from mlx_memory import enable_gated_delta_training

    record["recurrence_training"] = enable_gated_delta_training()

    model, tokenizer = load(args.model, revision=args.revision)
    model.freeze()
    load_adapters(model, args.adapter)
    optimizer = optim.Adam(learning_rate=args.lr)
    if history:
        checkpoint = out / f"step_{len(history):04d}"
        model.load_weights(str(checkpoint / "adapters.safetensors"), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(checkpoint / "optimizer.safetensors")).items()))
    grad_checkpoint(model.layers[0])
    rows = build_prompt_set(100, ["dice"], seed_offset=20000)
    fmt, edge = make_reward_fns()

    def loss(policy, prompt, completion, advantage):
        tokens = mx.array([prompt + completion])
        logits = policy(tokens[:, :-1])[:, len(prompt) - 1:].astype(mx.float32)
        targets = tokens[:, len(prompt):]
        logp = mx.take_along_axis(mx.log_softmax(logits, axis=-1), targets[..., None], axis=-1)
        return -advantage * logp.sum()

    value_grad = nn.value_and_grad(model, loss)
    try:
        for step in range(len(history), args.steps):
            mx.random.seed(args.seed + step)
            row = rows[step % len(rows)]
            prompt = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                  add_generation_prompt=True, enable_thinking=False)
            model.eval()
            completions, texts = [], []
            for _ in range(4):
                tokens, segments = [], []
                for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=128,
                                                sampler=make_sampler(temp=1.0)):
                    tokens.append(response.token)
                    segments.append(response.text)
                completions.append(tokens)
                texts.append("".join(segments))
            kwargs = {k: [row[k]] * 4 for k in ("game", "values", "probs", "m_pub", "ref_quote")}
            rewards = np.array(fmt(None, texts, **kwargs)) + np.array(edge(None, texts, **kwargs))
            advantages = leave_one_out(rewards)
            model.train()
            gradient = None
            for completion, advantage in zip(completions, advantages):
                _, grad = value_grad(model, prompt, completion, float(advantage))
                gradient = grad if gradient is None else tree_map(lambda a, b: a + b, gradient, grad)
                mx.eval(gradient)
            optimizer.update(model, tree_map(lambda g: g / 4, gradient))
            mx.eval(model.parameters(), optimizer.state)
            checkpoint = out / f"step_{step + 1:04d}"
            checkpoint.mkdir(exist_ok=True)
            mx.save_safetensors(str(checkpoint / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
            mx.save_safetensors(str(checkpoint / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
            history.append({"step": step + 1, "game": "dice", "rewards": rewards.tolist(),
                            "advantages": advantages.tolist(), "completions": texts})
            record.update(status="running", history=history, wall_seconds=time.monotonic() - started,
                          peak_memory_bytes=mx.get_peak_memory())
            result.write_text(json.dumps(record, indent=2) + "\n")
            print(f"step {step + 1}: rewards {rewards.tolist()}", flush=True)
        shutil.copyfile(checkpoint / "adapters.safetensors", out / "adapters.safetensors")
        shutil.copyfile(Path(args.adapter) / "adapter_config.json", out / "adapter_config.json")
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(wall_seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory())
        result.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
