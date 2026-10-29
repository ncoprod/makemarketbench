"""Collect resumable local on-policy states, label them with the Bayes dealer."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from dagger_data import LabelledPlayer
from mlx_evaluate import MLXQuoter

from makemarket.games import make_game
from makemarket.market import play_episode
from makemarket.provenance import file_sha256, receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=150, help="per game; 1,800 new states total")
    args = ap.parse_args()
    record = receipt(vars(args), args.revision)
    record["source_sha256"] = file_sha256(__file__)
    record["adapter_sha256"] = file_sha256(Path(args.adapter) / "adapters.safetensors")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "on_policy.jsonl"
    records = [json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r["game"], r["seed"]) for r in records}
    started = time.monotonic()

    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(args.model, adapter_path=args.adapter, revision=args.revision)
    for game_name in ("dice", "cards"):
        for seed in range(30000, 30000 + args.episodes):
            if (game_name, seed) in done:
                continue
            player = LabelledPlayer(MLXQuoter(model, tokenizer, "dagger-actor"))
            play_episode(make_game(game_name), seed, player, backend="py")
            row = {"game": game_name, "seed": seed, "examples": player.examples, "provenance": record}
            with raw.open("a") as f:
                f.write(json.dumps(row) + "\n")
            records.append(row)
            print(f"{game_name} seed={seed} states={len(player.examples)}", flush=True)
    new = [e for r in records for e in r["examples"]]
    old = [json.loads(s) for s in Path("train/data/train.jsonl").read_text().splitlines()][:4000]
    rows = new + old
    random.Random(4).shuffle(rows)
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "valid.jsonl").write_text(Path("train/data/valid.jsonl").read_text())
    record.update(on_policy_states=len(new), original_states=len(old), seed_range=[30000, 30000 + args.episodes - 1],
                  wall_seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory(),
                  data_sha256=file_sha256(out / "train.jsonl"))
    (out / "collection.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
