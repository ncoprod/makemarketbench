"""Evaluate a frozen MLX policy directly, with resumable paired JSON outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from makemarket.analysis import player_diagnostics
from makemarket.evaluate import load_results, run_player
from makemarket.games import make_game
from makemarket.llm import parse_quote
from makemarket.players import BASELINES
from makemarket.prompts import OUTPUT_SPEC, messages_for
from makemarket.provenance import file_sha256, receipt
from makemarket.scoring import markdown_table, summarize


class MLXQuoter:
    def __init__(self, model, tokenizer, name):
        self.model, self.tokenizer, self.name = model, tokenizer, name
        self.cache = {}

    def reset(self, seed):
        import mlx.core as mx

        mx.random.seed(seed)

    def quote(self, obs):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages, attempts = messages_for(obs), []
        key = json.dumps(messages, sort_keys=True)
        if key in self.cache:
            quote, info = self.cache[key]
            return quote, {**info, "cached": True}
        for attempt in range(2):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            started = time.monotonic()
            text = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=256,
                            sampler=make_sampler(temp=0), verbose=False)
            attempts.append({"content": text, "latency": time.monotonic() - started})
            try:
                quote = parse_quote(text, obs.game.params.max_size)
                info = {"attempts": attempts, "format_error": attempt > 0,
                        "failed": False, "cached": False}
                self.cache[key] = quote, info
                return quote, info
            except ValueError as exc:
                attempts[-1]["error"] = str(exc)
                messages += [{"role": "assistant", "content": text[-2000:]},
                             {"role": "user", "content": f"Your answer was invalid: {exc}. {OUTPUT_SPEC}"}]
        info = {"attempts": attempts, "format_error": True, "failed": True, "cached": False}
        self.cache[key] = None, info
        return None, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, default=80000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--games", nargs="+", default=["dice", "cards"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    record = receipt(vars(args), args.revision)
    record["source_sha256"] = file_sha256(__file__)
    if args.adapter:
        record["adapter_sha256"] = file_sha256(Path(args.adapter) / "adapters.safetensors")
    for name in args.games:
        path = Path(args.out) / "raw" / name / f"{args.label}.jsonl"
        for episode in load_results(path):
            old = episode.get("provenance", {})
            if any(old.get(k) != record.get(k) for k in ("adapter_sha256", "model_revision")):
                raise ValueError(f"{path} contains a different model/adapter; use a new output directory")
    started = time.monotonic()

    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(256 * 1024**2)
    model, tokenizer = load(args.model, adapter_path=args.adapter, revision=args.revision)
    player = MLXQuoter(model, tokenizer, args.label)
    seeds = list(range(args.seed, args.seed + args.episodes))
    for name in args.games:
        raw = Path(args.out) / "raw" / name
        for baseline in ("bayes", "bayes-public", "mean-sd", "fixed-prior", "random"):
            run_player(make_game(name), BASELINES[baseline], seeds, raw / f"{baseline}.jsonl",
                       provenance=receipt({"player": baseline, "seeds": seeds}))
        run_player(make_game(name), lambda: player,
                   seeds, raw / f"{args.label}.jsonl", provenance=record)
        players = {eps[0]["player"]: eps for f in sorted(raw.glob("*.jsonl"))
                   if (eps := load_results(f))}
        rows = summarize(players)
        for row in rows:
            row.update(player_diagnostics(players[row["player"]]))
        result = dict(game=name, seeds=seeds, rows=rows, provenance=record,
                      wall_seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory())
        dest = Path(args.out) / f"leaderboard_{name}"
        dest.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        dest.with_suffix(".md").write_text(markdown_table(rows) + "\n")
        print(markdown_table(rows), flush=True)


if __name__ == "__main__":
    main()
