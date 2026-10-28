"""Command-line interface.

Examples::

    mmb eval --game dice --player bayes mean-sd fixed-prior random
    mmb eval --game dice --llm openai/gpt-oss-20b --workers 2
    mmb eval --game dice --llm qwen3:4b --base-url http://localhost:11434/v1 --label qwen3-4b-ollama
    mmb score --game dice
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .evaluate import load_results, run_player
from .games import make_game
from .players import BASELINES

RESULTS = Path("results")


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def cmd_eval(args) -> None:
    game = make_game(args.game)
    seeds = parse_seeds(args.seeds)
    raw = RESULTS / "raw" / args.game
    for name in args.player or []:
        eps = run_player(game, BASELINES[name], seeds, raw / f"{slug(name)}.jsonl", workers=1,
                         backend=args.backend)
        print(f"{name}: {len(eps)} episodes")
    for model in args.llm or []:
        from .llm import ChatClient, LLMConfig, LLMQuoter, Throttle

        extra = json.loads(args.extra_body) if args.extra_body else {}
        cfg = LLMConfig(model=model, base_url=args.base_url, api_key_env=args.api_key_env,
                        temperature=args.temperature, max_tokens=args.max_tokens,
                        extra_body=extra, label=args.label)
        client = ChatClient(cfg, cache_dir=args.cache_dir, throttle=Throttle(args.workers))
        eps = run_player(game, lambda c=client: LLMQuoter(c), seeds, raw / f"{slug(cfg.name)}.jsonl",
                         workers=args.workers, backend=args.backend)
        print(f"{cfg.name}: {len(eps)} episodes")


def cmd_score(args) -> None:
    from .scoring import markdown_table, summarize

    seeds = set(parse_seeds(args.seeds))
    raw = RESULTS / "raw" / args.game
    eps = {}
    for f in sorted(raw.glob("*.jsonl")):
        rows = [e for e in load_results(f) if e["seed"] in seeds]
        if rows:
            eps[rows[0]["player"]] = rows
    if "bayes" not in eps:
        sys.exit("run `mmb eval --player bayes` first")
    rows = [r for r in summarize(eps) if r["episodes"] >= args.min_episodes]
    out = RESULTS / f"leaderboard_{args.game}.json"
    out.write_text(json.dumps({"game": args.game, "seeds": sorted(seeds), "rows": rows}, indent=2) + "\n")
    md = markdown_table(rows)
    (RESULTS / f"leaderboard_{args.game}.md").write_text(md + "\n")
    print(md)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mmb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="play episodes and store raw results")
    e.add_argument("--game", required=True)
    e.add_argument("--seeds", default="0-49")
    e.add_argument("--player", nargs="*", choices=sorted(BASELINES))
    e.add_argument("--llm", nargs="*", help="model ids served by an OpenAI-compatible endpoint")
    e.add_argument("--label", default=None, help="results name for a single --llm model")
    e.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    e.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    e.add_argument("--temperature", type=float, default=0.3)
    e.add_argument("--max-tokens", type=int, default=4096)
    e.add_argument("--extra-body", default=None, help="JSON merged into the request body")
    e.add_argument("--workers", type=int, default=2, help="max concurrent requests")
    e.add_argument("--cache-dir", default="cache/llm")
    e.add_argument("--backend", default=None, help="engine backend: cpp, py or auto")
    e.set_defaults(fn=cmd_eval)
    s = sub.add_parser("score", help="aggregate raw results into a leaderboard")
    s.add_argument("--game", required=True)
    s.add_argument("--seeds", default="0-49")
    s.add_argument("--min-episodes", type=int, default=1)
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
