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
from .provenance import receipt

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
    raw = Path(args.results_dir) / "raw" / args.game
    for name in args.player or []:
        eps = run_player(game, BASELINES[name], seeds, raw / f"{slug(name)}.jsonl", workers=1,
                         backend=args.backend,
                         provenance=receipt({"player": name, "seeds": seeds, "backend": args.backend}))
        print(f"{name}: {len(eps)} episodes")
    for model in args.llm or []:
        from .llm import ChatClient, LLMConfig, LLMQuoter, Throttle

        extra = json.loads(args.extra_body) if args.extra_body else {}
        cfg = LLMConfig(model=model, base_url=args.base_url, api_key_env=args.api_key_env,
                        temperature=args.temperature, max_tokens=args.max_tokens,
                        extra_body=extra, label=args.label)
        client = ChatClient(cfg, cache_dir=args.cache_dir, throttle=Throttle(args.workers))
        eps = run_player(game, lambda c=client: LLMQuoter(c, hint=args.hint), seeds, raw / f"{slug(cfg.name)}.jsonl",
                         workers=args.workers, backend=args.backend)
        print(f"{cfg.name}: {len(eps)} episodes")


def cmd_score(args) -> None:
    from .scoring import markdown_table, summarize

    seeds = set(parse_seeds(args.seeds))
    results = Path(args.results_dir)
    raw = results / "raw" / args.game
    eps = {}
    for f in sorted(raw.glob("*.jsonl")):
        rows = [e for e in load_results(f) if e["seed"] in seeds]
        if rows:
            eps[rows[0]["player"]] = rows
    if "bayes" not in eps:
        sys.exit("run `mmb eval --player bayes` first")
    from .analysis import player_diagnostics

    rows = [r for r in summarize(eps) if r["episodes"] >= args.min_episodes]
    for r in rows:
        r.update(player_diagnostics(eps[r["player"]]))
    out = results / f"leaderboard_{args.game}.json"
    out.write_text(json.dumps({"game": args.game, "seeds": sorted(seeds), "rows": rows,
                              "provenance": receipt({"game": args.game, "seeds": sorted(seeds),
                                                     "min_episodes": args.min_episodes,
                                                     "results_dir": args.results_dir})}, indent=2) + "\n")
    md = markdown_table(rows)
    (results / f"leaderboard_{args.game}.md").write_text(md + "\n")
    print(md)


def cmd_figgie(args) -> None:
    import numpy as np

    from .figgie_eval import SEAT0, LLMSeat, run_table
    from .scoring import bootstrap_ci

    seeds = parse_seeds(args.seeds)
    raw = RESULTS / "raw" / "figgie"
    for name in args.player or []:
        run_table(SEAT0[name], seeds, raw / f"{slug(name)}.jsonl")
    for model in args.llm or []:
        from .llm import ChatClient, LLMConfig, Throttle

        extra = json.loads(args.extra_body) if args.extra_body else {}
        cfg = LLMConfig(model=model, base_url=args.base_url, api_key_env=args.api_key_env,
                        temperature=args.temperature, max_tokens=args.max_tokens, extra_body=extra,
                        label=args.label)
        client = ChatClient(cfg, cache_dir=args.cache_dir, throttle=Throttle(1))
        run_table(lambda c=client: LLMSeat(c), seeds, raw / f"{slug(cfg.name)}.jsonl")
    rows = []
    for f in sorted(raw.glob("*.jsonl")):
        eps = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        eps = [e for e in eps if e["seed"] in set(seeds)]
        if not eps:
            continue
        g = np.array([e["trading_gain"] for e in eps])
        lo, hi = bootstrap_ci(g)
        fe = sum(e["format_errors"] for e in eps) / max(sum(e["rounds"] for e in eps), 1)
        rows.append({"player": eps[0]["player"], "episodes": len(eps), "trading_gain": float(g.mean()),
                     "trading_gain_ci": [lo, hi], "score": float(np.mean([e["score"] for e in eps])),
                     "seat0_trades": float(np.mean([e["seat0_trades"] for e in eps])),
                     "format_error_rate": fe})
    rows.sort(key=lambda r: -r["trading_gain"])
    (RESULTS / "leaderboard_figgie.json").write_text(json.dumps({"game": "figgie", "rows": rows}, indent=2) + "\n")
    for r in rows:
        print(f"{r['player']:30s} n={r['episodes']:3d} gain {r['trading_gain']:+6.2f} "
              f"[{r['trading_gain_ci'][0]:+.2f}, {r['trading_gain_ci'][1]:+.2f}] trades {r['seat0_trades']:.1f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mmb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="play episodes and store raw results")
    e.add_argument("--game", required=True)
    e.add_argument("--results-dir", default=str(RESULTS))
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
    e.add_argument("--hint", action="store_true",
                   help="ablation: show the public posterior mean/sd in the prompt (use with --label)")
    e.add_argument("--cache-dir", default="cache/llm")
    e.add_argument("--backend", default=None, help="engine backend: cpp, py or auto")
    e.set_defaults(fn=cmd_eval)
    s = sub.add_parser("score", help="aggregate raw results into a leaderboard")
    s.add_argument("--game", required=True)
    s.add_argument("--results-dir", default=str(RESULTS))
    s.add_argument("--seeds", default="0-49")
    s.add_argument("--min-episodes", type=int, default=1)
    s.set_defaults(fn=cmd_score)
    fg = sub.add_parser("figgie", help="Figgie-lite table: seat 0 vs three hand-Bayes bots")
    fg.add_argument("--seeds", default="0-199")
    fg.add_argument("--player", nargs="*", choices=["hand-bayes", "naive", "noise"])
    fg.add_argument("--llm", nargs="*")
    fg.add_argument("--label", default=None)
    fg.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    fg.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    fg.add_argument("--temperature", type=float, default=0.3)
    fg.add_argument("--max-tokens", type=int, default=4096)
    fg.add_argument("--extra-body", default=None)
    fg.add_argument("--cache-dir", default="cache/llm")
    fg.set_defaults(fn=cmd_figgie)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
