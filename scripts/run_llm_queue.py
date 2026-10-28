"""Run the API leaderboard sequentially (at most ``--workers`` requests in flight).

Each model is probed once with a short request; unreachable models are skipped
and logged. Results are resumable: rerunning only plays missing seeds.

    source ~/.zshrc  # provides NVIDIA_API_KEY
    python scripts/run_llm_queue.py --games dice cards --seeds 0-49
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from makemarket.cli import parse_seeds, slug
from makemarket.evaluate import run_player
from makemarket.games import make_game
from makemarket.llm import ChatClient, LLMConfig, LLMQuoter, Throttle

# Each model gets a list of request variants, tried in order on a real quoting
# prompt; the first variant that returns a valid quote within the 4096-token
# budget is used for the whole run (recorded in results/llm_protocol.json).
# Reasoning models that run out of budget with thinking enabled fall back to
# their non-thinking mode, labelled "(no-think)".
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
MODELS = [
    ("openai/gpt-oss-20b", [({"reasoning_effort": "low"}, "")]),
    ("nvidia/nemotron-3-ultra-550b-a55b", [({}, ""), (NO_THINK, " (no-think)")]),
    ("minimaxai/minimax-m3.1-flash", [({}, ""), (NO_THINK, " (no-think)")]),
    ("nvidia/nemotron-3.5-bolt-e4b", [({}, "")]),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", [({}, ""), (NO_THINK, " (no-think)")]),
    ("deepseek-ai/deepseek-v4.1-flash", [({}, ""), (NO_THINK, " (no-think)")]),
    ("z-ai/glm-5.3-flash", [({}, ""), (NO_THINK, " (no-think)")]),
]
PROTOCOL = Path("results/llm_protocol.json")


def probe(model: str, extra: dict) -> tuple[bool, str]:
    """One real quoting request (dice, seed 0, round 3). Returns (usable, note)."""
    from makemarket.llm import ChatClient, LLMConfig, parse_quote
    from makemarket.market import Observation
    from makemarket.prompts import messages_for

    g = make_game("dice")
    ep = g.episode(0)
    obs = Observation(g, 2, 6, ep.reveals[:3], [], 0, 0.0, ep.public_beliefs[2], ep.public_beliefs[2].mean)
    c = ChatClient(LLMConfig(model=model, extra_body=extra), cache_dir="cache/probe", max_retries=2)
    try:
        out = c.chat(messages_for(obs))
    except Exception as e:
        return False, f"error: {str(e)[:120]}"
    note = (f"finish={out['finish_reason']} tokens={out['completion_tokens']} "
            f"latency={out.get('latency', 0):.0f}s")
    try:
        parse_quote(out["content"], g.params.max_size)
    except ValueError as e:
        return False, f"invalid ({e}); {note}"
    return True, note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["dice", "cards"])
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    models = [m for m in MODELS if args.models is None or m[0] in args.models]
    throttle = Throttle(args.workers)
    protocol = json.loads(PROTOCOL.read_text()) if PROTOCOL.exists() else {}
    for game_name in args.games:
        game = make_game(game_name)
        for model, variants in models:
            entry = protocol.get(model)
            if entry is None:
                entry = {"probes": []}
                for extra, suffix in variants:
                    ok, note = probe(model, extra)
                    entry["probes"].append({"extra_body": extra, "usable": ok, "note": note})
                    print(f"[queue] probe {model} {extra}: {ok} {note}", flush=True)
                    if ok:
                        entry.update(extra_body=extra, label=model + suffix)
                        break
                protocol[model] = entry
                PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n")
            if "label" not in entry:
                print(f"[queue] {model}: no usable variant, skipping", flush=True)
                continue
            t0 = time.time()
            cfg = LLMConfig(model=model, extra_body=entry["extra_body"], label=entry["label"])
            client = ChatClient(cfg, throttle=throttle)
            eps = run_player(game, lambda c=client: LLMQuoter(c), seeds,
                             f"results/raw/{game_name}/{slug(cfg.name)}.jsonl", workers=args.workers,
                             log=lambda s: print(s, flush=True))
            print(f"[queue] {game_name} {cfg.name}: {len(eps)} episodes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
