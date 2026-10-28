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

from makemarket.cli import parse_seeds, slug
from makemarket.evaluate import run_player
from makemarket.games import make_game
from makemarket.llm import ChatClient, LLMConfig, LLMQuoter, Throttle

# (model id, extra request body). Reasoning effort is set to "low" where the
# endpoint supports it, to keep latency and rate-limit usage manageable.
MODELS = [
    ("openai/gpt-oss-20b", {"reasoning_effort": "low"}),
    ("nvidia/nemotron-3-ultra-550b-a55b", {}),
    ("minimaxai/minimax-m3.1-flash", {}),
    ("nvidia/nemotron-3.5-bolt-e4b", {}),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", {}),
    ("deepseek-ai/deepseek-v4.1-flash", {}),
    ("z-ai/glm-5.3-flash", {}),
]


def probe(model: str, extra: dict) -> str | None:
    import os

    from openai import OpenAI

    c = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"],
               timeout=120, max_retries=1)
    try:
        c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say OK."}],
                                  max_tokens=512, extra_body=extra or None)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:120]}"


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
    status = {}
    for game_name in args.games:
        game = make_game(game_name)
        for model, extra in models:
            if status.get(model, "").startswith("skip"):
                continue
            err = probe(model, extra)
            if err:
                status[model] = f"skip ({err})"
                print(f"[queue] {model}: unreachable, skipping: {err}", flush=True)
                continue
            t0 = time.time()
            client = ChatClient(LLMConfig(model=model, extra_body=extra), throttle=throttle)
            eps = run_player(game, lambda c=client: LLMQuoter(c), seeds,
                             f"results/raw/{game_name}/{slug(model)}.jsonl", workers=args.workers,
                             log=lambda s: print(s, flush=True))
            status[model] = f"ok {game_name}: {len(eps)} episodes"
            print(f"[queue] {game_name} {model}: {len(eps)} episodes in {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
