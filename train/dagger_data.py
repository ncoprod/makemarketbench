"""DAgger / expert-iteration data: let the current model play training episodes
and label every state it visits with the Bayes reference quote.

The reference label is computed from the model's own history (its quotes and
the fills they produced), so the data covers the states the model actually
reaches - the standard fix for compounding errors in behaviour cloning.

    python train/dagger_data.py --base-url http://127.0.0.1:8081/v1 --episodes 150 \
        --out train/data_dagger --merge train/data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from makemarket.games import make_game
from makemarket.llm import ChatClient, LLMConfig, LLMQuoter, Throttle
from makemarket.market import play_episode
from makemarket.players import BayesQuoter
from makemarket.prompts import messages_for

DAGGER_SEED_OFFSET = 30_000


class LabelledPlayer:
    def __init__(self, actor: LLMQuoter) -> None:
        self.actor, self.ref = actor, BayesQuoter(True)
        self.name = actor.name
        self.examples: list[dict] = []

    def reset(self, seed: int) -> None:
        self.actor.reset(seed)
        self.ref.reset(seed)

    def quote(self, obs):
        b = self.ref.belief(obs)
        ref_q, _ = self.ref.quote(obs)
        target = (f"Public E[V] = {obs.m_pub:.2f}; after my fills E[V] = {b.mean:.2f} "
                  f"(sd {b.sd:.2f}).\n" + json.dumps(ref_q.to_dict()))
        self.examples.append({"messages": messages_for(obs) + [{"role": "assistant", "content": target}]})
        return self.actor.quote(obs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="default_model")
    ap.add_argument("--games", nargs="+", default=["dice", "cards"])
    ap.add_argument("--episodes", type=int, default=150, help="per game")
    ap.add_argument("--out", default="train/data_dagger")
    ap.add_argument("--merge", default="train/data/train.jsonl")
    ap.add_argument("--merge-n", type=int, default=4000, help="original examples to keep")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    client = ChatClient(LLMConfig(model=args.model, base_url=args.base_url, api_key_env="NONE",
                                  temperature=0.0, max_tokens=256, label="dagger-actor"),
                        cache_dir="cache/dagger", throttle=Throttle(args.workers))
    new: list[dict] = []
    for g in args.games:
        game = make_game(g)

        def run(i, game=game):
            p = LabelledPlayer(LLMQuoter(client))
            play_episode(game, DAGGER_SEED_OFFSET + i, p, backend="py")
            return p.examples

        with ThreadPoolExecutor(args.workers) as pool:
            for ex in pool.map(run, range(args.episodes)):
                new.extend(ex)
        print(f"{g}: {len(new)} labelled states so far", flush=True)
    old = [json.loads(line) for line in Path(args.merge).read_text().splitlines()][: args.merge_n]
    rows = new + old
    random.Random(1).shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    valid = Path(args.merge).with_name("valid.jsonl")
    (out / "valid.jsonl").write_text(valid.read_text())
    print(f"wrote {len(rows)} examples ({len(new)} on-policy) to {out}")


if __name__ == "__main__":
    main()
