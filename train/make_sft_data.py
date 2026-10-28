"""Build supervised fine-tuning data from the Bayes reference dealer.

Each example is one quoting decision: the exact prompt the evaluation harness
sends (system + user) and, as the target, a one-line rationale with the
reference posterior followed by the reference quote as JSON. Training seeds
start at 10_000 and never overlap the evaluation seeds (0-49).

    python train/make_sft_data.py --episodes 800 --out train/data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from makemarket.games import make_game
from makemarket.market import Observation, play_episode
from makemarket.players import BayesQuoter
from makemarket.prompts import messages_for

TRAIN_SEED_OFFSET = 10_000


class RecordingBayes(BayesQuoter):
    """Bayes reference that also records (messages, target) for every round."""

    def __init__(self) -> None:
        super().__init__(learn_from_flow=True)
        self.examples: list[dict] = []

    def quote(self, obs: Observation):
        b = self.belief(obs)
        q, info = super().quote(obs)
        target = (f"Public E[V] = {obs.m_pub:.2f}; after my fills E[V] = {b.mean:.2f} "
                  f"(sd {b.sd:.2f}).\n" + json.dumps(q.to_dict()))
        self.examples.append({"messages": messages_for(obs) + [{"role": "assistant", "content": target}],
                              "meta": {"game": obs.game.name, "t": obs.t, "ref_ev": info["ev"]}})
        return q, info


def build(games: list[str], episodes: int, offset: int) -> list[dict]:
    out = []
    for g in games:
        game = make_game(g)
        for i in range(episodes):
            p = RecordingBayes()
            play_episode(game, offset + i, p, backend="py")
            out.extend(p.examples)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["dice", "cards"])
    ap.add_argument("--episodes", type=int, default=800, help="episodes per game")
    ap.add_argument("--valid-episodes", type=int, default=40)
    ap.add_argument("--out", default="train/data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = build(args.games, args.episodes, TRAIN_SEED_OFFSET)
    valid = build(args.games, args.valid_episodes, TRAIN_SEED_OFFSET + 50_000)
    random.Random(0).shuffle(train)
    for name, rows in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps({"messages": r["messages"]}) + "\n")
        print(f"{name}: {len(rows)} examples")


if __name__ == "__main__":
    main()
