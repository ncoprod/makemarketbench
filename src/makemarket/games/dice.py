"""Game 1 - "Dice sum": make a market on the sum of eight hidden dice.

One die is revealed before each of rounds 2..6; three dice are never revealed.
The public posterior is the revealed sum plus the convolution of the hidden dice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import Belief, Episode, MarketParams, draw_counterparties


def _sum_of_dice(k: int, faces: int = 6) -> Belief:
    dist = np.array([1.0])
    die = np.ones(faces) / faces
    for _ in range(k):
        dist = np.convolve(dist, die)
    values = np.arange(k, k * faces + 1) if k > 0 else np.array([0])
    return Belief(values.astype(np.int64), dist)


@dataclass
class DiceGame:
    name: str = "dice"
    n_dice: int = 8
    n_rounds: int = 6
    params: MarketParams = field(default_factory=lambda: MarketParams(alpha=0.3, noise_sd=4.0))

    def rules(self) -> str:
        return (
            f"V is the sum of {self.n_dice} fair six-sided dice that were rolled in secret. "
            f"There are {self.n_rounds} rounds. Before round 1 you have no information about the "
            f"dice. Before each of rounds 2 to {self.n_rounds}, one more die is revealed to "
            f"everyone, so {self.n_dice - (self.n_rounds - 1)} dice remain hidden at the end."
        )

    def episode(self, seed: int) -> Episode:
        rng = np.random.default_rng([1, seed])
        dice = rng.integers(1, 7, self.n_dice)
        value = int(dice.sum())
        reveals, beliefs = [], []
        for t in range(self.n_rounds):
            k = t  # dice revealed before round t (0-based)
            reveals.append("" if t == 0 else f"Die #{t} shows {int(dice[t - 1])}.")
            hidden = _sum_of_dice(self.n_dice - k)
            beliefs.append(Belief(hidden.values + int(dice[:k].sum()), hidden.probs))
        cps = draw_counterparties(rng, self.n_rounds, self.params)
        return Episode(self.name, seed, value, reveals, beliefs, cps, {"dice": dice.tolist()})
