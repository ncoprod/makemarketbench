"""Game 2 - "Cards with hints": make a market on the sum of five hidden cards.

Five cards are dealt without replacement from a standard 52-card deck
(A=1, 2..10, J=11, Q=12, K=13; suits do not matter). Before rounds 2..6 the
public learns, in order: a card, a hint, a card, a hint, a card. Hints are true
statements about the whole five-card hand (including already revealed cards),
so a good dealer must combine sampling without replacement with conditioning on
non-trivial events. The exact posterior is computed by enumerating rank
multisets of the unseen cards, weighted by the number of deck combinations.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .base import Belief, Episode, MarketParams, draw_counterparties

RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}


def rank_name(r: int) -> str:
    return RANK_NAMES.get(r, str(r))


# Each hint: (key, function(hand ranks) -> value, renderer(value) -> text)
def _hint_defs():
    return {
        "faces": (lambda h: sum(r >= 11 for r in h),
                  lambda v: f"Exactly {v} of the five cards {'is a' if v == 1 else 'are'} face card{'s' if v != 1 else ''} (J, Q or K)."),
        "max": (lambda h: max(h), lambda v: f"The highest card in the hand is a {rank_name(v)}."),
        "min": (lambda h: min(h), lambda v: f"The lowest card in the hand is a {rank_name(v)}."),
        "parity": (lambda h: sum(h) % 2, lambda v: f"The sum V is {'odd' if v else 'even'}."),
        "low": (lambda h: sum(r <= 5 for r in h),
                lambda v: f"Exactly {v} of the five cards {'is' if v == 1 else 'are'} 5 or lower (A counts as 1)."),
        "pair": (lambda h: int(max(Counter(h).values()) >= 2),
                 lambda v: "The hand contains at least two cards of the same rank." if v
                 else "All five cards have different ranks."),
    }


HINTS = _hint_defs()


def posterior(known: list[int], hints: list[tuple[str, int]], n_cards: int = 5) -> Belief:
    """Exact posterior of the hand sum given revealed ranks and hint outcomes."""
    avail = Counter({r: 4 for r in range(1, 14)})
    avail.subtract(known)
    k = n_cards - len(known)
    acc: dict[int, float] = {}
    for combo in itertools.combinations_with_replacement(range(1, 14), k):
        cnt = Counter(combo)
        w = 1.0
        for r, m in cnt.items():
            if m > avail[r]:
                w = 0.0
                break
            w *= math.comb(avail[r], m)
        if w == 0.0:
            continue
        hand = list(known) + list(combo)
        if all(HINTS[key][0](hand) == val for key, val in hints):
            s = sum(hand)
            acc[s] = acc.get(s, 0.0) + w
    vals = np.array(sorted(acc), dtype=np.int64)
    probs = np.array([acc[v] for v in sorted(acc)], dtype=float)
    return Belief(vals, probs / probs.sum())


@dataclass
class CardsGame:
    name: str = "cards"
    n_rounds: int = 6
    schedule: tuple[str, ...] = ("card", "hint", "card", "hint", "card")
    params: MarketParams = field(default_factory=lambda: MarketParams(alpha=0.3, noise_sd=6.0))

    def rules(self) -> str:
        return (
            "V is the sum of 5 cards dealt face down, without replacement, from a standard "
            "52-card deck. Card values: A=1, 2-10 at face value, J=11, Q=12, K=13 (suits do "
            "not matter). There are 6 rounds. Before round 1 you have no information. Before "
            "rounds 2-6 the following is revealed to everyone, in order: a card, a hint, a "
            "card, a hint, a card. Hints are true statements about the whole five-card hand "
            "(including cards already revealed). Two cards are never revealed."
        )

    def episode(self, seed: int) -> Episode:
        rng = np.random.default_rng([2, seed])
        deck = np.repeat(np.arange(1, 14), 4)
        hand = [int(x) for x in rng.choice(deck, 5, replace=False)]
        value = sum(hand)
        hint_keys = list(rng.choice(sorted(HINTS), 2, replace=False))
        known: list[int] = []
        hints: list[tuple[str, int]] = []
        reveals, beliefs = [""], [posterior([], [])]
        ci = hi = 0
        for kind in self.schedule:
            if kind == "card":
                c = hand[ci]
                ci += 1
                known.append(c)
                reveals.append(f"Card #{ci} is revealed: {rank_name(c)} (value {c}).")
            else:
                key = str(hint_keys[hi])
                hi += 1
                val = HINTS[key][0](hand)
                hints.append((key, val))
                reveals.append(f"Hint: {HINTS[key][1](val)}")
            beliefs.append(posterior(known, hints))
        cps = draw_counterparties(rng, self.n_rounds, self.params)
        return Episode(self.name, seed, value, reveals, beliefs, cps,
                       {"hand": hand, "hints": [[k, v] for k, v in hints]})
