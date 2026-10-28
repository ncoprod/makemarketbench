"""Game 3 - "Figgie-lite": a four-player card-trading table on four order books.

Rules (a simplified Figgie):

* A 40-card deck has four suits: one suit has 12 cards, two have 10 and one
  has 8, assigned uniformly at random (12 equally likely layouts). The **goal
  suit** has the same colour as the 12-card suit (spades/clubs are black,
  hearts/diamonds red), so it has 10 or 8 cards.
* Each of the 4 players antes 50 chips (pot = 200) and is dealt 10 cards.
* There are ``n_rounds`` trading rounds. In each round, the players act in a
  random order. On each suit, a player may post a bid and/or an ask for one
  card at an integer price. Quotes rest in that suit's order book until the
  end of the round and cross with price-time priority. A player can only offer
  cards it holds, and can only cancel its own orders.
* At the end, every goal-suit card pays 10 chips from the pot. The remainder
  (200 - 10 x goal-suit count) is split equally between the players holding
  the most goal-suit cards.

A player's score is final chips minus the ante. The table is zero-sum. The
reported metric is the **trading gain**: score minus the score the player
would have had if nobody traded, which removes most of the luck of the deal.

The **hand-Bayes bot** computes P(layout | own hand) exactly (multivariate
hypergeometric) and values a marginal card of each suit by Monte Carlo over
layouts and the other hands. It does *not* learn from other players' trades,
so it is a strong heuristic, not an equilibrium strategy.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ..engine import BUY, SELL, get_backend

SUITS = ("spades", "clubs", "hearts", "diamonds")
PARTNER = {0: 1, 1: 0, 2: 3, 3: 2}  # same-colour suit
N_PLAYERS, HAND, ANTE, POT, GOAL_PAY = 4, 10, 50, 200, 10


def layouts() -> list[tuple[int, int, int, int]]:
    """All 12 suit-count layouts (counts per suit in SUITS order)."""
    out = []
    for big, small in itertools.permutations(range(4), 2):
        counts = [10] * 4
        counts[big], counts[small] = 12, 8
        out.append(tuple(counts))
    return out


LAYOUTS = layouts()


def goal_suit(counts) -> int:
    return PARTNER[int(np.argmax(counts))]


def layout_posterior(hand_counts) -> np.ndarray:
    """P(layout | own 10-card hand), uniform prior over the 12 layouts."""
    w = np.array([math.prod(math.comb(n, h) for n, h in zip(c, hand_counts)) for c in LAYOUTS], float)
    return w / w.sum()


def payout(goal_counts: np.ndarray, goal_cards: int) -> np.ndarray:
    """Chips paid from the pot to each player given their goal-suit holdings."""
    pay = GOAL_PAY * goal_counts.astype(float)
    bonus = POT - GOAL_PAY * goal_cards
    winners = goal_counts == goal_counts.max()
    pay[winners] += bonus / winners.sum()
    return pay


@dataclass
class FiggieDeal:
    seed: int
    counts: tuple[int, int, int, int]
    hands: np.ndarray  # (4 players, 4 suits) card counts
    order: list[list[int]]  # acting order per round

    @property
    def goal(self) -> int:
        return goal_suit(self.counts)


def deal(seed: int, n_rounds: int) -> FiggieDeal:
    rng = np.random.default_rng([3, seed])
    counts = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    deck = np.repeat(np.arange(4), counts)
    rng.shuffle(deck)
    hands = np.zeros((N_PLAYERS, 4), dtype=np.int64)
    for p in range(N_PLAYERS):
        for c in deck[p * HAND : (p + 1) * HAND]:
            hands[p, c] += 1
    order = [list(rng.permutation(N_PLAYERS)) for _ in range(n_rounds)]
    return FiggieDeal(seed, counts, hands, [[int(x) for x in o] for o in order])


@dataclass
class SeatView:
    """What one player sees when it acts."""

    seat: int
    round: int
    n_rounds: int
    hand: list[int]  # current holdings per suit
    chips: int  # current chips (after ante, incl. trading cash)
    book: dict  # suit -> {"bid": best bid or None, "ask": best ask or None}
    tape: list[dict]  # all trades so far (public)
    rng_seed: int


@dataclass
class TableResult:
    seed: int
    seats: list[str]
    goal: int
    counts: tuple
    final_hands: list[list[int]]
    scores: list[float]
    no_trade_scores: list[float]  # scores had nobody traded (removes the luck of the deal)
    trades: list[dict]
    infos: list[list[dict]] = field(default_factory=list)


class FiggieTable:
    name = "figgie"

    def __init__(self, n_rounds: int = 4, backend: str | None = None) -> None:
        self.n_rounds = n_rounds
        self.backend = backend

    def play(self, seed: int, players: list) -> TableResult:
        d = deal(seed, self.n_rounds)
        mod = get_backend(self.backend)
        books = [mod.OrderBook() for _ in SUITS]
        hands = d.hands.copy()
        chips = np.full(N_PLAYERS, -ANTE, dtype=np.int64)
        tape: list[dict] = []
        infos: list[list[dict]] = [[] for _ in range(N_PLAYERS)]
        next_id = 1
        for p, pl in enumerate(players):
            pl.reset(seed, p)
        for r in range(self.n_rounds):
            resting: dict[int, list[tuple[int, int]]] = {p: [] for p in range(N_PLAYERS)}  # id, suit
            pending_asks = np.zeros((N_PLAYERS, 4), dtype=np.int64)
            for p in d.order[r]:
                view = SeatView(
                    p, r, self.n_rounds, hands[p].tolist(), int(chips[p]),
                    {SUITS[s]: {"bid": books[s].best_bid(), "ask": books[s].best_ask()} for s in range(4)},
                    list(tape), seed * 1000 + r * 10 + p)
                orders, info = players[p].act(view)
                infos[p].append(info)
                for s, side, price in orders:
                    if side == SELL and hands[p, s] - pending_asks[p, s] < 1:
                        continue  # cannot offer a card it does not hold
                    oid = next_id
                    next_id += 1
                    fills = books[s].submit_limit(oid, p, side, int(price), 1, False)
                    if side == SELL:
                        pending_asks[p, s] += 1
                    for f in fills:
                        taker, maker, qty, px = f[2], f[3], f[6], f[5]
                        buyer, seller = (taker, maker) if f[4] == BUY else (maker, taker)
                        hands[buyer, s] += qty
                        hands[seller, s] -= qty
                        chips[buyer] -= px * qty
                        chips[seller] += px * qty
                        pending_asks[seller, s] -= qty  # the seller's reserved card is delivered
                        tape.append({"round": r, "suit": SUITS[s], "price": int(px),
                                     "buyer": int(buyer), "seller": int(seller)})
                    if books[s].has_order(oid):
                        resting[p].append((oid, s))
            for p in range(N_PLAYERS):  # quotes live for one round; owners cancel their own
                for oid, s in resting[p]:
                    books[s].cancel(oid, p)
        g = d.goal
        pay = payout(hands[:, g], int(d.counts[g]))
        scores = (chips + pay).tolist()
        no_trade = (payout(d.hands[:, g], int(d.counts[g])) - ANTE).tolist()
        return TableResult(seed, [pl.name for pl in players], g, d.counts, hands.tolist(),
                           [float(x) for x in scores], [float(x) for x in no_trade], tape, infos)


# ---------------------------------------------------------------------------- bots
def card_values(hand: list[int], seat_seed: int, n_samples: int = 300) -> np.ndarray:
    """Expected marginal value (chips) of one more card of each suit for a holder
    of ``hand``: goal-suit payout plus the change in the expected bonus share.
    Monte Carlo over layouts (exact posterior) and over the other hands, which
    are treated as a uniform deal of the remaining cards (initial-hand view)."""
    rng = np.random.default_rng(seat_seed)
    post = layout_posterior(hand)
    h = np.array(hand)
    up = np.zeros(4)
    for _ in range(n_samples):
        counts = np.array(LAYOUTS[rng.choice(len(LAYOUTS), p=post)])
        rest = np.repeat(np.arange(4), counts - h)
        rng.shuffle(rest)
        others = np.zeros((3, 4), dtype=np.int64)
        for i in range(3):
            for c in rest[i * 10 : (i + 1) * 10]:
                others[i, c] += 1
        g = goal_suit(counts)
        base = payout(np.concatenate([[h[g]], others[:, g]]), int(counts[g]))[0]
        for s in range(4):
            if s != g:
                continue  # non-goal cards are worth nothing at settlement
            plus = payout(np.concatenate([[h[g] + 1], others[:, g]]), int(counts[g]))[0]
            up[s] += plus - base
    return up / n_samples


class HandBayesBot:
    """Quotes every suit around its Monte Carlo marginal card value."""

    def __init__(self, edge: int = 2, name: str = "hand-bayes") -> None:
        self.edge, self.name = edge, name

    def reset(self, seed: int, seat: int) -> None:
        pass

    def act(self, v: SeatView):
        vals = card_values(v.hand, v.rng_seed)
        orders = []
        for s in range(4):
            fair = vals[s]
            bid = max(int(math.floor(fair - self.edge)), 0)
            ask = int(math.ceil(fair + self.edge))
            if bid > 0:
                orders.append((s, BUY, bid))
            if v.hand[s] > 0:
                orders.append((s, SELL, max(ask, bid + 1)))
        return orders, {"values": [round(float(x), 2) for x in vals]}


class NaiveBot:
    """Ignores its hand: values every suit at the unconditional 10 x P(goal) = 2.5."""

    name = "naive"

    def reset(self, seed: int, seat: int) -> None:
        pass

    def act(self, v: SeatView):
        orders = []
        for s in range(4):
            orders.append((s, BUY, 1))
            if v.hand[s] > 0:
                orders.append((s, SELL, 5))
        return orders, {}


class NoiseBot:
    """Random bids/asks between 0 and 20 chips."""

    name = "noise"

    def reset(self, seed: int, seat: int) -> None:
        self.rng = np.random.default_rng([7, seed, seat])

    def act(self, v: SeatView):
        orders = []
        for s in range(4):
            if self.rng.random() < 0.5:
                orders.append((s, BUY, int(self.rng.integers(0, 12))))
            if v.hand[s] > 0 and self.rng.random() < 0.5:
                orders.append((s, SELL, int(self.rng.integers(4, 20))))
        return orders, {}
