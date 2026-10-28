"""Shared market mechanics for the single-dealer games.

Every game has a hidden integer value ``V`` drawn from a known generative
process and ``n_rounds`` quoting rounds. Before round ``t`` some public
information is revealed (a die, a card, a hint). The dealer (the model under
test) then posts a two-sided quote: ``bid``/``bid_size`` and ``ask``/``ask_size``.
Exactly one counterparty arrives per round:

* with probability ``alpha`` an **informed trader** who knows ``V``: buys the
  whole ask size if ``V > ask``, sells the whole bid size if ``V < bid``;
* otherwise a **noise trader** with valuation ``u = m_pub + noise_sd * z``,
  ``z ~ N(0, 1)``, where ``m_pub`` is E[V | public information]. If ``u > m_pub``
  they want to buy ``d ~ U{1..d_max}`` units at any price <= u, otherwise sell
  ``d`` units at any price >= u.

The quote rests in the order book for one round; the counterparty sends an IOC
order; unfilled quote size is cancelled. Inventory carries over and is settled
at ``V`` at the end. All randomness (V, reveals, counterparties) is a function
of the episode seed, so different players face identical counterparties
(common random numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TICKS_PER_UNIT = 10  # prices have one decimal place


@dataclass(frozen=True)
class MarketParams:
    alpha: float = 0.3  # probability the arriving counterparty is informed
    noise_sd: float = 4.0  # dispersion of noise-trader valuations around m_pub
    d_max: int = 5  # noise demand ~ U{1..d_max}
    max_size: int = 10  # maximum quote size per side


@dataclass
class Counterparty:
    informed: bool
    z: float  # standard normal draw for the noise valuation
    demand: int  # noise demand (ignored for informed traders)


@dataclass
class Belief:
    """A discrete distribution over integer values of V."""

    values: np.ndarray  # int64, sorted
    probs: np.ndarray  # float64, sums to 1

    @property
    def mean(self) -> float:
        return float(self.values @ self.probs)

    @property
    def sd(self) -> float:
        m = self.mean
        return float(math.sqrt(max(((self.values - m) ** 2) @ self.probs, 0.0)))

    def quantile(self, q: float) -> float:
        c = np.cumsum(self.probs)
        return float(self.values[min(int(np.searchsorted(c, q - 1e-12)), len(c) - 1)])

    def reweight(self, likelihood: np.ndarray) -> Belief:
        w = self.probs * likelihood
        s = w.sum()
        if s <= 0:  # observation impossible under the model (should not happen); keep prior
            return self
        return Belief(self.values, w / s)


@dataclass
class Episode:
    game: str
    seed: int
    value: int  # hidden V
    reveals: list[str]  # reveals[t]: public info shown before round t ("" if none)
    public_beliefs: list[Belief]  # posterior given public info before round t
    counterparties: list[Counterparty]
    meta: dict = field(default_factory=dict)

    @property
    def n_rounds(self) -> int:
        return len(self.reveals)


# ---------------------------------------------------------------------------
# Flow model: closed-form expectations used by the reference quoter, the
# Rao-Blackwellised scorer and the training reward.
# ---------------------------------------------------------------------------

def _norm_sf(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * np.vectorize(math.erfc)(x / math.sqrt(2.0))


def noise_buy_prob(ask: np.ndarray | float, m_pub: float, p: MarketParams) -> np.ndarray:
    """P(noise trader is a buyer whose limit u >= ask)."""
    t = (np.asarray(ask, dtype=float) - m_pub) / p.noise_sd
    return _norm_sf(np.maximum(t, 0.0))


def noise_sell_prob(bid: np.ndarray | float, m_pub: float, p: MarketParams) -> np.ndarray:
    """P(noise trader is a seller whose limit u <= bid)."""
    t = (m_pub - np.asarray(bid, dtype=float)) / p.noise_sd
    return _norm_sf(np.maximum(t, 0.0))


def expected_min_demand(q: np.ndarray | int, p: MarketParams) -> np.ndarray:
    """E[min(d, q)] for d ~ U{1..d_max}."""
    d = np.arange(1, p.d_max + 1)
    q = np.asarray(q)
    return np.minimum(d[None, :], q.reshape(-1, 1)).mean(axis=1).reshape(q.shape)


def prob_min_demand(x: int, q: int, p: MarketParams) -> float:
    """P(min(d, q) == x) for d ~ U{1..d_max}."""
    d = np.arange(1, p.d_max + 1)
    return float((np.minimum(d, q) == x).mean())


@dataclass
class Quote:
    bid: float
    bid_size: int
    ask: float
    ask_size: int

    def ticks(self) -> tuple[int, int]:
        return round(self.bid * TICKS_PER_UNIT), round(self.ask * TICKS_PER_UNIT)

    def to_dict(self) -> dict:
        return {"bid": self.bid, "bid_size": self.bid_size, "ask": self.ask, "ask_size": self.ask_size}


def round_expectations(q: Quote, value_or_belief, m_pub: float, p: MarketParams) -> dict:
    """Expected one-round PnL of quote ``q``, split into the noise edge and the
    adverse-selection cost, either given the true value (int) or averaged over a
    Belief. Prices are evaluated on the tick grid exactly as the engine does."""
    if isinstance(value_or_belief, Belief):
        vals, probs = value_or_belief.values.astype(float), value_or_belief.probs
    else:
        vals, probs = np.array([float(value_or_belief)]), np.array([1.0])
    bt, at = q.ticks()
    bid, ask = bt / TICKS_PER_UNIT, at / TICKS_PER_UNIT
    noise = adverse = 0.0
    if q.ask_size > 0:
        e_noise = float(noise_buy_prob(ask, m_pub, p)) * float(expected_min_demand(q.ask_size, p))
        noise += (1 - p.alpha) * e_noise * float(((ask - vals) * probs).sum())
        adverse += p.alpha * q.ask_size * float((np.maximum(vals - ask, 0.0) * probs).sum())
    if q.bid_size > 0:
        e_noise = float(noise_sell_prob(bid, m_pub, p)) * float(expected_min_demand(q.bid_size, p))
        noise += (1 - p.alpha) * e_noise * float(((vals - bid) * probs).sum())
        adverse += p.alpha * q.bid_size * float((np.maximum(bid - vals, 0.0) * probs).sum())
    return {"expected_pnl": noise - adverse, "noise_edge": noise, "adverse_cost": adverse}


def flow_likelihood(values: np.ndarray, q: Quote, m_pub: float, p: MarketParams,
                    traded_side: str | None, qty: int) -> np.ndarray:
    """P(observed dealer fill | V) for each candidate V.

    ``traded_side`` is "sold" (counterparty lifted the ask), "bought" (hit the
    bid) or None (no trade)."""
    v = values.astype(float)
    bt, at = q.ticks()
    bid, ask = bt / TICKS_PER_UNIT, at / TICKS_PER_UNIT
    pb = float(noise_buy_prob(ask, m_pub, p)) if q.ask_size > 0 else 0.0
    ps = float(noise_sell_prob(bid, m_pub, p)) if q.bid_size > 0 else 0.0
    inf_buys = (v > ask) & (q.ask_size > 0)
    inf_sells = (v < bid) & (q.bid_size > 0) & ~inf_buys
    if traded_side is None:
        return p.alpha * (~inf_buys & ~inf_sells) + (1 - p.alpha) * (1.0 - pb - ps)
    if traded_side == "sold":
        return (p.alpha * inf_buys * (qty == q.ask_size)
                + (1 - p.alpha) * pb * prob_min_demand(qty, q.ask_size, p))
    if traded_side == "bought":
        return (p.alpha * inf_sells * (qty == q.bid_size)
                + (1 - p.alpha) * ps * prob_min_demand(qty, q.bid_size, p))
    raise ValueError(traded_side)


def draw_counterparties(rng: np.random.Generator, n: int, p: MarketParams) -> list[Counterparty]:
    u = rng.random(n)
    z = rng.standard_normal(n)
    d = rng.integers(1, p.d_max + 1, n)
    return [Counterparty(bool(u[i] < p.alpha), float(z[i]), int(d[i])) for i in range(n)]
