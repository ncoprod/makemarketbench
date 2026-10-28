"""Episode runner: plays one game episode through the matching engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .engine import BUY, SELL, Exchange
from .games.base import TICKS_PER_UNIT, Belief, Episode, Quote, round_expectations

DEALER, COUNTERPARTY = 0, 1


@dataclass
class RoundRecord:
    t: int
    reveal: str
    quote: Quote | None  # None = no valid quote this round
    traded_side: str | None = None  # "sold" / "bought" from the dealer's point of view
    qty: int = 0
    price: float = 0.0
    position: int = 0  # after the round
    cash: float = 0.0  # after the round


@dataclass
class Observation:
    """Everything the dealer may see before quoting round ``t``."""

    game: object
    t: int
    n_rounds: int
    reveals: list[str]  # public information revealed so far (reveals[0..t])
    history: list[RoundRecord]  # previous rounds
    position: int
    cash: float
    public_belief: Belief  # E.g. for the reference quoter; LLM prompts do not include it
    m_pub: float


class Player(Protocol):
    name: str

    def reset(self, episode_seed: int) -> None: ...

    def quote(self, obs: Observation) -> tuple[Quote | None, dict]: ...


@dataclass
class EpisodeResult:
    game: str
    seed: int
    player: str
    value: int
    rounds: list[RoundRecord]
    round_info: list[dict] = field(default_factory=list)  # per-round player metadata
    expectations: list[dict] = field(default_factory=list)  # per-round E[PnL | V, quote]

    @property
    def realized_pnl(self) -> float:
        last = self.rounds[-1]
        return last.cash + last.position * self.value


def validate_quote(raw: dict, max_size: int) -> Quote:
    """Validate a quote dict; raises ValueError with a human-readable reason."""
    try:
        bid, ask = float(raw["bid"]), float(raw["ask"])
        bs, as_ = raw["bid_size"], raw["ask_size"]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"missing or non-numeric field: {e}") from None
    for name, s in (("bid_size", bs), ("ask_size", as_)):
        if isinstance(s, bool) or not float(s).is_integer():
            raise ValueError(f"{name} must be an integer")
    bs, as_ = int(bs), int(as_)
    if not (math.isfinite(bid) and math.isfinite(ask)):
        raise ValueError("prices must be finite")
    if not (0 <= bs <= max_size and 0 <= as_ <= max_size):
        raise ValueError(f"sizes must be integers between 0 and {max_size}")
    bid, ask = round(bid, 1), round(ask, 1)
    if bid < 0 or ask > 10_000:
        raise ValueError("prices out of range")
    if not bid < ask:
        raise ValueError("bid must be strictly below ask")
    return Quote(bid, bs, ask, as_)


def play_episode(game, seed: int, player: Player, backend: str | None = None) -> EpisodeResult:
    ep: Episode = game.episode(seed)
    p = game.params
    ex = Exchange(backend)
    rounds: list[RoundRecord] = []
    infos, exps = [], []
    player.reset(seed)
    v_ticks = ep.value * TICKS_PER_UNIT
    for t in range(ep.n_rounds):
        belief = ep.public_beliefs[t]
        obs = Observation(game, t, ep.n_rounds, ep.reveals[: t + 1], list(rounds),
                          ex.ledger.position(DEALER), ex.ledger.cash(DEALER) / TICKS_PER_UNIT,
                          belief, belief.mean)
        q, info = player.quote(obs)
        infos.append(info)
        rec = RoundRecord(t, ep.reveals[t], q)
        if q is not None:
            bt, at = q.ticks()
            if q.bid_size > 0:
                ex.limit(DEALER, BUY, bt, q.bid_size)
            if q.ask_size > 0:
                ex.limit(DEALER, SELL, at, q.ask_size)
            cp = ep.counterparties[t]
            fills = []
            if cp.informed:
                if q.ask_size > 0 and v_ticks > at:
                    fills = ex.limit(COUNTERPARTY, BUY, v_ticks, q.ask_size, ioc=True)
                elif q.bid_size > 0 and v_ticks < bt:
                    fills = ex.limit(COUNTERPARTY, SELL, v_ticks, q.bid_size, ioc=True)
            else:
                u = obs.m_pub + p.noise_sd * cp.z
                if cp.z > 0:
                    fills = ex.limit(COUNTERPARTY, BUY, math.floor(u * TICKS_PER_UNIT + 1e-9),
                                     cp.demand, ioc=True)
                else:
                    fills = ex.limit(COUNTERPARTY, SELL, math.ceil(u * TICKS_PER_UNIT - 1e-9),
                                     cp.demand, ioc=True)
            ex.book.cancel_owner(DEALER)
            if fills:
                rec.traded_side = "sold" if fills[0][4] == BUY else "bought"
                rec.qty = sum(f[6] for f in fills)
                rec.price = fills[0][5] / TICKS_PER_UNIT
            exps.append(round_expectations(q, ep.value, obs.m_pub, p))
        else:
            exps.append({"expected_pnl": 0.0, "noise_edge": 0.0, "adverse_cost": 0.0})
        rec.position = ex.ledger.position(DEALER)
        rec.cash = ex.ledger.cash(DEALER) / TICKS_PER_UNIT
        rounds.append(rec)
    return EpisodeResult(ep.game, seed, player.name, ep.value, rounds, infos, exps)
