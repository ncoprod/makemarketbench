"""Reference and baseline dealers."""

from __future__ import annotations

import numpy as np

from .games.base import (
    TICKS_PER_UNIT,
    Belief,
    MarketParams,
    Quote,
    expected_min_demand,
    flow_likelihood,
    noise_buy_prob,
    noise_sell_prob,
)
from .market import Observation


def best_quote(belief: Belief, m_pub: float, p: MarketParams) -> tuple[Quote, dict]:
    """One-round Bayes-optimal quote: maximise E[PnL | belief] separately for the
    ask (price, size) and the bid (price, size) over the tick grid. Risk-neutral
    and myopic; sides decouple because exactly one counterparty arrives."""
    vals = belief.values.astype(float)
    mu = belief.mean
    lo = int(np.floor(vals.min() - 1)) * TICKS_PER_UNIT
    hi = int(np.ceil(vals.max() + 1)) * TICKS_PER_UNIT
    grid = np.arange(lo, hi + 1) / TICKS_PER_UNIT
    sizes = np.arange(0, p.max_size + 1)
    emin = expected_min_demand(sizes, p)  # (S,)
    # ask side
    up = (np.maximum(vals[None, :] - grid[:, None], 0.0) * belief.probs).sum(1)  # E[(V-a)+]
    ev_ask = ((1 - p.alpha) * (noise_buy_prob(grid, m_pub, p) * (grid - mu))[:, None] * emin[None, :]
              - p.alpha * up[:, None] * sizes[None, :])
    # bid side
    dn = (np.maximum(grid[:, None] - vals[None, :], 0.0) * belief.probs).sum(1)  # E[(b-V)+]
    ev_bid = ((1 - p.alpha) * (noise_sell_prob(grid, m_pub, p) * (mu - grid))[:, None] * emin[None, :]
              - p.alpha * dn[:, None] * sizes[None, :])
    ia, qa = np.unravel_index(np.argmax(ev_ask), ev_ask.shape)
    ib, qb = np.unravel_index(np.argmax(ev_bid), ev_bid.shape)
    ask, bid = float(grid[ia]), float(grid[ib])
    if ev_ask[ia, qa] <= 1e-12:
        qa, ask = 0, round(mu + 3 * belief.sd + 1, 1)
    if ev_bid[ib, qb] <= 1e-12:
        qb, bid = 0, round(mu - 3 * belief.sd - 1, 1)
    if bid >= ask:  # only possible when a side is empty
        ask = bid + 1.0 / TICKS_PER_UNIT
    q = Quote(round(bid, 1), int(qb), round(ask, 1), int(qa))
    return q, {"ev": float(max(ev_ask[ia, qa], 0) + max(ev_bid[ib, qb], 0)), "mu": mu}


class BayesQuoter:
    """Reference dealer: exact posterior over V from public information and
    (optionally) from its own fills, then the one-round optimal quote."""

    def __init__(self, learn_from_flow: bool = True) -> None:
        self.learn_from_flow = learn_from_flow
        self.name = "bayes" if learn_from_flow else "bayes-public"

    def reset(self, episode_seed: int) -> None:
        self._mpubs: list[float] = []

    def belief(self, obs: Observation) -> Belief:
        b = obs.public_belief
        if self.learn_from_flow:
            p = obs.game.params
            for rec, m in zip(obs.history, self._mpubs):
                if rec.quote is not None:
                    b = b.reweight(flow_likelihood(b.values, rec.quote, m, p, rec.traded_side, rec.qty))
        return b

    def quote(self, obs: Observation):
        b = self.belief(obs)
        self._mpubs.append(obs.m_pub)
        q, info = best_quote(b, obs.m_pub, obs.game.params)
        info["sd"] = b.sd
        return q, info


class FixedSpreadQuoter:
    """Naive baseline: prior mean +/- one prior s.d., constant size; ignores all info."""

    name = "fixed-prior"

    def __init__(self, size: int = 3, width_sd: float = 1.0) -> None:
        self.size, self.width_sd = size, width_sd

    def reset(self, episode_seed: int) -> None:
        self._prior: Belief | None = None

    def quote(self, obs: Observation):
        if self._prior is None:
            self._prior = obs.public_belief  # round 0 belief is the prior
        m, s = self._prior.mean, self._prior.sd
        return Quote(round(m - self.width_sd * s, 1), self.size,
                     round(m + self.width_sd * s, 1), self.size), {}


class PublicMeanQuoter:
    """Heuristic baseline: public posterior mean +/- one posterior s.d., constant
    size. Uses the exact posterior but does not optimise the spread or size and
    ignores order flow."""

    name = "mean-sd"

    def __init__(self, size: int = 3, width_sd: float = 1.0) -> None:
        self.size, self.width_sd = size, width_sd

    def reset(self, episode_seed: int) -> None:
        pass

    def quote(self, obs: Observation):
        b = obs.public_belief
        w = max(self.width_sd * b.sd, 0.5)
        return Quote(round(b.mean - w, 1), self.size, round(b.mean + w, 1), self.size), {}


class RandomQuoter:
    """Random baseline: prices uniform within two prior s.d. of the prior mean,
    sizes uniform in 0..max_size."""

    name = "random"

    def reset(self, episode_seed: int) -> None:
        self._rng = np.random.default_rng([99, episode_seed])
        self._prior = None

    def quote(self, obs: Observation):
        if self._prior is None:
            self._prior = obs.public_belief
        m, s = self._prior.mean, self._prior.sd
        r = self._rng
        bid = round(float(r.uniform(m - 2 * s, m)), 1)
        ask = round(float(r.uniform(m, m + 2 * s)), 1)
        if ask <= bid:
            ask = bid + 0.1
        ms = obs.game.params.max_size
        return Quote(bid, int(r.integers(0, ms + 1)), ask, int(r.integers(0, ms + 1))), {}


BASELINES = {
    "bayes": lambda: BayesQuoter(True),
    "bayes-public": lambda: BayesQuoter(False),
    "mean-sd": PublicMeanQuoter,
    "fixed-prior": FixedSpreadQuoter,
    "random": RandomQuoter,
}
