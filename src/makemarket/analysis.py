"""State-level diagnostics: compare each quote with the Bayes reference *at the
same state* (same public information and the player's own fill history).

* ``mid_error``: |quote mid - Bayes posterior mean|, an inference-error proxy.
* ``spread_ratio``: quoted spread / Bayes-optimal spread at that state.
* ``regret``: E[PnL | posterior] of the Bayes quote minus that of the player's
  quote (>= 0 up to tick effects); this is exactly the training reward basis.
* ``flow_shift``: mean (mid - public mean) in rounds that follow a round where
  the player was lifted (sold), minus the same after being hit (bought). A
  dealer that learns from order flow shifts up after being lifted (> 0).
  Descriptive only: for dealers whose quotes do not track the public mean,
  persistent mispricing correlates with fills and confounds this number, so it
  is not reported in the leaderboard tables.
"""

from __future__ import annotations

import numpy as np

from .games import make_game
from .games.base import Quote, flow_likelihood, round_expectations
from .players import best_quote


def episode_diagnostics(ep: dict) -> list[dict]:
    game = make_game(ep["game"])
    p = game.params
    e = game.episode(ep["seed"])
    out = []
    flows = []  # (quote, m_pub, side, qty) of previous rounds
    for t, r in enumerate(ep["rounds"]):
        b = e.public_beliefs[t]
        m_pub = b.mean
        for q, m, side, qty in flows:
            b = b.reweight(flow_likelihood(b.values, q, m, p, side, qty))
        ref_q, _ = best_quote(b, m_pub, p)
        ref_ev = round_expectations(ref_q, b, m_pub, p)["expected_pnl"]
        row = {"t": t, "post_mean": b.mean, "post_sd": b.sd, "m_pub": m_pub, "ref_ev": ref_ev,
               "ref_spread": ref_q.ask - ref_q.bid if ref_q.bid_size and ref_q.ask_size else np.nan}
        if r["quote"] is not None:
            q = Quote(**r["quote"])
            ev = round_expectations(q, b, m_pub, p)["expected_pnl"]
            row.update(mid=(q.bid + q.ask) / 2, spread=q.ask - q.bid, ev=ev, regret=ref_ev - ev,
                       mid_error=abs((q.bid + q.ask) / 2 - b.mean))
            flows.append((q, m_pub, r["traded_side"], r["qty"]))
        else:
            row.update(mid=np.nan, spread=np.nan, ev=0.0, regret=ref_ev, mid_error=np.nan)
        prev = ep["rounds"][t - 1]["traded_side"] if t > 0 else None
        row["prev_side"] = prev
        out.append(row)
    return out


def player_diagnostics(eps: list[dict]) -> dict:
    rows = [d for ep in eps for d in episode_diagnostics(ep)]
    mid_err = np.array([d["mid_error"] for d in rows], float)
    ratio = np.array([d["spread"] / d["ref_spread"] for d in rows], float)
    regret = np.array([d["regret"] for d in rows], float)
    n_rounds = max(d["t"] for d in rows) + 1
    by_round = [float(np.mean([d["regret"] for d in rows if d["t"] == t])) for t in range(n_rounds)]
    def shift(side):
        v = [d["mid"] - d["m_pub"] for d in rows if d["prev_side"] == side and np.isfinite(d["mid"])]
        return float(np.mean(v)) if v else None

    sold, bought = shift("sold"), shift("bought")
    mid_err, ratio = mid_err[np.isfinite(mid_err)], ratio[np.isfinite(ratio)]

    from .scoring import bootstrap_ci

    # episode-level clustering for the CI: resample episodes, not rounds
    n_r = max(d["t"] for d in rows) + 1
    per_ep = regret.reshape(-1, n_r).mean(1) if len(regret) % n_r == 0 else regret
    ev = np.array([d["ev"] for d in rows], float)
    ref_ev = np.array([d["ref_ev"] for d in rows], float)
    return {
        "regret_per_round_ci": list(bootstrap_ci(per_ep)),
        "posterior_edge_pct": float(100 * ev.sum() / ref_ev.sum()),
        "flow_shift": sold - bought if sold is not None and bought is not None else None,
        "mid_error": float(mid_err.mean()) if mid_err.size else None,
        "median_spread_ratio": float(np.median(ratio)) if ratio.size else None,
        "regret_per_round": float(regret.mean()),
        "regret_by_round": by_round,
    }
