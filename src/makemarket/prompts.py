"""Prompt rendering shared by the LLM harness and the training scripts."""

from __future__ import annotations

from .market import Observation

OUTPUT_SPEC = (
    'Respond with a single JSON object, and nothing after it, of the form '
    '{"bid": <number>, "bid_size": <integer>, "ask": <number>, "ask_size": <integer>}. '
    "You may reason briefly before the JSON."
)


def system_prompt(game) -> str:
    p = game.params
    return f"""You are a market maker in a trading game.

GAME. {game.rules()}

QUOTING. In every round you post a two-sided quote: a bid price and size (you buy up to bid_size units at the bid) and an ask price and size (you sell up to ask_size units at the ask). Prices may have one decimal place; sizes are integers from 0 to {p.max_size}; size 0 means no quote on that side; the bid must be strictly below the ask. Your quote is live for this round only; unfilled size is cancelled.

COUNTERPARTIES. In every round exactly one counterparty arrives:
- With probability {p.alpha:.0%} it is an INFORMED trader who knows V exactly. If V > ask they buy your whole ask size; if V < bid they sell you your whole bid size; otherwise they do nothing.
- With probability {1 - p.alpha:.0%} it is a NOISE trader. Let M be the expected value of V given only the public information revealed so far. The noise trader's private valuation is u = M + e with e ~ Normal(0, {p.noise_sd:g}^2). If u > M they want to buy D units at any price up to u; otherwise they want to sell D units at any price down to u. D is uniform on 1..{p.d_max}. They trade min(D, your size) at your price if it is within their limit.
You see your own fills, but not who traded with you.

SCORING. Trades execute at your quoted prices. Your position carries over between rounds and is settled at V after the last round. Your score is your total profit: cash + position * V. Be risk-neutral: maximise expected profit.

OUTPUT. {OUTPUT_SPEC}"""


def _fmt_quote(q) -> str:
    return f"bid {q.bid:.1f} x {q.bid_size}, ask {q.ask:.1f} x {q.ask_size}"


def user_prompt(obs: Observation) -> str:
    lines = [f"Round {obs.t + 1} of {obs.n_rounds}.", "", "Public information so far:"]
    info = [(i, r) for i, r in enumerate(obs.reveals) if r]
    if info:
        lines += [f"- Before round {i + 1}: {r}" for i, r in info]
    else:
        lines.append("- None yet.")
    lines += ["", "Your previous rounds:"]
    if obs.history:
        for rec in obs.history:
            if rec.quote is None:
                lines.append(f"- Round {rec.t + 1}: no valid quote.")
                continue
            if rec.traded_side == "sold":
                res = f"a counterparty bought {rec.qty} from you at {rec.price:.1f}"
            elif rec.traded_side == "bought":
                res = f"a counterparty sold {rec.qty} to you at {rec.price:.1f}"
            else:
                res = "no trade"
            lines.append(f"- Round {rec.t + 1}: {_fmt_quote(rec.quote)} -> {res}.")
    else:
        lines.append("- None yet.")
    pos = obs.position
    side = " (long)" if pos > 0 else " (short)" if pos < 0 else ""
    lines += ["", f"Current position: {pos:+d}{side}. Cash: {obs.cash:+.1f}.", "",
              f"Post your quote for round {obs.t + 1}."]
    return "\n".join(lines)


def messages_for(obs: Observation) -> list[dict]:
    return [{"role": "system", "content": system_prompt(obs.game)},
            {"role": "user", "content": user_prompt(obs)}]
