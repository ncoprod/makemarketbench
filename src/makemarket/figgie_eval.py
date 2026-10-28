"""Evaluation of one seat at the Figgie-lite table (seat 0) against three
hand-Bayes bots, plus an LLM seat that speaks JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .engine import BUY, SELL
from .games.figgie import (
    ANTE,
    GOAL_PAY,
    N_PLAYERS,
    POT,
    SUITS,
    FiggieTable,
    HandBayesBot,
    NaiveBot,
    NoiseBot,
    SeatView,
)

FIGGIE_SYSTEM = f"""You are playing Figgie-lite, a four-player card trading game. You are player 0.

DECK. 40 cards in four suits: one suit has 12 cards, two suits have 10 and one has 8 (all 12 layouts equally likely). The GOAL SUIT is the suit of the same colour as the 12-card suit (spades and clubs are black, hearts and diamonds are red); it has 10 or 8 cards. Nobody is told the layout. Each player is dealt 10 cards.

CHIPS. Each player paid {ANTE} chips into a pot of {POT}. At the end, every goal-suit card pays {GOAL_PAY} chips to its holder, and the rest of the pot ({POT} minus {GOAL_PAY} per goal-suit card) is split equally among the players holding the most goal-suit cards. Cards of other suits are worth nothing. Your score is your final chips minus the ante.

TRADING. There are several rounds. In each round every player acts once, in random order. There is one order book per suit; every order is for exactly one card at an integer price. Your orders rest until the end of the round and trade with price-time priority (the earlier/better price trades first, at the resting price). You can only offer cards you hold. You may post at most one bid and one ask per suit per round.

OUTPUT. Respond with one JSON object and nothing after it, of the form {{"orders": [{{"suit": "hearts", "side": "buy", "price": 6}}, ...]}}. Use an empty list to pass. You may reason briefly before the JSON."""


def figgie_user_prompt(v: SeatView) -> str:
    hand = ", ".join(f"{SUITS[s]}: {v.hand[s]}" for s in range(4))
    book = "; ".join(f"{s}: bid {b['bid'] if b['bid'] is not None else '-'} / ask {b['ask'] if b['ask'] is not None else '-'}"
                     for s, b in v.book.items())
    tape = v.tape[-30:]
    lines = [f"Round {v.round + 1} of {v.n_rounds}.", f"Your cards: {hand}.",
             f"Your chips change so far: {v.chips + ANTE:+d} (before settlement).",
             f"Current best quotes resting this round: {book}.", "Trades so far (latest last):"]
    lines += [f"- round {t['round'] + 1}: {t['suit']} at {t['price']} (player {t['buyer']} bought from player {t['seller']})"
              for t in tape] or ["- none"]
    lines.append("Post your orders.")
    return "\n".join(lines)


def parse_orders(text: str) -> list[tuple[int, int, int]]:
    from .llm import _THINK_RE

    text = _THINK_RE.sub("", text or "")
    start = text.rfind('{"orders"')
    if start < 0:
        start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    out, seen = [], set()
    for o in obj.get("orders", []):
        suit = SUITS.index(str(o["suit"]).lower())
        side = {"buy": BUY, "bid": BUY, "sell": SELL, "ask": SELL}[str(o["side"]).lower()]
        price = int(o["price"])
        if price < 0 or price > POT or (suit, side) in seen:
            raise ValueError("invalid order")
        seen.add((suit, side))
        out.append((suit, side, price))
    return out


class LLMSeat:
    def __init__(self, client) -> None:
        self.client, self.name = client, client.cfg.name

    def reset(self, seed: int, seat: int) -> None:
        pass

    def act(self, v: SeatView):
        msgs = [{"role": "system", "content": FIGGIE_SYSTEM}, {"role": "user", "content": figgie_user_prompt(v)}]
        attempts = []
        for attempt in range(2):
            out = self.client.chat(msgs)
            attempts.append({"content": out["content"]})
            try:
                return parse_orders(out["content"]), {"attempts": attempts, "format_error": attempt > 0}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                attempts[-1]["error"] = str(e)
                msgs = msgs + [{"role": "assistant", "content": out["content"][-2000:]},
                               {"role": "user", "content": f"Invalid answer: {e}. Respond with the JSON object only."}]
        return [], {"attempts": attempts, "format_error": True, "failed": True}


SEAT0 = {"hand-bayes": HandBayesBot, "naive": NaiveBot, "noise": NoiseBot}


def run_table(seat0, seeds, out_path, n_rounds: int = 4) -> list[dict]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["seed"] for line in out_path.read_text().splitlines() if line.strip()}
    table = FiggieTable(n_rounds)
    for s in seeds:
        if s in done:
            continue
        players = [seat0()] + [HandBayesBot() for _ in range(N_PLAYERS - 1)]
        r = table.play(s, players)
        with open(out_path, "a") as fh:
            fh.write(json.dumps({"seed": s, "player": players[0].name, "score": r.scores[0],
                                 "trading_gain": r.scores[0] - r.no_trade_scores[0],
                                 "scores": r.scores, "goal": SUITS[r.goal], "counts": list(r.counts),
                                 "final_hands": r.final_hands, "n_trades": len(r.trades),
                                 "seat0_trades": sum(1 for t in r.trades if 0 in (t["buyer"], t["seller"])),
                                 "format_errors": sum(bool(i.get("format_error")) for i in r.infos[0]),
                                 "rounds": len(r.infos[0]), "seat0_info": r.infos[0]}) + "\n")
    return [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
