import pytest

from makemarket.games import make_game
from makemarket.llm import extract_json, parse_quote
from makemarket.market import Observation, validate_quote
from makemarket.prompts import system_prompt, user_prompt


def test_extract_last_json_after_reasoning():
    text = 'Thinking {"bid": 1} ... final:\n```json\n{"bid": 20.5, "bid_size": 3, "ask": 24, "ask_size": 2}\n```'
    assert extract_json(text)["bid"] == 20.5


def test_think_blocks_are_ignored():
    text = '<think>maybe {"bid": 99, "ask": 1}</think>{"bid": 10, "bid_size": 1, "ask": 12, "ask_size": 1}'
    assert parse_quote(text, 10).bid == 10


@pytest.mark.parametrize("bad", [
    "no json here",
    '{"bid": 30, "bid_size": 3, "ask": 29, "ask_size": 3}',  # crossed
    '{"bid": 20, "bid_size": 11, "ask": 25, "ask_size": 3}',  # too big
    '{"bid": 20, "bid_size": 2.5, "ask": 25, "ask_size": 3}',  # fractional size
    '{"bid": "x", "bid_size": 2, "ask": 25, "ask_size": 3}',
    '<think>unterminated {"bid": 1, "bid_size": 1, "ask": 2, "ask_size": 1}',
])
def test_invalid_quotes_rejected(bad):
    with pytest.raises(ValueError):
        parse_quote(bad, 10)


def test_prices_round_to_tick():
    q = validate_quote({"bid": 20.04, "bid_size": 1, "ask": 20.26, "ask_size": 0}, 10)
    assert (q.bid, q.ask) == (20.0, 20.3)


def test_prompt_renders_history():
    g = make_game("dice")
    ep = g.episode(1)
    obs = Observation(g, 2, 6, ep.reveals[:3], [], 0, 0.0, ep.public_beliefs[2], 0.0)
    text = user_prompt(obs)
    assert "Round 3 of 6" in text and "Die #2" in text
    sp = system_prompt(g)
    assert "30%" in sp and "Normal(0, 4^2)" in sp


def test_hint_ablation_adds_posterior_moments():
    g = make_game("dice")
    ep = g.episode(1)
    obs = Observation(g, 1, 6, ep.reveals[:2], [], 0, 0.0, ep.public_beliefs[1], 0.0)
    assert "Hint" not in user_prompt(obs)
    assert f"E[V] = {ep.public_beliefs[1].mean:.2f}" in user_prompt(obs, hint=True)
