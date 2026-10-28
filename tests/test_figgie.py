import numpy as np
import pytest

from makemarket.figgie_eval import parse_orders
from makemarket.games.figgie import (
    LAYOUTS,
    FiggieTable,
    HandBayesBot,
    NaiveBot,
    NoiseBot,
    deal,
    goal_suit,
    layout_posterior,
    payout,
)


def test_layouts_and_goal():
    assert len(LAYOUTS) == 12 and all(sum(c) == 40 for c in LAYOUTS)
    for c in LAYOUTS:
        assert c[goal_suit(c)] in (8, 10)


def test_posterior_is_normalised_and_consistent():
    d = deal(5, 4)
    post = layout_posterior(d.hands[0].tolist())
    assert post.sum() == pytest.approx(1.0)
    assert post[LAYOUTS.index(d.counts)] > 0


def test_payout_is_pot():
    assert payout(np.array([3, 3, 2, 2]), 10).sum() == pytest.approx(200)


@pytest.mark.parametrize("seed", range(6))
def test_table_conserves_cards_and_chips(seed):
    r = FiggieTable(4).play(seed, [NoiseBot(), HandBayesBot(), NaiveBot(), NoiseBot()])
    hands = np.array(r.final_hands)
    assert (hands >= 0).all()
    assert (hands.sum(0) == np.array(r.counts)).all()
    assert sum(r.scores) == pytest.approx(0.0)  # zero-sum
    assert sum(r.no_trade_scores) == pytest.approx(0.0)


def test_parse_orders():
    out = parse_orders('think... {"orders": [{"suit": "Hearts", "side": "buy", "price": 6}]}')
    assert out == [(2, 0, 6)]
    with pytest.raises(ValueError):
        parse_orders('{"orders": [{"suit": "hearts", "side": "buy", "price": -1}]}')
