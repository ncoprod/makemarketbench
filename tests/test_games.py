import itertools
import math

import numpy as np
import pytest

from makemarket.games import make_game
from makemarket.games.base import Belief, Counterparty, Quote, flow_likelihood, round_expectations
from makemarket.games.cards import posterior
from makemarket.market import play_episode
from makemarket.players import BASELINES, BayesQuoter, best_quote


def test_dice_prior_moments():
    ep = make_game("dice").episode(0)
    b = ep.public_beliefs[0]
    assert b.mean == pytest.approx(28.0)
    assert b.sd == pytest.approx(math.sqrt(8 * 35 / 12))
    assert ep.value == sum(ep.meta["dice"])


def test_cards_posterior_matches_brute_force():
    """Enumeration-by-multiset posterior equals brute force over card subsets."""
    known, hints = [13, 2], [("faces", 2), ("parity", 1)]
    b = posterior(known, hints)
    deck = [r for r in range(1, 14) for _ in range(4)]
    for r in known:
        deck.remove(r)
    counts: dict[int, int] = {}
    for combo in itertools.combinations(range(len(deck)), 3):
        hand = known + [deck[i] for i in combo]
        if sum(x >= 11 for x in hand) == 2 and sum(hand) % 2 == 1:
            counts[sum(hand)] = counts.get(sum(hand), 0) + 1
    tot = sum(counts.values())
    brute = {v: c / tot for v, c in counts.items()}
    assert dict(zip(b.values.tolist(), b.probs.tolist())) == pytest.approx(brute)


def test_cards_hints_are_true():
    g = make_game("cards")
    for s in range(30):
        ep = g.episode(s)
        for b in ep.public_beliefs:
            assert ep.value in b.values.tolist()


@pytest.mark.parametrize("game", ["dice", "cards"])
def test_true_value_always_in_support(game):
    g = make_game(game)
    for s in range(20):
        ep = g.episode(s)
        assert all(ep.value in b.values for b in ep.public_beliefs)


def test_expected_pnl_matches_monte_carlo_through_engine():
    """Closed-form round expectation == average engine PnL over counterparties."""
    g = make_game("dice")
    p = g.params
    rng = np.random.default_rng(0)
    q = Quote(25.3, 4, 30.1, 6)
    m_pub, value = 27.0, 29
    from makemarket.engine import BUY, SELL, Exchange

    n, tot = 40000, 0.0
    for _ in range(n):
        cp = Counterparty(bool(rng.random() < p.alpha), float(rng.standard_normal()),
                          int(rng.integers(1, p.d_max + 1)))
        ex = Exchange("py")
        ex.limit(0, BUY, 253, 4)
        ex.limit(0, SELL, 301, 6)
        if cp.informed:
            if value * 10 > 301:
                ex.limit(1, BUY, value * 10, 6, ioc=True)
            elif value * 10 < 253:
                ex.limit(1, SELL, value * 10, 4, ioc=True)
        else:
            u = m_pub + p.noise_sd * cp.z
            if cp.z > 0:
                ex.limit(1, BUY, math.floor(u * 10 + 1e-9), cp.demand, ioc=True)
            else:
                ex.limit(1, SELL, math.ceil(u * 10 - 1e-9), cp.demand, ioc=True)
        tot += ex.ledger.mark(0, value * 10) / 10
    mc = tot / n
    exp = round_expectations(q, value, m_pub, p)["expected_pnl"]
    assert mc == pytest.approx(exp, abs=0.06)


def test_flow_likelihood_is_a_distribution_over_observations():
    p = make_game("dice").params
    q = Quote(25.0, 3, 31.0, 7)
    vals = np.arange(10, 50)
    total = flow_likelihood(vals, q, 28.0, p, None, 0)
    for qty in range(1, 11):
        total = total + flow_likelihood(vals, q, 28.0, p, "sold", qty)
        total = total + flow_likelihood(vals, q, 28.0, p, "bought", qty)
    assert np.allclose(total, 1.0)


def test_best_quote_beats_perturbations():
    g = make_game("cards")
    b = g.episode(4).public_beliefs[3]
    q, _ = best_quote(b, b.mean, g.params)
    best = round_expectations(q, b, b.mean, g.params)["expected_pnl"]
    for d_ask, d_bid, ds in [(0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0), (0, -0.5, 0), (0, 0, 1), (0, 0, -1)]:
        alt = Quote(q.bid + d_bid, max(q.bid_size + ds, 0), q.ask + d_ask, max(q.ask_size + ds, 0))
        assert round_expectations(alt, b, b.mean, g.params)["expected_pnl"] <= best + 1e-9


def test_reference_beats_baselines_on_average():
    g = make_game("dice")
    scores = {}
    for name in ("bayes", "mean-sd", "random"):
        tot = 0.0
        for s in range(100, 160):
            r = play_episode(g, s, BASELINES[name]())
            tot += sum(e["expected_pnl"] for e in r.expectations)
        scores[name] = tot / 60
    assert scores["bayes"] > scores["mean-sd"] > scores["random"]


def test_backends_give_identical_episodes():
    from makemarket.engine import cpp_available

    if not cpp_available():
        pytest.skip("no C++ extension")
    g = make_game("cards")
    for s in range(5):
        a = play_episode(g, s, BayesQuoter(), backend="py")
        b = play_episode(g, s, BayesQuoter(), backend="cpp")
        assert [(r.traded_side, r.qty, r.position, r.cash) for r in a.rounds] == \
               [(r.traded_side, r.qty, r.position, r.cash) for r in b.rounds]


def test_belief_reweight_keeps_normalisation():
    b = Belief(np.arange(5), np.ones(5) / 5)
    b2 = b.reweight(np.array([0, 1, 2, 3, 4.0]))
    assert b2.probs.sum() == pytest.approx(1.0) and b2.probs[0] == 0
