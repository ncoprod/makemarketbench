"""C++ vs pure-Python parity on randomised event streams."""

import pytest

from makemarket.engine import BUY, SELL, cpp_available, get_backend
from makemarket.engine.streams import random_stream, stream_lists

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def _step(book, op, i, owner, side, price, qty):
    try:
        if op == 0:
            return ("fills", book.submit_limit(i, owner, side, price, qty, False))
        if op == 1:
            return ("fills", book.submit_limit(i, owner, side, price, qty, True))
        if op == 2:
            return ("fills", book.submit_market(i, owner, side, qty))
        return ("cancel", book.cancel(i, owner))
    except ValueError as e:  # e.g. duplicate id never happens with these streams
        return ("error", str(e))


@pytest.mark.parametrize("seed", range(12))
def test_event_by_event_parity(seed):
    s = random_stream(3000, seed=seed, width=5 + seed)
    py, cpp = get_backend("py").OrderBook(), get_backend("cpp").OrderBook()
    for ev in zip(*stream_lists(s)):
        a, b = _step(py, *ev), _step(cpp, *ev)
        assert [tuple(x) for x in a[1]] == [tuple(x) for x in b[1]] if a[0] == "fills" else a == b
        assert py.best_bid() == cpp.best_bid() and py.best_ask() == cpp.best_ask()
        assert py.order_count() == cpp.order_count()
    for side in (BUY, SELL):
        assert py.depth(side, 1000) == [tuple(x) for x in cpp.depth(side, 1000)]


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_replay_parity(seed):
    s = random_stream(20000, seed=seed)
    a = get_backend("py").OrderBook().replay(*stream_lists(s))
    b = get_backend("cpp").OrderBook().replay(*stream_lists(s))
    assert a == dict(b)
    assert a["fills"] > 1000  # the stream actually trades


def test_ledger_parity():
    s = random_stream(5000, seed=7)
    mods = [get_backend("py"), get_backend("cpp")]
    books = [m.OrderBook() for m in mods]
    ledgers = [m.Ledger() for m in mods]
    for ev in zip(*stream_lists(s)):
        for book, ledger in zip(books, ledgers):
            res = _step(book, *ev)
            if res[0] == "fills":
                for f in res[1]:
                    ledger.apply(tuple(f))
    for owner in range(8):
        assert ledgers[0].position(owner) == ledgers[1].position(owner)
        assert ledgers[0].cash(owner) == ledgers[1].cash(owner)
    assert sum(ledgers[0].position(o) for o in range(8)) == 0


def test_cancel_owner_mismatch_parity():
    results = []
    for mod in (get_backend("py"), get_backend("cpp")):
        b = mod.OrderBook()
        b.submit_limit(1, 5, BUY, 100, 3)
        b.submit_limit(2, 6, SELL, 105, 3)
        out = [b.cancel(1, 6), b.cancel(2, 5), b.has_order(1), b.has_order(2),
               b.cancel(1, 5), b.cancel(2, 6), b.order_count()]
        results.append(out)
    assert results[0] == results[1] == [False, False, True, True, True, True, 0]


def test_stream_exercises_foreign_cancels():
    s = random_stream(20000, seed=3)
    a = get_backend("py").OrderBook().replay(*stream_lists(s))
    n_cancel = int((s["op"] == 3).sum())
    assert 0 < a["cancels_ok"] < n_cancel


@pytest.mark.parametrize("width,p_cancel", [(300, 0.45), (2, 0.1)])
def test_replay_parity_deep_and_shallow_books(width, p_cancel):
    """Deep books (many levels, many resting orders -> index rehash and
    backward-shift deletions) and very shallow ones (heavy crossing)."""
    s = random_stream(60000, seed=width, width=width, p_cancel=p_cancel)
    py, cpp = get_backend("py").OrderBook(), get_backend("cpp").OrderBook()
    a, b = py.replay(*stream_lists(s)), cpp.replay(*stream_lists(s))
    assert a == dict(b)
    assert py.order_count() == cpp.order_count()
    for side in (BUY, SELL):
        assert py.depth(side, 10_000) == [tuple(x) for x in cpp.depth(side, 10_000)]
