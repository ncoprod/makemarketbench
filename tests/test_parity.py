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
        return ("cancel", book.cancel(i))
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
