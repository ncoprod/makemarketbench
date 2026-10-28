import pytest

from makemarket.engine import BUY, SELL, Exchange, cpp_available, get_backend

BACKENDS = ["py"] + (["cpp"] if cpp_available() else [])


@pytest.fixture(params=BACKENDS)
def mod(request):
    return get_backend(request.param)


def test_price_time_priority(mod):
    b = mod.OrderBook()
    b.submit_limit(1, 10, SELL, 101, 5)
    b.submit_limit(2, 11, SELL, 100, 3)
    b.submit_limit(3, 12, SELL, 100, 4)
    assert b.best_ask() == 100
    fills = b.submit_limit(4, 20, BUY, 101, 10)
    assert [(f[1], f[5], f[6]) for f in fills] == [(2, 100, 3), (3, 100, 4), (1, 101, 3)]
    assert b.volume_at(SELL, 101) == 2
    assert b.best_bid() is None


def test_ioc_and_market_do_not_rest(mod):
    b = mod.OrderBook()
    assert b.submit_limit(1, 1, BUY, 99, 5, True) == []
    assert b.order_count() == 0
    b.submit_limit(2, 1, BUY, 99, 5)
    fills = b.submit_market(3, 2, SELL, 8)
    assert len(fills) == 1 and fills[0][6] == 5 and b.order_count() == 0


def test_errors(mod):
    b = mod.OrderBook()
    b.submit_limit(1, 1, BUY, 99, 5)
    with pytest.raises(ValueError):
        b.submit_limit(1, 1, BUY, 98, 1)
    with pytest.raises(ValueError):
        b.submit_limit(2, 1, BUY, 98, 0)


def test_depth_and_cancel(mod):
    b = mod.OrderBook()
    for i, p in enumerate([99, 98, 99, 97]):
        b.submit_limit(i + 1, 1, BUY, p, 2)
    assert b.depth(BUY, 2) == [(99, 4), (98, 2)]
    assert not b.cancel(1, 2)  # another owner cannot cancel
    assert b.cancel(1, 1) and not b.cancel(1, 1)
    assert b.depth(BUY, 5) == [(99, 2), (98, 2), (97, 2)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_exchange_ledger(backend):
    ex = Exchange(backend)
    ex.limit(7, SELL, 100, 4)
    ex.limit(8, BUY, 105, 3, ioc=True)
    assert ex.ledger.position(7) == -3 and ex.ledger.cash(7) == 300
    assert ex.ledger.position(8) == 3 and ex.ledger.mark(8, 110) == 30


@pytest.mark.parametrize("backend", BACKENDS)
def test_exchange_cancel_checks_owner(backend):
    ex = Exchange(backend)
    ex.limit(1, BUY, 100, 2)
    assert not ex.cancel(1, owner=2)
    assert ex.cancel(1, owner=1)
