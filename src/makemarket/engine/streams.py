"""Random order-event streams for parity tests and benchmarks."""

from __future__ import annotations

import numpy as np

OP_GTC, OP_IOC, OP_MARKET, OP_CANCEL = 0, 1, 2, 3


def random_stream(n: int, seed: int = 0, n_owners: int = 8, mid: int = 10_000,
                  width: int = 20, p_cancel: float = 0.3, p_ioc: float = 0.15,
                  p_market: float = 0.05) -> dict[str, np.ndarray]:
    """Generate ``n`` events. Cancels target a uniformly random earlier id, which
    may already be filled or cancelled, and 10% are sent by a random owner
    (exercising both ``False`` paths)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    op = np.full(n, OP_GTC, dtype=np.int8)
    op[u < p_cancel + p_ioc + p_market] = OP_MARKET
    op[u < p_cancel + p_ioc] = OP_IOC
    op[u < p_cancel] = OP_CANCEL
    op[0] = OP_GTC
    ids = np.arange(1, n + 1, dtype=np.uint64)
    # cancels refer to a random earlier order id
    earlier = (rng.random(n) * np.arange(n)).astype(np.uint64) + 1
    ids = np.where(op == OP_CANCEL, earlier, ids).astype(np.uint64)
    side = rng.integers(0, 2, n).astype(np.int8)
    # buys centred slightly below mid, sells slightly above, so books overlap often
    offset = rng.integers(-width, width + 1, n)
    price = (mid + offset + np.where(side == 0, -2, 2)).astype(np.int64)
    qty = rng.integers(1, 11, n).astype(np.int64)
    owner = rng.integers(0, n_owners, n).astype(np.int32)
    # cancels are sent by the order's owner 90% of the time, by someone else otherwise
    cancel_idx = np.nonzero(op == OP_CANCEL)[0]
    true_owner = owner[(ids[cancel_idx] - 1).astype(np.int64)]
    honest = rng.random(len(cancel_idx)) < 0.9
    owner[cancel_idx] = np.where(honest, true_owner, rng.integers(0, n_owners, len(cancel_idx)))
    return {"op": op, "id": ids, "owner": owner, "side": side, "price": price, "qty": qty}


def stream_lists(s: dict[str, np.ndarray]) -> tuple[list, ...]:
    """Convert a stream to Python lists (argument order of ``OrderBook.replay``)."""
    return tuple(s[k].tolist() for k in ("op", "id", "owner", "side", "price", "qty"))
