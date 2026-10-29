"""Pure-Python reference implementation of the matching engine.

Semantics are identical to the C++ engine in ``cpp/src/order_book.cpp`` (see the
header comment there); ``tests/test_parity.py`` checks this on randomised event
streams. This backend needs no compiler, so training notebooks can use it.
"""

from __future__ import annotations

from bisect import bisect_left, insort

BUY, SELL = 0, 1
_MASK = (1 << 64) - 1
_FNV_OFFSET = 1469598103934665603
_FNV_PRIME = 1099511628211

# A fill is a tuple: (taker_id, maker_id, taker_owner, maker_owner, taker_side, price, qty)
Fill = tuple[int, int, int, int, int, int, int]


class OrderBook:
    def __init__(self) -> None:
        # price -> {order_id: [owner, qty]} ; dicts keep insertion (time) order.
        self._bids: dict[int, dict[int, list[int]]] = {}
        self._asks: dict[int, dict[int, list[int]]] = {}
        self._bid_prices: list[int] = []  # ascending; best bid is the last element
        self._ask_prices: list[int] = []  # ascending; best ask is the first element
        self._index: dict[int, tuple[int, int]] = {}  # id -> (side, price)

    # ------------------------------------------------------------------ orders
    def submit_limit(self, id: int, owner: int, side: int, price: int, qty: int,
                     ioc: bool = False) -> list[Fill]:
        return self._submit(id, owner, side, price, qty, ioc)

    def submit_market(self, id: int, owner: int, side: int, qty: int) -> list[Fill]:
        return self._submit(id, owner, side, None, qty, True)

    def _submit(self, id, owner, side, price, qty, ioc) -> list[Fill]:
        if side not in (BUY, SELL):
            raise ValueError("side must be 0 (buy) or 1 (sell)")
        if qty <= 0:
            raise ValueError("qty must be positive")
        if id == _MASK:
            raise ValueError("order id 2^64-1 is reserved")
        if id in self._index:
            raise ValueError("duplicate live order id")
        fills: list[Fill] = []
        if side == BUY:
            book, prices = self._asks, self._ask_prices
            while qty > 0 and prices:
                p = prices[0]
                if price is not None and p > price:
                    break
                qty = self._consume(book, prices, 0, p, id, owner, side, qty, fills)
        else:
            book, prices = self._bids, self._bid_prices
            while qty > 0 and prices:
                p = prices[-1]
                if price is not None and p < price:
                    break
                qty = self._consume(book, prices, -1, p, id, owner, side, qty, fills)
        if qty > 0 and not ioc and price is not None:
            own_book = self._bids if side == BUY else self._asks
            own_prices = self._bid_prices if side == BUY else self._ask_prices
            level = own_book.get(price)
            if level is None:
                level = own_book[price] = {}
                insort(own_prices, price)
            level[id] = [owner, qty]
            self._index[id] = (side, price)
        return fills

    def _consume(self, book, prices, pos, p, id, owner, side, qty, fills) -> int:
        level = book[p]
        while qty > 0 and level:
            maker_id = next(iter(level))
            maker = level[maker_id]
            q = qty if qty < maker[1] else maker[1]
            fills.append((id, maker_id, owner, maker[0], side, p, q))
            qty -= q
            maker[1] -= q
            if maker[1] == 0:
                del level[maker_id]
                del self._index[maker_id]
        if not level:
            del book[p]
            prices.pop(pos)
        return qty

    def cancel(self, id: int, owner: int) -> bool:
        """Cancel on behalf of ``owner``; False if not live or owned by someone else."""
        loc = self._index.get(id)
        if loc is None:
            return False
        side, price = loc
        book = self._bids if side == BUY else self._asks
        prices = self._bid_prices if side == BUY else self._ask_prices
        level = book[price]
        if level[id][0] != owner:
            return False
        del self._index[id]
        del level[id]
        if not level:
            del book[price]
            prices.pop(bisect_left(prices, price))
        return True

    def cancel_owner(self, owner: int) -> int:
        ids = []
        for book in (self._bids, self._asks):
            for level in book.values():
                ids.extend(oid for oid, (o, _) in level.items() if o == owner)
        for oid in ids:
            self.cancel(oid, owner)
        return len(ids)

    # ----------------------------------------------------------------- queries
    def best_bid(self) -> int | None:
        return self._bid_prices[-1] if self._bid_prices else None

    def best_ask(self) -> int | None:
        return self._ask_prices[0] if self._ask_prices else None

    def volume_at(self, side: int, price: int) -> int:
        book = self._bids if side == BUY else self._asks
        level = book.get(price)
        return sum(q for _, q in level.values()) if level else 0

    def depth(self, side: int, levels: int) -> list[tuple[int, int]]:
        if side == BUY:
            prices, book = reversed(self._bid_prices), self._bids
        else:
            prices, book = iter(self._ask_prices), self._asks
        out = []
        for p in prices:
            if len(out) >= levels:
                break
            out.append((p, sum(q for _, q in book[p].values())))
        return out

    def order_count(self) -> int:
        return len(self._index)

    def has_order(self, id: int) -> bool:
        return id in self._index

    def clear(self) -> None:
        self.__init__()

    def replay(self, op, id, owner, side, price, qty) -> dict:
        n = len(op)
        if not (len(id) == len(owner) == len(side) == len(price) == len(qty) == n):
            raise ValueError("replay: arrays must have equal length")
        h = _FNV_OFFSET
        fills_n = volume = cancels_ok = 0
        for i in range(n):
            o = op[i]
            if o == 0:
                fills = self._submit(id[i], owner[i], side[i], price[i], qty[i], False)
            elif o == 1:
                fills = self._submit(id[i], owner[i], side[i], price[i], qty[i], True)
            elif o == 2:
                fills = self._submit(id[i], owner[i], side[i], None, qty[i], True)
            elif o == 3:
                cancels_ok += self.cancel(id[i], owner[i])
                continue
            else:
                raise ValueError("replay: bad op code")
            for f in fills:
                fills_n += 1
                volume += f[6]
                for v in (f[1], f[5], f[6]):
                    h = ((h ^ (v & _MASK)) * _FNV_PRIME) & _MASK
        return {"fills": fills_n, "volume": volume, "cancels_ok": cancels_ok,
                "checksum": h & 0x7FFFFFFFFFFFFFFF}


class Ledger:
    def __init__(self) -> None:
        self._pos: dict[int, int] = {}
        self._cash: dict[int, int] = {}

    def apply(self, f: Fill) -> None:
        _, _, taker, maker, side, price, qty = f
        sgn = 1 if side == BUY else -1
        self._pos[taker] = self._pos.get(taker, 0) + sgn * qty
        self._cash[taker] = self._cash.get(taker, 0) - sgn * price * qty
        self._pos[maker] = self._pos.get(maker, 0) - sgn * qty
        self._cash[maker] = self._cash.get(maker, 0) + sgn * price * qty

    def position(self, owner: int) -> int:
        return self._pos.get(owner, 0)

    def cash(self, owner: int) -> int:
        return self._cash.get(owner, 0)

    def mark(self, owner: int, value: int) -> int:
        return self.cash(owner) + self.position(owner) * value

    def clear(self) -> None:
        self._pos.clear()
        self._cash.clear()
