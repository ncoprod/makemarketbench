"""Matching-engine backends.

``get_backend("cpp")`` returns the compiled pybind11 module, ``get_backend("py")``
the pure-Python reference. ``"auto"`` prefers C++ and falls back to Python.
Both expose ``OrderBook`` and ``Ledger`` with the same API, and fills are plain
tuples ``(taker_id, maker_id, taker_owner, maker_owner, taker_side, price, qty)``.
"""

from __future__ import annotations

import os
from types import ModuleType

from . import pybook

BUY, SELL = 0, 1


def cpp_available() -> bool:
    try:
        from makemarket import _engine  # noqa: F401
    except ImportError:
        return False
    return True


def get_backend(name: str | None = None) -> ModuleType:
    name = name or os.environ.get("MMB_BACKEND", "auto")
    if name == "py":
        return pybook
    if name in ("cpp", "auto"):
        try:
            from makemarket import _engine
        except ImportError:
            if name == "cpp":
                raise
            return pybook
        return _engine
    raise ValueError(f"unknown backend {name!r}")


class Exchange:
    """Order book plus ledger: every fill is booked to both counterparties."""

    def __init__(self, backend: str | None = None) -> None:
        mod = get_backend(backend)
        self.backend = "cpp" if mod is not pybook else "py"
        self.book = mod.OrderBook()
        self.ledger = mod.Ledger()
        self._next_id = 1

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def limit(self, owner: int, side: int, price: int, qty: int, ioc: bool = False) -> list:
        fills = self.book.submit_limit(self.new_id(), owner, side, price, qty, ioc)
        for f in fills:
            self.ledger.apply(f)
        return fills

    def market(self, owner: int, side: int, qty: int) -> list:
        fills = self.book.submit_market(self.new_id(), owner, side, qty)
        for f in fills:
            self.ledger.apply(f)
        return fills
