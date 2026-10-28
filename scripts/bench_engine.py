"""Matching-engine throughput: C++ (native / pybind11 batch / pybind11 per call)
vs the pure-Python reference. Writes results/engine_bench.json.

All variants replay the *same* event stream and must produce the same fill
checksum; we report the best of ``--repeats`` runs (events per second, where an
event is a limit, IOC, market or cancel message).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from makemarket.engine import get_backend
from makemarket.engine.streams import random_stream, stream_lists

ROOT = Path(__file__).resolve().parents[1]


def per_call(book, lists):
    op, ids, owner, side, price, qty = lists
    sl, sm, cancel = book.submit_limit, book.submit_market, book.cancel
    fills = 0
    for o, i, w, s, p, q in zip(op, ids, owner, side, price, qty):
        if o == 0:
            fills += len(sl(i, w, s, p, q, False))
        elif o == 1:
            fills += len(sl(i, w, s, p, q, True))
        elif o == 2:
            fills += len(sm(i, w, s, q))
        else:
            cancel(i, w)
    return fills


def best_of(fn, repeats):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=1_000_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--native", default=str(ROOT / "build/native/bench_book"))
    args = ap.parse_args()

    s = random_stream(args.events, seed=2026)
    lists = stream_lists(s)
    n = args.events
    py, cpp = get_backend("py"), get_backend("cpp")
    res = {}

    t, st = best_of(lambda: cpp.OrderBook().replay(*lists), args.repeats)
    res["cpp_pybind_batch"] = {"seconds": t, "events_per_sec": n / t, "checksum": st["checksum"]}
    ref_checksum, fills = st["checksum"], st["fills"]

    t, f = best_of(lambda: per_call(cpp.OrderBook(), lists), args.repeats)
    res["cpp_pybind_per_call"] = {"seconds": t, "events_per_sec": n / t, "fills": f}

    t, st = best_of(lambda: py.OrderBook().replay(*lists), args.repeats)
    res["python_batch"] = {"seconds": t, "events_per_sec": n / t, "checksum": st["checksum"]}

    t, f = best_of(lambda: per_call(py.OrderBook(), lists), args.repeats)
    res["python_per_call"] = {"seconds": t, "events_per_sec": n / t, "fills": f}

    native = Path(args.native)
    if native.exists():
        blob = ROOT / "build" / "bench_stream.bin"
        cols = np.stack([s[k].astype(np.int64) for k in ("op", "id", "owner", "side", "price", "qty")])
        with open(blob, "wb") as fh:
            fh.write(np.int64(n).tobytes())
            fh.write(cols.tobytes())
        out = subprocess.run([str(native), str(blob), str(max(args.repeats, 5))],
                             capture_output=True, text=True, check=True).stdout
        nat = json.loads(out)
        res["cpp_native"] = {"seconds": nat["best_seconds"], "events_per_sec": nat["events_per_sec"],
                             "checksum": nat["checksum"]}

    for k, v in res.items():
        if "checksum" in v:
            assert v["checksum"] == ref_checksum, f"checksum mismatch in {k}"
        if "fills" in v:
            assert v["fills"] == fills, f"fill count mismatch in {k}"

    out = {
        "events": n,
        "fills": fills,
        "event_mix": {"gtc": float((s["op"] == 0).mean()), "ioc": float((s["op"] == 1).mean()),
                      "market": float((s["op"] == 2).mean()), "cancel": float((s["op"] == 3).mean())},
        "machine": f"{platform.machine()} {platform.platform()}",
        "python": platform.python_version(),
        "results": res,
        "speedup_native_vs_python_batch": res.get("cpp_native", res["cpp_pybind_batch"])["events_per_sec"]
        / res["python_batch"]["events_per_sec"],
    }
    path = ROOT / "results" / "engine_bench.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in res.items():
        print(f"{k:22s} {v['events_per_sec']:>14,.0f} events/s")


if __name__ == "__main__":
    main()
