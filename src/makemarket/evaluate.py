"""Run players on fixed seeds and store one JSON line per episode."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .market import EpisodeResult, play_episode


def episode_to_dict(r: EpisodeResult) -> dict:
    rounds = []
    for rec, info, exp in zip(r.rounds, r.round_info, r.expectations):
        d = {
            "t": rec.t,
            "reveal": rec.reveal,
            "quote": rec.quote.to_dict() if rec.quote else None,
            "traded_side": rec.traded_side,
            "qty": rec.qty,
            "price": rec.price,
            "position": rec.position,
            "cash": rec.cash,
            **{f"exp_{k}": v for k, v in exp.items()},
        }
        if info:
            d["info"] = info
        rounds.append(d)
    return {"game": r.game, "seed": r.seed, "player": r.player, "value": r.value,
            "realized_pnl": r.realized_pnl, "rounds": rounds}


def load_results(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def run_player(game, make_player, seeds: list[int], out_path: str | Path, workers: int = 1,
               backend: str | None = None, log=print) -> list[dict]:
    """Play ``seeds`` with fresh players from ``make_player()``; resumable (seeds
    already present in ``out_path`` are skipped). Returns all episode dicts."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {d["seed"] for d in load_results(out_path)}
    todo = [s for s in seeds if s not in done]
    lock = threading.Lock()
    t0 = time.time()

    def work(seed: int) -> dict:
        res = episode_to_dict(play_episode(game, seed, make_player(), backend))
        with lock, open(out_path, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        return res

    n_done = 0
    if workers <= 1:
        for s in todo:
            work(s)
            n_done += 1
    else:
        with ThreadPoolExecutor(workers) as pool:
            futs = {pool.submit(work, s): s for s in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # keep going; the seed can be retried later
                    log(f"[{out_path.stem}] seed {futs[f]} failed: {e}")
                    continue
                n_done += 1
                if n_done % 5 == 0:
                    log(f"[{out_path.stem}] {n_done}/{len(todo)} episodes, {time.time() - t0:.0f}s")
    keep = set(seeds)
    return [d for d in load_results(out_path) if d["seed"] in keep]
