"""Held-out acquisition must not expose scores before candidate selection closes."""

import json
import sys
from types import SimpleNamespace

from train import mlx_evaluate


def test_raw_only_retains_acquisition_without_scoring(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["mlx_evaluate.py", "--model", "test", "--revision", "fixed",
                                     "--label", "policy", "--episodes", "2", "--seed", "200",
                                     "--games", "dice", "--out", str(tmp_path), "--raw-only"])
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(set_cache_limit=lambda _: None,
                                                                 get_peak_memory=lambda: 123))
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=lambda *a, **kw: (None, None)))
    monkeypatch.setattr(mlx_evaluate, "run_player", lambda *a, **kw: [{"seed": 200}, {"seed": 201}])

    def forbidden(*args):
        raise AssertionError("held-out acquisition exposed aggregate scores")

    monkeypatch.setattr(mlx_evaluate, "summarize", forbidden)
    mlx_evaluate.main()
    record = json.loads((tmp_path / "acquisition_dice.json").read_text())
    assert record["episodes"] == 2 and record["seeds"] == [200, 201]
    assert record["peak_memory_bytes"] == 123 and record["provenance"]["config"]["raw_only"]
    assert not list(tmp_path.glob("leaderboard*"))
