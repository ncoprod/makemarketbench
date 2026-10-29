"""CPU checks of the GPU script's rewards and exported held-out evaluation."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def kernel():
    spec = importlib.util.spec_from_file_location(
        "kaggle_training", Path(__file__).parents[1] / "train/grpo_unsloth_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_reward_prefers_reference_to_invalid_output(kernel):
    rows = kernel.build_prompt_set(1, ["dice"])
    row = rows[0]
    keys = ("game", "values", "probs", "m_pub", "ref_quote")
    kwargs = {k: [row[k], row[k]] for k in keys}
    fmt, edge = kernel.make_reward_fns()
    completions = [kernel.sft_target(row), "garbage"]
    assert fmt(None, completions, **kwargs) == [0.5, -1.0]
    assert edge(None, completions, **kwargs) == pytest.approx([0.0, -2.0])


def test_kaggle_evaluation_writes_paired_json_without_model_conversion(kernel, monkeypatch, tmp_path):
    from makemarket.players import BayesQuoter

    def oracle_model(model, tokenizer, label):
        player = BayesQuoter()
        player.name = label
        return player

    monkeypatch.setattr(kernel, "WORK", tmp_path)
    monkeypatch.setattr(kernel, "HFQuoter", oracle_model)
    args = Namespace(games=["dice", "cards"], eval_episodes=3, eval_seed=200, label="test-policy")
    kernel.evaluate(None, None, args)
    for game in args.games:
        result = json.loads((tmp_path / f"results/leaderboard_{game}.json").read_text())
        assert result["seeds"] == [200, 201, 202]
        assert {r["player"] for r in result["rows"]} == {"bayes", "test-policy"}
        assert all(r["episodes"] == 3 for r in result["rows"])
        assert all(r["pct_of_bayes"] == pytest.approx(100) for r in result["rows"])


def test_rloo_baseline_is_leave_one_out_and_translation_invariant(monkeypatch):
    import importlib

    import numpy as np

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "train"))
    rloo = importlib.import_module("mlx_rloo")
    advantages = rloo.leave_one_out([1, 2, 3, 4])
    assert advantages == pytest.approx([-2, -2 / 3, 2 / 3, 2])
    assert advantages.sum() == pytest.approx(0)
    np.testing.assert_allclose(advantages, rloo.leave_one_out([101, 102, 103, 104]))


def test_final_cohort_rejects_duplicates_missing_seeds_and_short_episodes(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "final_scoring", Path(__file__).parents[1] / "scripts/score_final.py")
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    path = tmp_path / "episodes.jsonl"
    rows = [{"seed": s, "rounds": [{}] * 6} for s in range(200, 400)]

    def write(records):
        path.write_text("".join(json.dumps(r) + "\n" for r in records))

    write(rows)
    assert len(scoring.cohort(path)) == 200
    write(rows[:-1])
    assert scoring.cohort(path) is None
    write(rows + [rows[0]])
    assert scoring.cohort(path) is None
    rows[0]["rounds"] = [{}] * 5
    write(rows)
    assert scoring.cohort(path) is None


def test_final_candidate_stays_frozen_when_validation_files_change(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "final_candidate", Path(__file__).parents[1] / "train/final_candidate.py")
    selection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selection)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(selection, "receipt", lambda config, revision: {"model_revision": revision})
    calls = []
    monkeypatch.setattr(selection.subprocess, "run", lambda command, **kw: calls.append(command))
    monkeypatch.setattr("sys.argv", ["final_candidate.py", "--model", "test-model", "--revision", "fixed"])
    for method, percent in (("sft", 90), ("dagger", 80), ("rloo", 70)):
        label = f"qwen3.5-4b-{method}"
        adapter = tmp_path / f"train/runs/{label}"
        adapter.mkdir(parents=True)
        (adapter / "adapters.safetensors").write_bytes(method.encode())
        for game in ("dice", "cards"):
            path = tmp_path / f"results/validation/qwen35-{method}/leaderboard_{game}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"seeds": list(range(80000, 80100)),
                                        "rows": [{"player": label, "episodes": 100,
                                                  "pct_of_bayes": percent}]}))
    (tmp_path / "results/training").mkdir()
    selection.main()
    frozen = tmp_path / "results/training/selection.json"
    before = frozen.read_bytes()
    for path in (tmp_path / "results/validation").rglob("*.json"):
        path.write_text("invalid JSON: a resumed test must not reselect")
    selection.main()
    assert frozen.read_bytes() == before
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert "qwen3.5-4b-sft" in calls[0]
