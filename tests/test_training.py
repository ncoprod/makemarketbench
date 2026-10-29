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
