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
    args = Namespace(games=["dice", "cards"], eval_episodes=3, eval_seed=200, label="test-policy", seed=0)
    kernel.evaluate(None, None, args)
    for game in args.games:
        result = json.loads((tmp_path / f"results/leaderboard_{game}.json").read_text())
        assert result["seeds"] == [200, 201, 202]
        assert {r["player"] for r in result["rows"]} == {"bayes", "test-policy"}
        assert all(r["episodes"] == 3 for r in result["rows"])
        assert all(r["pct_of_bayes"] == pytest.approx(100) for r in result["rows"])


def test_greedy_cuda_cache_reuses_repairs_and_stays_with_one_adapter(kernel):
    from makemarket.games import make_game
    from makemarket.market import Observation

    game = make_game("dice")
    ep = game.episode(80000)
    obs = Observation(game, 0, ep.n_rounds, ep.reveals[:1], [], 0, 0,
                      ep.public_beliefs[0], ep.public_beliefs[0].mean)
    player = kernel.HFQuoter(None, None, "adapter-a")
    responses = iter(["invalid", '{"bid": 1, "bid_size": 1, "ask": 99, "ask_size": 1}'])
    player._gen = lambda messages: next(responses)
    expected_quote, original = player.quote(obs)
    actual_quote, cached = player.quote(obs)
    assert actual_quote == expected_quote
    assert cached["attempts"] == original["attempts"]
    assert cached["cached"] and not original["cached"]
    assert cached["format_error"] and not cached["failed"]
    assert not kernel.HFQuoter(None, None, "adapter-b").cache


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


@pytest.mark.parametrize("mode", ["scalar", "masked", "vector"])
def test_training_recurrence_preserves_values_and_gradients(monkeypatch, mode):
    import importlib

    mx = pytest.importorskip("mlx.core")
    gd = pytest.importorskip("mlx_lm.models.gated_delta")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "train"))
    original = gd.gated_delta_ops
    monkeypatch.setattr(gd, "gated_delta_ops", original)
    importlib.import_module("mlx_memory").enable_gated_delta_training(chunk_size=4)
    with mx.stream(mx.cpu):
        mx.random.seed(123)
        q, k = [mx.random.normal((1, 17, 2, 8)) * 0.1 for _ in range(2)]
        v = mx.random.normal((1, 17, 4, 8)) * 0.1
        gate_shape = (1, 17, 4, 8) if mode == "vector" else (1, 17, 4)
        args = [q, k, v, mx.full(gate_shape, 0.9), mx.full((1, 17, 4), 0.2),
                mx.random.normal((1, 4, 8, 8)) * 0.1]
        mask = mx.arange(17)[None, :] % 3 != 0 if mode == "masked" else None

        def objective(fn, *values):
            output, state = fn(*values, mask=mask)
            return mx.square(output).sum() + mx.square(state).sum()

        expected = mx.value_and_grad(lambda *x: objective(original, *x), argnums=range(6))(*args)
        actual = mx.value_and_grad(lambda *x: objective(gd.gated_delta_ops, *x), argnums=range(6))(*args)
        for before, after in zip([expected[0], *expected[1]], [actual[0], *actual[1]]):
            assert mx.allclose(before, after, atol=1e-6, rtol=1e-5).item()


@pytest.mark.parametrize("decay", [0.9, 0.02])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_parallel_recurrence_full_chunks_strong_decay_and_gradients(monkeypatch, decay, dtype_name):
    import importlib

    mx = pytest.importorskip("mlx.core")
    gd = pytest.importorskip("mlx_lm.models.gated_delta")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "train"))
    parallel = importlib.import_module("gated_delta_chunkwise").gated_delta_chunkwise
    with mx.stream(mx.cpu):
        mx.random.seed(456)
        dtype = getattr(mx, dtype_name)
        q, k = [(mx.random.normal((1, 129, 2, 8)) * 0.1).astype(dtype) for _ in range(2)]
        v = (mx.random.normal((1, 129, 4, 5)) * 0.1).astype(dtype)
        args = [q, k, v, mx.full((1, 129, 4), decay), mx.full((1, 129, 4), 0.2),
                mx.random.normal((1, 4, 5, 8)) * 0.1]

        def objective(fn, *values):
            output, state = fn(*values)
            return mx.square(output.astype(mx.float32)).sum() + mx.square(state).sum()

        tolerance = 2e-3 if dtype_name == "bfloat16" else 1e-5
        for before, after in zip(gd.gated_delta_ops(*args), parallel(*args)):
            assert mx.allclose(before, after, atol=tolerance, rtol=tolerance).item()
        expected = mx.value_and_grad(lambda *x: objective(gd.gated_delta_ops, *x), argnums=range(6))(*args)
        actual = mx.value_and_grad(lambda *x: objective(parallel, *x), argnums=range(6))(*args)
        for before, after in zip([expected[0], *expected[1]], [actual[0], *actual[1]]):
            assert mx.allclose(before, after, atol=tolerance, rtol=tolerance).item()


@pytest.mark.parametrize('tied', [False, True])
def test_answer_window_preserves_full_causal_loss_and_every_parameter_gradient(monkeypatch, tied):
    import importlib
    from types import SimpleNamespace

    mx = pytest.importorskip('mlx.core')
    nn = pytest.importorskip('mlx.nn')
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import default_loss

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'train'))
    answer = importlib.import_module('mlx_answer_loss')

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(31, 8)

        def __call__(self, tokens):
            return mx.cumsum(self.embed_tokens(tokens), axis=1) * 0.1

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = nn.Linear(8, 31)
            self.args = SimpleNamespace(tie_word_embeddings=tied)

        def __call__(self, tokens):
            hidden = self.model(tokens)
            return (self.model.embed_tokens.as_linear(hidden) if tied else self.lm_head(hidden))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Text()

        def __call__(self, tokens):
            return self.language_model(tokens)

    with mx.stream(mx.cpu):
        mx.random.seed(99)
        model = Model()
        batch = mx.array([list(range(20)), list(reversed(range(20)))])
        lengths = mx.array([[14, 18], [15, 19]])
        expected = nn.value_and_grad(model, lambda m: default_loss(m, batch, lengths)[0])(model)
        actual = nn.value_and_grad(model, lambda m: answer.answer_loss(m, batch, lengths, width=8)[0])(model)
        assert actual[0].item() == pytest.approx(expected[0].item(), abs=1e-6)
        before, after = dict(tree_flatten(expected[1])), dict(tree_flatten(actual[1]))
        assert before.keys() == after.keys()
        for name in before:
            assert mx.allclose(before[name], after[name], atol=1e-6, rtol=1e-5).item(), name


def test_answer_window_rejects_even_one_dropped_supervised_token(monkeypatch):
    import importlib

    mx = pytest.importorskip('mlx.core')
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'train'))
    answer = importlib.import_module('mlx_answer_loss')
    with mx.stream(mx.cpu):
        options = dict(batch_size=1, max_seq_length=64, loop=False, width=8)
        assert len(list(answer.answer_batches([(list(range(26)), 25)], **options))) == 1
        with pytest.raises(ValueError, match='discard supervised tokens'):
            list(answer.answer_batches([(list(range(26)), 24)], **options))


def test_batched_cuda_repairs_only_invalid_unique_prompts(kernel):
    from dataclasses import replace

    from makemarket.games import make_game
    from makemarket.market import Observation

    game = make_game("dice")
    ep = game.episode(80000)
    first = Observation(game, 0, ep.n_rounds, ep.reveals[:1], [], 0, 0,
                        ep.public_beliefs[0], ep.public_beliefs[0].mean)
    second = replace(first, position=1)
    player, calls = kernel.HFQuoter(None, None, "batched"), []
    valid = '{"bid": 1, "bid_size": 1, "ask": 99, "ask_size": 1}'

    def generate(messages):
        calls.append(messages)
        return [valid, "invalid"] if len(calls) == 1 else [valid]

    player._gen_many = generate
    answers = player.quote_many([first, second, first])
    assert [len(batch) for batch in calls] == [2, 1]
    assert calls[1][0][-1]["role"] == "user"
    assert all(quote is not None and not info["failed"] for quote, info in answers)
    assert [info["format_error"] for _, info in answers] == [False, True, False]
    assert all(not info["cached"] for _, info in answers)
    assert all(info["cached"] for _, info in player.quote_many([first, second]))
    assert len(calls) == 2


@pytest.mark.parametrize("mismatch", [False, True])
def test_gpu_batch_check_uses_training_states_and_keeps_failures(kernel, monkeypatch, tmp_path, mismatch):
    from makemarket.games.base import Quote

    seen = []

    class Quoter:
        def __init__(self, model, tokenizer, label):
            pass

        def quote(self, obs):
            seen.append((obs.game.name, obs.t))
            quote = Quote(1, 1, 99, 1)
            return quote, {"attempts": [{"content": json.dumps(quote.to_dict())}]}

        def quote_many(self, observations):
            answers = [self.quote(obs) for obs in observations]
            if mismatch:
                answers[0] = (None, answers[0][1])
            return answers

    monkeypatch.setattr(kernel, "WORK", tmp_path)
    monkeypatch.setattr(kernel, "HFQuoter", Quoter)
    args = Namespace(games=["dice", "cards"], label="test", seed=0, eval_batch_size=4)
    if mismatch:
        with pytest.raises(ValueError, match="Serial/batched quotes differ"):
            kernel.check_batching(None, None, args)
    else:
        kernel.check_batching(None, None, args)
    record = json.loads((tmp_path / "batch_parity.json").read_text())
    assert record["quotes_equal"] is not mismatch
    assert record["training_seeds"] == [20000, 20003]
    assert seen[:8] == [(game, t) for game in args.games for t in (0, 1, 3, 5)]


@pytest.mark.parametrize('warm_exists', [False, True])
def test_newest_continuation_preserves_sft_cadence_and_rejects_partial_checkpoints(
        monkeypatch, tmp_path, warm_exists):
    import hashlib

    spec = importlib.util.spec_from_file_location(
        'qwen38_continue', Path(__file__).parents[1] / 'train/qwen38_continue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    core = tmp_path / 'train_core.py'
    core.write_text('# frozen source\n')
    revision = '8aa5f05d26b7205477066e1449e0af13f762a299'
    (tmp_path / 'runtime.json').write_text(json.dumps({
        'config': {'revision': revision}, 'model_revision': revision, 'seed': 0,
        'status': 'failed', 'source_file_sha256': hashlib.sha256(core.read_bytes()).hexdigest(),
    }))
    complete = tmp_path / 'sft/checkpoint-750'
    complete.mkdir(parents=True)
    for name in ['adapter_model.safetensors', 'trainer_state.json', 'optimizer.pt',
                 'scheduler.pt', 'rng_state.pth']:
        (complete / name).touch()
    partial = tmp_path / 'sft/checkpoint-800'
    partial.mkdir()
    (partial / 'adapter_model.safetensors').touch()
    assert module.latest_checkpoint(tmp_path / 'sft') == complete
    warm = tmp_path / 'warmstart_adapter'
    if warm_exists:
        warm.mkdir()
        for name in ['adapter_model.safetensors', 'adapter_config.json']:
            (warm / name).touch()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        warm.mkdir(exist_ok=True)

    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr('sys.argv', ['qwen38_continue.py', '--core', str(core),
                                    '--source-commit', 'test-commit'])
    module.main()
    assert len(calls) == (1 if warm_exists else 2)
    if not warm_exists:
        assert calls[0][-6:] == ['--sft-resume', str(complete), '--grpo-steps', '0', '--save-steps', '50']
    assert calls[-1][-4:] == ['--grpo-steps', '32', '--save-steps', '5']
    record = json.loads((tmp_path / 'continuation.json').read_text())
    assert record['status'] == 'completed'
    assert record['grpo_total_ceiling'] == 32
