"""Batched inference must preserve the sequential market and resumable cohorts."""

import json

import pytest

from makemarket.evaluate import load_results, run_batched_player, run_player
from makemarket.games import make_game
from makemarket.games.base import Quote


class HistoryQuoter:
    name = "history"

    def reset(self, seed):
        pass

    def quote(self, obs):
        if obs.t == 2:
            return None, {"failed": True}
        width = 1 + abs(obs.position) / 10
        return Quote(round(obs.m_pub - width, 1), 2, round(obs.m_pub + width, 1), 3), {}

    def quote_many(self, observations):
        return [self.quote(obs) for obs in observations]


@pytest.mark.parametrize("game_name", ["dice", "cards"])
def test_batches_preserve_every_fill_and_expectation_and_resume(tmp_path, game_name):
    game, seeds = make_game(game_name), list(range(80000, 80007))
    expected = run_player(game, HistoryQuoter, seeds, tmp_path / "serial.jsonl",
                          backend="py", provenance={"test": True})
    path = tmp_path / "batch.jsonl"
    actual = run_batched_player(game, HistoryQuoter(), seeds, path, 3, {"test": True})
    assert actual == expected
    before = path.read_bytes()
    player = HistoryQuoter()
    player.quote_many = lambda _: pytest.fail("Completed episodes must not be generated again")
    assert run_batched_player(game, player, seeds, path, 3, {"test": True}) == actual
    assert path.read_bytes() == before


def test_interrupted_group_is_atomic_and_only_missing_groups_resume(tmp_path):
    path, seeds = tmp_path / "raw.jsonl", list(range(80000, 80007))
    game, player = make_game("cards"), HistoryQuoter()
    calls = 0

    def interrupt(observations):
        nonlocal calls
        calls += 1
        if calls == 8:
            raise RuntimeError("interrupted")
        return [player.quote(obs) for obs in observations]

    player.quote_many = interrupt
    with pytest.raises(RuntimeError, match="interrupted"):
        run_batched_player(game, player, seeds, path, 3, {})
    assert [row["seed"] for row in load_results(path)] == seeds[:3]
    batches = []
    resumed = HistoryQuoter()
    resumed.quote_many = lambda obs: batches.append(len(obs)) or [resumed.quote(o) for o in obs]
    assert len(run_batched_player(game, resumed, seeds, path, 3, {})) == 7
    assert batches == [3] * 6 + [1] * 6


def test_partial_group_is_rejected(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"seed": 80000}) + "\n")
    with pytest.raises(ValueError, match="partial seed group"):
        run_batched_player(make_game("dice"), HistoryQuoter(), [80000, 80001], path, 2, {})
