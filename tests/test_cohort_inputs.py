import sys

import pytest

from scripts.check_cohort_games import main


def test_missing_explicit_input_cannot_silently_reduce_verified_cohorts(monkeypatch, tmp_path):
    missing = tmp_path / "misspelled-cohort.jsonl"
    monkeypatch.setattr(sys, "argv", ["check_cohort_games.py", str(tmp_path), str(missing)])
    with pytest.raises(AssertionError, match="Missing cohort input"):
        main()
