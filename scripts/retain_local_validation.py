"""Verify and retain a completed registered local validation cohort."""

import argparse
import json
from pathlib import Path

from makemarket.provenance import file_sha256, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("sft", "dagger", "dagger-last", "rloo", "rloo96", "rloo192", "rloo384"))
    args = parser.parse_args()
    root = Path(f"results/validation/qwen35-{args.method}")
    label = f"qwen3.5-4b-{args.method}"
    games, hashes, identities = {}, {}, set()
    for game in ("dice", "cards"):
        for player in (label, "bayes"):
            path = root / "raw" / game / f"{player}.jsonl"
            episodes = [json.loads(s) for s in path.read_text().splitlines()]
            assert sorted(e["seed"] for e in episodes) == list(range(80000, 80100))
            assert all(e["game"] == game and len(e["rounds"]) == 6 for e in episodes)
            hashes[str(path)] = file_sha256(path)
            if player == label:
                identities.update((e["provenance"]["model_revision"], e["provenance"]["adapter_sha256"]) for e in episodes)
        path = root / f"leaderboard_{game}.json"
        board = json.loads(path.read_text())
        assert board["seeds"] == list(range(80000, 80100))
        games[game] = next(row for row in board["rows"] if row["player"] == label)
        assert games[game]["episodes"] == 100
        hashes[str(path)] = file_sha256(path)
    assert len(identities) == 1
    revision, adapter = identities.pop()
    record = {"provenance": receipt(vars(args), revision), "source_sha256": file_sha256(__file__),
              "label": label, "adapter_sha256": adapter, "input_sha256": hashes, "games": games,
              "selection_score": min(games["dice"]["pct_of_bayes"] - 80, games["cards"]["pct_of_bayes"]),
              "interpretation": "Completed validation only; 95% descriptive paired-bootstrap intervals. Same registered max-min selection rule; no held-out scoring or training-seed replication claim."}
    (root / "cohorts.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"label": label, "selection_score": record["selection_score"], "percent_of_bayes": {g: r["pct_of_bayes"] for g, r in games.items()}}))


if __name__ == "__main__":
    main()
