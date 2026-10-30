"""Paired validation contrasts for the registered local follow-up trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from makemarket.provenance import file_sha256, receipt
from makemarket.scoring import bootstrap_ci, episode_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("dagger", "dagger-last", "rloo", "rloo96", "rloo192"))
    parser.add_argument("--baseline", choices=("sft", "rloo", "rloo96"), default="sft")
    args = parser.parse_args()
    hashes, games = {}, {}
    for game in ("dice", "cards"):
        values = {}
        for method in (args.baseline, args.method):
            path = Path(f"results/validation/qwen35-{method}/raw/{game}/qwen3.5-4b-{method}.jsonl")
            rows = sorted((json.loads(s) for s in path.read_text().splitlines()), key=lambda r: r["seed"])
            assert [r["seed"] for r in rows] == list(range(80000, 80100))
            assert all(r["game"] == game and len(r["rounds"]) == 6 for r in rows)
            hashes[str(path)] = file_sha256(path)
            values[method] = np.array([episode_metrics(r)["epnl"] for r in rows])
        delta = values[args.method] - values[args.baseline]
        games[game] = {"mean_expected_pnl_change": float(delta.mean()),
                       "ci_95": bootstrap_ci(delta),
                       "family_ci": bootstrap_ci(delta, level=1 - 0.05 / 14, n_boot=56000)}
    manifests = {}
    for method in (args.baseline, args.method):
        path = Path(f"results/validation/qwen35-{method}/cohorts.json")
        manifests[method] = json.loads(path.read_text())
        hashes[str(path)] = file_sha256(path)
    record = {"provenance": receipt(vars(args), manifests[args.method]["provenance"]["model_revision"]),
              "source_sha256": file_sha256(__file__), "input_sha256": hashes, "games": games,
              "selection_scores": {m: r["selection_score"] for m, r in manifests.items()},
              "family": "Fourteen validation contrasts: DAgger-best, DAgger-last, RLOO24, RLOO96 and RLOO192 versus SFT; RLOO96 versus RLOO24 and RLOO192 versus RLOO96; each on dice/cards. Bonferroni level 1-0.05/14; 56000 resamples, seed 0. Budget extension registered after observing RLOO24 validation; earlier six-contrast receipts retained separately.",
              "interpretation": "Validation diagnostics conditional on the registered recipes and frozen weights. These seeds select the local winner; no held-out claim, fresh-cache variance or training-seed inference."}
    path = Path(f"results/validation/qwen35-{args.method}/vs_{args.baseline}.json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(games, indent=2))


if __name__ == "__main__":
    main()
