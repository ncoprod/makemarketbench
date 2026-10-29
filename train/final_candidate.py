"""Freeze the preregistered validation winner, then evaluate its fresh test once."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from makemarket.provenance import file_sha256, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()
    frozen = Path("results/training/selection.json")
    if frozen.exists():
        selection = json.loads(frozen.read_text())
    else:
        trials = []
        for method in ("sft", "dagger", "rloo"):
            label = f"qwen3.5-4b-{method}"
            scores, inputs = {}, {}
            for game in ("dice", "cards"):
                path = Path(f"results/validation/qwen35-{method}/leaderboard_{game}.json")
                data = json.loads(path.read_text())
                assert data["seeds"] == list(range(80000, 80100)), "wrong validation split"
                row = next(r for r in data["rows"] if r["player"] == label)
                assert row["episodes"] == 100, "incomplete validation cohort"
                scores[game] = row
                inputs[str(path)] = file_sha256(path)
            trials.append({"label": label, "adapter": f"train/runs/{label}", "scores": scores,
                           "input_sha256": inputs,
                           "selection_score": min(scores["dice"]["pct_of_bayes"] - 80,
                                                  scores["cards"]["pct_of_bayes"])})
        winner = max(trials, key=lambda t: t["selection_score"])
        selection = {"provenance": receipt(vars(args), args.revision), "trials": trials,
                     "rule": "max min(dice percentage - 80, cards percentage); earlier trial wins ties",
                     "label": winner["label"], "adapter": winner["adapter"],
                     "adapter_sha256": file_sha256(Path(winner["adapter"]) / "adapters.safetensors")}
        frozen.write_text(json.dumps(selection, indent=2) + "\n")
    assert selection["provenance"]["model_revision"] == args.revision
    assert file_sha256(Path(selection["adapter"]) / "adapters.safetensors") == selection["adapter_sha256"]
    command = [sys.executable, "train/mlx_evaluate.py", "--model", args.model,
               "--revision", args.revision, "--adapter", selection["adapter"],
               "--label", selection["label"], "--seed", "200", "--episodes", "200",
               "--out", "results/final/selected"]
    print("Frozen validation winner:", selection["label"], flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
