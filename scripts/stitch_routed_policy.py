"""Reuse frozen test cohorts for the validation-registered two-adapter policy."""

import argparse
import json
from pathlib import Path

from score_final import cohort

from makemarket.provenance import file_sha256, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, nargs="+", help="downloaded staged Kaggle test output roots")
    ap.add_argument("--out", default="results/final/staged-routed")
    ap.add_argument("--plan", default="results/training/staged_routing_plan.json")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.out)
    saved = out / "cohorts.json"
    if saved.exists():
        old = json.loads(saved.read_text())
        assert old["plan_sha256"] == file_sha256(args.plan)
        for path, expected in {**old["input_sha256"], **old["output_sha256"]}.items():
            assert file_sha256(path) == expected, path
        return
    record = {"provenance": receipt(vars(args), plan["provenance"]["model_revision"]),
              "source_sha256": file_sha256(__file__), "plan_sha256": file_sha256(args.plan),
              "input_sha256": {}, "output_sha256": {}, "routes": plan["routes"],
              "interpretation": "Reused deterministic cohorts, not new model calls; one base model and two game-specific LoRA adapters."}
    for game, underlying in plan["routes"].items():
        roots = [Path(root) / underlying / "results/raw" / game for root in args.input]
        matches = [root for root in roots if (root / f"{underlying}.jsonl").exists()]
        assert len(matches) == 1, f"Expected one source for {underlying}/{game}: {matches}"
        raw = matches[0]
        for label in (underlying, "bayes"):
            source = raw / f"{label}.jsonl"
            episodes = cohort(source)
            assert episodes is not None, f"Incomplete frozen test cohort: {source}"
            record["input_sha256"][str(source)] = file_sha256(source)
            target_label = plan["candidate_label"] if label == underlying else label
            assert all(e["player"] == label and e["game"] == game for e in episodes)
            if label == underlying:
                for episode in episodes:
                    provenance = episode["provenance"]
                    assert provenance["model_revision"] == plan["provenance"]["model_revision"]
                    assert provenance["config"]["adapter_sha256"] == plan["adapter_sha256"][underlying]
                    episode["player"] = target_label
                    provenance["routed_policy"] = {"label": target_label, "underlying_player": underlying,
                                                   "plan_sha256": record["plan_sha256"]}
            target = out / "raw" / game / f"{target_label}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(json.dumps(e) + "\n" for e in episodes))
            record["output_sha256"][str(target)] = file_sha256(target)
    saved.write_text(json.dumps(record, indent=2) + "\n")
    print("Complete routed cohorts retained; aggregate scoring is separate.")


if __name__ == "__main__":
    main()
