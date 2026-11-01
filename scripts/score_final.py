"""Score the frozen policy on the registered fresh test, retaining every cohort."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from makemarket.evaluate import load_results
from makemarket.provenance import file_sha256, receipt
from makemarket.scoring import bootstrap_ci, episode_metrics, paired_ratio_ci, summarize

SEEDS = list(range(200, 400))


def cohort(path):
    episodes = load_results(path)
    seeds = [e["seed"] for e in episodes]
    if sorted(seeds) != SEEDS or any(len(e["rounds"]) != 6 for e in episodes):
        return None
    return sorted(episodes, key=lambda e: e["seed"])


def epnl(episodes):
    return np.array([episode_metrics(e)["epnl"] for e in episodes])


def group_means(values, size):
    return np.asarray(values).reshape(-1, size).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", required=True, help="frozen MLX evaluation directory")
    ap.add_argument("--label", required=True)
    ap.add_argument("--api", default="results/expansion")
    ap.add_argument("--out", default="results/final.json")
    ap.add_argument("--games", nargs="+", choices=["dice", "cards"], default=["dice", "cards"],
                    help="score only explicitly complete games; missing-game targets stay unknown")
    ap.add_argument("--candidates", type=int, default=11,
                    help="registered final candidates, including the two-adapter game router")
    args = ap.parse_args()
    assert args.candidates >= 1
    comparisons = sum(cohort(p) is not None for p in (Path(args.api) / "raw/dice").glob("*.jsonl"))
    level = 1 - 0.05 / (args.candidates * (comparisons + 2))
    # Keep at least 100 bootstrap draws in each corrected tail, without a large
    # episodes-by-resamples allocation (the scorer samples in small blocks).
    family_resamples = max(10_000, math.ceil(200 / (1 - level)))
    record = {"provenance": receipt(vars(args)), "source_sha256": file_sha256(__file__),
              "seeds": SEEDS, "games": {}, "input_sha256": {}, "incomplete_cohorts": []}
    for game in args.games:
        raw = Path(args.local) / "raw" / game
        paths = list(raw.glob("*.jsonl")) + list((Path(args.api) / "raw" / game).glob("*.jsonl"))
        players, api_names = {}, []
        for path in sorted(paths):
            episodes = cohort(path)
            record["input_sha256"][str(path)] = file_sha256(path)
            if episodes is None:
                record["incomplete_cohorts"].append(str(path))
                continue
            name = episodes[0]["player"]
            assert name not in players, f"duplicate player cohort: {name}"
            players[name] = episodes
            if path.parent == Path(args.api) / "raw" / game:
                api_names.append(name)
        assert args.label in players and "bayes" in players, f"missing complete {game} policy/reference"
        source_provenance = players[args.label][0]["provenance"]
        record["provenance"]["model_revision"] = source_provenance.get(
            "model_revision", source_provenance["config"].get("revision"))
        local, bayes = epnl(players[args.label]), epnl(players["bayes"])
        sizes = {e["provenance"]["config"].get("eval_batch_size", 1) for e in players[args.label]}
        assert len(sizes) == 1, "Mixed inference group sizes require explicit grouping"
        size = sizes.pop()
        assert size > 0 and len(local) % size == 0
        # K paired differences, one dice ratio, and one cards positivity claim.
        marginal = {r["player"]: r for r in summarize(players)}
        differences = {}
        for name in api_names if game == "dice" else []:
            delta = local - epnl(players[name])
            interval = bootstrap_ci(delta, level=level, n_boot=family_resamples)
            differences[name] = {"mean": float(delta.mean()), "family_ci": interval,
                                 "inference_group_family_ci": interval if size == 1 else bootstrap_ci(
                                     group_means(delta, size), level=level, n_boot=family_resamples)}
        local_interval = bootstrap_ci(local, level=level, n_boot=family_resamples)
        ratio_interval = paired_ratio_ci(local, bayes, level=level, n_boot=family_resamples)[1:]
        record["games"][game] = {
            "rows": list(marginal.values()), "api_cohorts": api_names, "family_level": level,
            "family_bootstrap_resamples": family_resamples,
            "local_epnl_family_ci": local_interval,
            "local_percent_family_ci": ratio_interval,
            "inference_group_size": size,
            "inference_group_count": len(local) // size,
            "local_epnl_inference_group_family_ci": local_interval if size == 1 else bootstrap_ci(
                group_means(local, size), level=level, n_boot=family_resamples),
            "local_percent_inference_group_family_ci": ratio_interval if size == 1 else paired_ratio_ci(
                group_means(local, size), group_means(bayes, size), level=level,
                n_boot=family_resamples)[1:],
            "paired_api_differences": differences,
            "lower_95_above_every_api_upper_95": bool(api_names) and all(
                marginal[args.label]["epnl_ci"][0] > marginal[n]["epnl_ci"][1] for n in api_names),
            "lower_percent_95_above_every_api_upper_percent_95": bool(api_names) and all(
                marginal[args.label]["pct_of_bayes_ci"][0] > marginal[n]["pct_of_bayes_ci"][1]
                for n in api_names),
            "all_paired_family_lower_bounds_positive": bool(api_names) and all(
                d["family_ci"][0] > 0 and d["inference_group_family_ci"][0] > 0
                for d in differences.values())}
    dice, cards = (record["games"].get(g) for g in ("dice", "cards"))
    local_dice = next(r for r in dice["rows"] if r["player"] == args.label) if dice else None
    record["targets"] = {
        "dice_mean_at_least_80_percent": bool(local_dice["pct_of_bayes"] >= 80) if dice else None,
        "dice_above_api_intervals": dice["lower_95_above_every_api_upper_95"] and
                                    dice["lower_percent_95_above_every_api_upper_percent_95"] and
                                    dice["all_paired_family_lower_bounds_positive"] if dice else None,
        "cards_positive_with_family_lower_above_zero": cards["local_epnl_family_ci"][0] > 0 and
            cards["local_epnl_inference_group_family_ci"][0] > 0 if cards else None,
        "at_least_four_complete_api_cohorts_per_game": all(
            len(g["api_cohorts"]) >= 4 for g in (dice, cards)) if dice and cards else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["targets"], indent=2))


if __name__ == "__main__":
    main()
