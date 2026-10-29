"""Check the saved engine comparison using circular block bootstraps.

Keep all registered block lengths and both execution orders. No new benchmark
is run, and no samples are removed.
"""

import json
from pathlib import Path

import numpy as np

from makemarket.provenance import file_sha256, receipt


def main():
    source = Path("results/engine_trials/inline_matching_confirmation.json")
    data = json.loads(source.read_text())
    samples = data["samples"]
    pairs = sorted({s["pair"] for s in samples})
    before, after = [], []
    for pair in pairs:
        rows = [s for s in samples if s["pair"] == pair]
        assert len(rows) == 2 and {s["old"] for s in rows} == {True, False}
        assert all(s["checksum"] == data["checksum"] for s in rows)
        before.append(next(s["wall_seconds"] for s in rows if s["old"]))
        after.append(next(s["wall_seconds"] for s in rows if not s["old"]))
    ratios = np.asarray(before) / np.asarray(after)
    lengths, draws = [1, 5, 10, 20], 50_000
    tail = 0.05 / (2 * 3 * len(lengths))
    record = {"provenance": receipt({"seed": 0, "block_lengths": lengths,
                                     "resamples": draws, "engine_candidates": 3}),
              "source_sha256": file_sha256(__file__), "input": str(source),
              "input_sha256": file_sha256(source), "pairs": len(pairs),
              "checksum": data["checksum"], "median_speedup": float(np.median(ratios)),
              "lag_one_ratio_correlation": float(np.corrcoef(ratios[:-1], ratios[1:])[0, 1]),
              "order_medians": {"old_first": float(np.median(ratios[::2])),
                                "new_first": float(np.median(ratios[1::2]))},
              "confidence": 1 - 2 * tail, "block_bootstrap": {}}
    for length in lengths:
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(draws // 1000):
            starts = rng.integers(0, len(ratios), (1000, (len(ratios) + length - 1) // length))
            indices = (starts[..., None] + np.arange(length)) % len(ratios)
            indices = indices.reshape(1000, -1)[:, :len(ratios)]
            boot.extend(np.median(ratios[indices], axis=1))
        record["block_bootstrap"][str(length)] = np.quantile(boot, [tail, 1 - tail]).tolist()
    record["all_corrected_lower_bounds_above_one"] = all(
        interval[0] > 1 for interval in record["block_bootstrap"].values())
    dest = source.with_name("inline_matching_block_intervals.json")
    dest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["block_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
