"""Regenerate the result tables in README.md from committed result files.

Tables live between ``<!-- TABLE:name -->`` and ``<!-- /TABLE:name -->`` markers,
so every number in them is copied programmatically from results/*.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt_ci(x, ci, nd=0):
    return f"{x:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def fmt_optional(x):
    return "n/a" if x is None else f"{x:.2f}"


def leaderboard(game: str) -> str:
    path = ROOT / "results" / f"leaderboard_{game}.json"
    rows = json.loads(path.read_text())["rows"]
    out = ["| Dealer | N | % of Bayes edge [95% CI] | E[PnL] / episode [95% CI] | Regret / round | "
           "Spread / optimal | Mid error | Adverse cost | Invalid answers |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['player']} | {r['episodes']} | {fmt_ci(r['pct_of_bayes'], r['pct_of_bayes_ci'])} | "
            f"{fmt_ci(r['epnl'], r['epnl_ci'], 1)} | {r['regret_per_round']:.2f} | "
            f"{fmt_optional(r['median_spread_ratio'])} | {fmt_optional(r['mid_error'])} | "
            f"{r['adverse_cost']:.1f} | {100 * r['format_error_rate']:.0f}% |")
    return "\n".join(out)


def engine() -> str:
    d = json.loads((ROOT / "results" / "engine_bench.json").read_text())
    names = {
        "cpp_native": "C++ engine, native loop (flat pool-based book)",
        "cpp_native_v1_map_list": "C++ v1 layout, native loop (std::map + std::list)",
        "cpp_pybind_batch": "C++ via pybind11, one call per stream (`replay`)",
        "cpp_pybind_per_call": "C++ via pybind11, one Python call per event",
        "python_per_call": "Pure-Python reference, one call per event",
        "python_batch": "Pure-Python reference, `replay` loop with checksum",
    }
    out = ["| Implementation | Events / s | Time for 1M events |", "|---|---|---|"]
    for k, label in names.items():
        if k in d["results"]:
            v = d["results"][k]
            out.append(f"| {label} | {v['events_per_sec'] / 1e6:.2f} M | {v['seconds'] * 1e3:.0f} ms |")
    return "\n".join(out)


def training() -> str:
    rows = []
    for game in ("dice", "cards"):
        path = ROOT / "results" / f"leaderboard_{game}.json"
        for r in json.loads(path.read_text())["rows"]:
            if r["player"].startswith("qwen3-"):
                rows.append({**r, "game": game})
    if not rows:
        return "_Pending._"
    order = {"base": 0, "sft": 1, "dagger": 2}
    rows.sort(key=lambda r: (r["game"] != "dice", order.get(r["player"].rsplit("-", 1)[-1], 9)))
    out = ["| Model | Game | N | % of Bayes edge [95% CI] | E[PnL] / episode [95% CI] | Regret / round | "
           "Invalid answers |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['player']} | {r['game']} | {r['episodes']} | "
                   f"{fmt_ci(r['pct_of_bayes'], r['pct_of_bayes_ci'])} | {fmt_ci(r['epnl'], r['epnl_ci'], 1)} | "
                   f"{r['regret_per_round']:.2f} | {100 * r['format_error_rate']:.0f}% |")
    return "\n".join(out)


def figgie() -> str:
    out = ["| Seat 0 | Seeds | N | Trading gain / table [95% CI] | Seat-0 trades / table | Invalid answers |",
           "|---|---|---|---|---|---|"]
    for seeds in ("0-199", "0-29"):
        path = ROOT / "results" / f"leaderboard_figgie_seeds{seeds}.json"
        if not path.exists():
            continue
        lo, hi = (int(x) for x in seeds.split("-"))
        for r in json.loads(path.read_text())["rows"]:
            if r["episodes"] != hi - lo + 1:  # only players run on every seed of this block
                continue
            out.append(f"| {r['player']} | {seeds} | {r['episodes']} | "
                       f"{fmt_ci(r['trading_gain'], r['trading_gain_ci'], 2)} | {r['seat0_trades']:.1f} | "
                       f"{100 * r['format_error_rate']:.0f}% |")
    return "\n".join(out)


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    tables = {"dice": lambda: leaderboard("dice"), "cards": lambda: leaderboard("cards"),
              "engine": engine, "training": training,
              "figgie": figgie}
    for name, fn in tables.items():
        pat = re.compile(rf"(<!-- TABLE:{name} -->\n)(?:.*?\n)?(<!-- /TABLE:{name} -->)", re.DOTALL)
        if pat.search(text):
            text = pat.sub(lambda m, fn=fn: m.group(1) + fn() + "\n" + m.group(2), text)
    readme.write_text(text)
    print("README tables updated")


if __name__ == "__main__":
    main()
