"""Figures for the README, generated from committed result files only.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from makemarket.evaluate import load_results  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LLM, TRAINED, BASE = "#2a78d6", "#eb6834", "#a3a29c"
BASELINES = {"bayes", "bayes-public", "mean-sd", "fixed-prior", "random"}


def kind(player: str) -> str:
    if player in BASELINES:
        return "baseline"
    if any(k in player for k in ("sft", "dagger", "grpo")):
        return "trained"
    return "llm"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def leaderboard_figure(game: str) -> None:
    path = ROOT / "results" / f"leaderboard_{game}.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    rows = sorted(rows, key=lambda r: r["pct_of_bayes"])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2), facecolor=SURFACE)
    y = np.arange(len(rows))
    colors = {"baseline": BASE, "llm": LLM, "trained": TRAINED}
    for i, r in enumerate(rows):
        c = colors[kind(r["player"])]
        lo, hi = r["pct_of_bayes_ci"]
        ax.barh(i, r["pct_of_bayes"], height=0.55, color=c)
        if r["player"] != "bayes":
            ax.plot([lo, hi], [i, i], color=INK, linewidth=1.2)
        ax.text(max(r["pct_of_bayes"], hi if r["player"] != "bayes" else 0, 0) + 3, i,
                f"{r['pct_of_bayes']:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, [f"{r['player']} (n={r['episodes']})" for r in rows], color=INK)
    ax.set_xlabel("Expected PnL as % of the Bayes reference (paired bootstrap 95% CI)", color=MUTED, fontsize=9)
    ax.set_title(f"{game}: edge captured relative to the Bayes reference", color=INK, fontsize=11, loc="left")
    style(ax)
    from matplotlib.patches import Patch

    present = {kind(r["player"]) for r in rows}
    names = {"baseline": "reference / baselines", "llm": "API LLMs", "trained": "fine-tuned local"}
    ax.legend(handles=[Patch(color=colors[k], label=names[k]) for k in ("baseline", "llm", "trained") if k in present],
              frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"leaderboard_{game}.png", dpi=160)
    plt.close(fig)


def decomposition_figure(game: str) -> None:
    """Noise edge earned vs adverse-selection cost paid, per player."""
    path = ROOT / "results" / f"leaderboard_{game}.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor=SURFACE)
    colors = {"baseline": BASE, "llm": LLM, "trained": TRAINED}
    for r in rows:
        ax.scatter(r["adverse_cost"], r["noise_edge"], s=46, color=colors[kind(r["player"])],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(r["player"], (r["adverse_cost"], r["noise_edge"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5, color=INK)
    m = max(max(r["adverse_cost"] for r in rows), max(r["noise_edge"] for r in rows)) * 1.1
    ax.plot([0, m], [0, m], color=MUTED, linestyle="--", linewidth=0.8)
    ax.text(m * 0.62, m * 0.55, "break-even", color=MUTED, fontsize=8, rotation=35)
    ax.set_xlabel("Adverse-selection cost per episode (expected loss to informed traders)", color=MUTED, fontsize=9)
    ax.set_ylabel("Noise edge per episode", color=MUTED, fontsize=9)
    ax.set_title(f"{game}: where the PnL comes from", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG / f"decomposition_{game}.png", dpi=160)
    plt.close(fig)


def spread_by_round_figure(game: str, players: list[str]) -> None:
    """Mean quoted spread per round vs the reference: do models tighten as information arrives?"""
    raw = ROOT / "results" / "raw" / game
    series = {}
    for f in sorted(raw.glob("*.jsonl")):
        eps = [e for e in load_results(f) if e["seed"] < 50]
        if not eps or eps[0]["player"] not in players:
            continue
        n = len(eps[0]["rounds"])
        sp = [np.mean([e["rounds"][t]["quote"]["ask"] - e["rounds"][t]["quote"]["bid"]
                       for e in eps if e["rounds"][t]["quote"]]) for t in range(n)]
        series[eps[0]["player"]] = sp
    if not series:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0), facecolor=SURFACE)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300", "#eda100", "#e34948"]
    i = 0
    for name, sp in series.items():
        if name == "bayes":
            color, lw = INK, 2.4
        else:
            color, lw = palette[i % len(palette)], 1.8
            i += 1
        x = np.arange(1, len(sp) + 1)
        ax.plot(x, sp, color=color, linewidth=lw, marker="o", markersize=4, label=name)
    ax.set_xlabel("Round (more information revealed each round)", color=MUTED, fontsize=9)
    ax.set_ylabel("Mean quoted spread (ask - bid)", color=MUTED, fontsize=9)
    ax.set_title(f"{game}: spread by round", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / f"spread_by_round_{game}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for g in ("dice", "cards"):
        leaderboard_figure(g)
        decomposition_figure(g)
        path = ROOT / "results" / f"leaderboard_{g}.json"
        if path.exists():
            rows = json.loads(path.read_text())["rows"]
            names = ["bayes"] + [r["player"] for r in rows if kind(r["player"]) != "baseline"]
            spread_by_round_figure(g, names)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
