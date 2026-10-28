"""Episode metrics, paired comparisons against the Bayes reference, bootstrap CIs."""

from __future__ import annotations

import numpy as np

N_BOOT = 10_000


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = N_BOOT, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    stats = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def paired_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = N_BOOT, seed: int = 0,
                    level: float = 0.95) -> tuple[float, float, float]:
    """100 * mean(num) / mean(den) with a paired (same-seed) bootstrap CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    r = 100 * num[idx].mean(1) / den[idx].mean(1)
    a = (1 - level) / 2
    return 100 * num.mean() / den.mean(), float(np.quantile(r, a)), float(np.quantile(r, 1 - a))


def episode_metrics(ep: dict) -> dict:
    v = ep["value"]
    rs = ep["rounds"]
    valid = [r for r in rs if r["quote"] is not None]
    two_sided = [r for r in valid if r["quote"]["bid_size"] > 0 and r["quote"]["ask_size"] > 0]
    fmt = [bool(r.get("info", {}).get("format_error", False)) for r in rs]
    failed = [r["quote"] is None for r in rs]
    return {
        "seed": ep["seed"],
        "epnl": sum(r["exp_expected_pnl"] for r in rs),
        "noise_edge": sum(r["exp_noise_edge"] for r in rs),
        "adverse_cost": sum(r["exp_adverse_cost"] for r in rs),
        "realized_pnl": ep["realized_pnl"],
        "spread": float(np.mean([r["quote"]["ask"] - r["quote"]["bid"] for r in two_sided])) if two_sided else np.nan,
        "coverage": float(np.mean([r["quote"]["bid"] <= v <= r["quote"]["ask"] for r in two_sided])) if two_sided else np.nan,
        "mean_size": float(np.mean([(r["quote"]["bid_size"] + r["quote"]["ask_size"]) / 2 for r in valid])) if valid else 0.0,
        "abs_final_position": abs(rs[-1]["position"]),
        "format_error_rate": float(np.mean(fmt)),
        "fail_rate": float(np.mean(failed)),
    }


def summarize(player_eps: dict[str, list[dict]], reference: str = "bayes") -> list[dict]:
    """Aggregate per-player metrics on the seeds shared with the reference."""
    ref = {e["seed"]: episode_metrics(e) for e in player_eps[reference]}
    rows = []
    for name, eps in player_eps.items():
        ms = [episode_metrics(e) for e in eps if e["seed"] in ref]
        if not ms:
            continue
        ms.sort(key=lambda m: m["seed"])
        seeds = [m["seed"] for m in ms]
        col = {k: np.array([m[k] for m in ms], dtype=float) for k in ms[0] if k != "seed"}
        ref_epnl = np.array([ref[s]["epnl"] for s in seeds])
        pct, plo, phi = paired_ratio_ci(col["epnl"], ref_epnl)
        lo, hi = bootstrap_ci(col["epnl"])
        rlo, rhi = bootstrap_ci(col["realized_pnl"])
        rows.append({
            "player": name,
            "episodes": len(ms),
            "pct_of_bayes": pct, "pct_of_bayes_ci": [plo, phi],
            "epnl": float(col["epnl"].mean()), "epnl_ci": [lo, hi],
            "realized_pnl": float(col["realized_pnl"].mean()), "realized_pnl_ci": [rlo, rhi],
            "realized_pnl_sd": float(col["realized_pnl"].std(ddof=1)) if len(ms) > 1 else 0.0,
            "noise_edge": float(col["noise_edge"].mean()),
            "adverse_cost": float(col["adverse_cost"].mean()),
            "spread": float(np.nanmean(col["spread"])) if np.isfinite(col["spread"]).any() else None,
            "coverage": float(np.nanmean(col["coverage"])) if np.isfinite(col["coverage"]).any() else None,
            "mean_size": float(col["mean_size"].mean()),
            "abs_final_position": float(col["abs_final_position"].mean()),
            "format_error_rate": float(col["format_error_rate"].mean()),
            "fail_rate": float(col["fail_rate"].mean()),
        })
    rows.sort(key=lambda r: -r["pct_of_bayes"])
    return rows


def markdown_table(rows: list[dict]) -> str:
    head = ("| Player | N | % of Bayes [95% CI] | E[PnL] [95% CI] | Realized PnL | Noise edge | "
            "Adverse cost | Spread | Spread / optimal | Mid error | V in quote | Size | abs(final pos) | "
            "Format err | No quote |")
    sep = "|" + "---|" * 15
    lines = [head, sep]
    for r in rows:
        def f(x, nd=1):
            return "n/a" if x is None else f"{x:.{nd}f}"
        cov = "n/a" if r["coverage"] is None else f"{100 * r['coverage']:.0f}%"
        lines.append(
            f"| {r['player']} | {r['episodes']} | {r['pct_of_bayes']:.0f} "
            f"[{r['pct_of_bayes_ci'][0]:.0f}, {r['pct_of_bayes_ci'][1]:.0f}] | "
            f"{r['epnl']:.1f} [{r['epnl_ci'][0]:.1f}, {r['epnl_ci'][1]:.1f}] | {r['realized_pnl']:.1f} | "
            f"{r['noise_edge']:.1f} | {r['adverse_cost']:.1f} | {f(r['spread'])} | "
            f"{f(r.get('median_spread_ratio'), 2)} | {f(r.get('mid_error'), 2)} | "
            f"{cov} | {r['mean_size']:.1f} | "
            f"{r['abs_final_position']:.1f} | {100 * r['format_error_rate']:.0f}% | {100 * r['fail_rate']:.0f}% |"
        )
    return "\n".join(lines)
