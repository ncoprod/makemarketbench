"""Block sampling preserves the original paired bootstrap exactly."""

import numpy as np
import pytest

from makemarket.scoring import bootstrap_ci, paired_ratio_ci


@pytest.mark.parametrize("stat", [np.mean, np.median])
def test_block_bootstrap_matches_single_allocation(stat):
    x = np.array([-20, -1, 0, 3, 100], dtype=float)
    idx = np.random.default_rng(7).integers(0, len(x), (2345, len(x)))
    expected = np.quantile([stat(x[i]) for i in idx], [0.025, 0.975])
    assert bootstrap_ci(x, stat=stat, n_boot=2345, seed=7) == pytest.approx(expected)


def test_block_ratio_bootstrap_keeps_seed_pairing():
    num, den = np.arange(1, 18), np.arange(1, 18) ** 2
    idx = np.random.default_rng(11).integers(0, len(num), (2345, len(num)))
    ratios = 100 * num[idx].mean(1) / den[idx].mean(1)
    expected = [100 * num.mean() / den.mean(), *np.quantile(ratios, [0.025, 0.975])]
    assert paired_ratio_ci(num, den, n_boot=2345, seed=11) == pytest.approx(expected)
