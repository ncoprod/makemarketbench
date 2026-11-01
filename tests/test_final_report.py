import json
import sys

import pytest

from scripts import score_final


@pytest.mark.parametrize('fraction,games', [(1.0, ['dice', 'cards']), (0.79, ['dice', 'cards']), (1.0, ['dice'])])
def test_final_report_serializes_target_verdicts(monkeypatch, tmp_path, fraction, games):
    root = tmp_path / 'local'
    for game in games:
        folder = root / 'raw' / game
        folder.mkdir(parents=True)
        for player, pnl in [('bayes', 1.0), ('policy', fraction)]:
            rounds = [{'quote': {'bid': 0, 'ask': 2, 'bid_size': 1, 'ask_size': 1},
                       'exp_expected_pnl': pnl, 'exp_noise_edge': pnl, 'exp_adverse_cost': 0,
                       'position': 0, 'info': {}} for _ in range(6)]
            rows = [{'seed': seed, 'player': player, 'game': game, 'value': 1,
                     'rounds': rounds, 'realized_pnl': 6 * pnl,
                     'provenance': {'config': {'eval_batch_size': 1}}} for seed in range(200, 400)]
            (folder / f'{player}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    out = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['score_final.py', '--local', str(root), '--label', 'policy',
                                     '--api', str(tmp_path / 'no_api'), '--out', str(out), '--games', *games])
    score_final.main()
    result = json.loads(out.read_text())
    assert result['targets']['dice_mean_at_least_80_percent'] is (fraction >= .8)
    if 'cards' in games:
        assert all(type(value) is bool for value in result['targets'].values())
        assert result['targets']['at_least_four_complete_api_cohorts_per_game'] is False
    else:
        assert set(result['games']) == {'dice'}
        assert result['targets']['cards_positive_with_family_lower_above_zero'] is None
        assert result['targets']['at_least_four_complete_api_cohorts_per_game'] is None
