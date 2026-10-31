"""Audit cards positivity with ten times the registered bootstrap draws."""

import json
import time
from pathlib import Path

import numpy as np

from makemarket.evaluate import load_results
from makemarket.provenance import file_sha256, receipt
from makemarket.scoring import bootstrap_ci, episode_metrics

started = time.monotonic()
manifest_path = Path('results/final/summary.json')
manifest = json.loads(manifest_path.read_text())
record = {'provenance': receipt({'purpose': 'fixed all-candidate Monte Carlo precision audit',
                                 'draw_multiplier': 10, 'bootstrap_seed': 0}),
          'source_sha256': file_sha256(__file__), 'manifest_sha256': file_sha256(manifest_path),
          'candidates': {}, 'scope': 'Numerical sensitivity only; retain the registered primary intervals and verdicts. '
                                    'Do not upgrade a failed primary claim on this audit.'}
for label, entry in manifest['results'].items():
    path = Path(entry['file'])
    assert file_sha256(path) == entry['sha256']
    data = json.loads(path.read_text())
    raw = Path(data['provenance']['config']['local']) / f'raw/cards/{label}.jsonl'
    assert file_sha256(raw) == data['input_sha256'][str(raw)]
    episodes = sorted(load_results(raw), key=lambda row: row['seed'])
    assert [row['seed'] for row in episodes] == list(range(200, 400))
    assert all(len(row['rounds']) == 6 for row in episodes)
    values = np.array([episode_metrics(row)['epnl'] for row in episodes])
    primary = data['games']['cards']
    draws = primary['family_bootstrap_resamples'] * 10
    interval = bootstrap_ci(values, n_boot=draws, level=primary['family_level'], seed=0)
    size = primary['inference_group_size']
    grouped = interval if size == 1 else bootstrap_ci(values.reshape(-1, size).mean(axis=1), n_boot=draws,
                                                     level=primary['family_level'], seed=0)
    positive = interval[0] > 0 and grouped[0] > 0
    record['candidates'][label] = {'score_file': str(path), 'score_sha256': file_sha256(path),
        'raw_file': str(raw), 'raw_sha256': file_sha256(raw), 'draws': draws,
        'model_revision': episodes[0]['provenance']['config'].get('revision'),
        'family_level': primary['family_level'], 'inference_group_size': size,
        'primary_cards_positive': data['targets']['cards_positive_with_family_lower_above_zero'],
        'higher_precision_epnl_ci': interval, 'higher_precision_group_epnl_ci': grouped,
        'higher_precision_cards_positive': bool(positive),
        'positive_under_both': bool(positive and data['targets']['cards_positive_with_family_lower_above_zero'])}
record['wall_seconds'] = time.monotonic() - started
Path('results/final/bootstrap_precision.json').write_text(json.dumps(record, indent=2) + '\n')
print('Audited cards bootstrap precision for', len(record['candidates']), 'fixed policies')
