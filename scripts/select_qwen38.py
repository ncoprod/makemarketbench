"""Freeze the registered SFT/GRPO validation winner before its one held-out test."""

import argparse
import json
from pathlib import Path

from makemarket.provenance import file_sha256, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training-root', required=True, type=Path)
    ap.add_argument('--validation-root', required=True, type=Path)
    ap.add_argument('--test-root', required=True, type=Path)
    ap.add_argument('--batch-size', type=int, choices=[1, 4], default=1)
    ap.add_argument('--certify-incomplete-loser', action='store_true',
                    help='allow missing GRPO cards only when its complete dice score proves it cannot win')
    ap.add_argument('--out', type=Path, default=Path('results/training/qwen38_selection.json'))
    args = ap.parse_args()
    assert not args.out.exists(), 'Selection is already frozen; do not overwrite it'
    trials = []
    revision = '8aa5f05d26b7205477066e1449e0af13f762a299'
    runtime_path = args.training_root / 'runtime.json'
    runtime = json.loads(runtime_path.read_text())
    assert runtime['seed'] == 0, 'Only the registered primary training seed may enter test selection'
    assert runtime['model_revision'] == revision, 'Wrong training model revision'
    for method, directory in [('sft', 'warmstart_adapter'), ('grpo', 'lora_adapter')]:
        family = 'qwen3.8-27b-batch4' if args.batch_size == 4 else 'qwen3.8-27b'
        label = f'{family}-{method}'
        adapter = args.training_root / directory / 'adapter_model.safetensors'
        adapter_hash = file_sha256(adapter)
        scores, inputs = {}, {}
        for game in ('dice', 'cards'):
            path = args.validation_root / label / 'results' / f'leaderboard_{game}.json'
            if not path.exists() and args.certify_incomplete_loser and method == 'grpo' and game == 'cards':
                continue
            data = json.loads(path.read_text())
            assert data['seeds'] == list(range(80000, 80100)), 'Wrong validation split'
            cfg = data['provenance']['config']
            assert cfg['revision'] == revision and cfg['adapter_sha256'] == adapter_hash
            assert cfg['eval_batch_size'] == args.batch_size, 'Wrong registered inference variant'
            assert cfg['seed'] == 0, 'Wrong validation inference seed'
            row = next(r for r in data['rows'] if r['player'] == label)
            assert row['episodes'] == 100, 'Incomplete validation cohort'
            scores[game] = row
            inputs[str(path)] = file_sha256(path)
        trials.append({'label': label, 'method': method, 'adapter_file': str(adapter),
                       'adapter_sha256': adapter_hash, 'scores': scores, 'input_sha256': inputs,
                       'selection_score': min(scores['dice']['pct_of_bayes'] - 80,
                                              scores['cards']['pct_of_bayes']) if 'cards' in scores else None})
    winner = max((t for t in trials if t['selection_score'] is not None), key=lambda t: t['selection_score'])
    for trial in trials:
        if trial['selection_score'] is None:
            upper = trial['scores']['dice']['pct_of_bayes'] - 80
            assert winner['method'] == 'sft' and upper <= winner['selection_score'], 'Incomplete candidate could still win'
            trial['elimination_certificate'] = {
                'selection_score_upper_bound': upper, 'winning_score': winner['selection_score'],
                'proof': 'min(dice_percent - 80, cards_percent) <= dice_percent - 80. SFT also wins a tie.',
                'scope': 'Exact selection-rule consequence, not statistical superiority. Partial cards are unscored.'}
    record = {'provenance': receipt({k: str(v) for k, v in vars(args).items()}, revision),
              'status': 'frozen', 'seed': 0, 'trials': trials,
              'training_runtime': str(runtime_path), 'training_runtime_sha256': file_sha256(runtime_path),
              'rule': 'max min(dice percentage - 80, cards percentage); SFT wins ties',
              **{k: winner[k] for k in ('label', 'method', 'adapter_file', 'adapter_sha256')},
              'test_root': str(args.test_root / winner['label'] / 'results'),
              'test_seeds': [200, 399], 'test_episodes_per_game': 200,
              'eval_batch_size': args.batch_size,
              'final_candidate_family': 11 if args.batch_size == 4 else 9}
    if any(t['selection_score'] is None for t in trials):
        record['completion_gate_deviation'] = (
            'Original acquisition plan required 100/game for both adapters. Stop the incomplete GRPO cards '
            'because its complete dice result mathematically precludes winning the unchanged max-min rule. '
            'This certificate is recorded before any held-out aggregate; both trials still count.')
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print('Frozen newest-family validation winner:', winner['label'])


if __name__ == '__main__':
    main()
