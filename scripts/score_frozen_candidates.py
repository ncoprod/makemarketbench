"""Score registered complete candidates after freezing local and 27B selections."""

import json
import subprocess
from pathlib import Path

from makemarket.provenance import file_sha256, receipt
from makemarket.scoring import summarize

selection_path = Path('results/training/selection.json')
selection = json.loads(selection_path.read_text())
newest_path = Path('results/training/qwen38_selection.json')
newest = json.loads(newest_path.read_text())
family = newest['final_candidate_family']
assert newest['status'] == 'frozen' and family == (11 if newest['eval_batch_size'] == 4 else 9)
# Reporting replays committed episodes; private weight files are not required.
newest_games = []
for frozen, root in [(selection, 'results/final/selected'), (newest, newest['test_root'])]:
    for game in ('dice', 'cards'):
        path = Path(root) / f"raw/{game}/{frozen['label']}.jsonl"
        if frozen is newest and not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 200
        if frozen is newest:
            newest_games.append(game)
        for row in rows:
            provenance = row['provenance']
            adapter_hash = provenance.get('adapter_sha256', provenance['config'].get('adapter_sha256'))
            assert adapter_hash == frozen['adapter_sha256'], 'Recorded adapter differs from frozen choice'
            assert provenance['config']['revision'] == frozen['provenance']['model_revision']
assert newest_games, 'At least one complete newest-family game must be retained'
# Both validation decisions precede opening any new aggregate test score.
candidates = [
    ('qwen3-1.7b-sft800', 'results/final/qwen17-sft800'),
    (selection['label'], 'results/final/selected'),
    ('qwen3.5-4b-grpo-warmstart', 'results/kaggle/original-v1/results'),
    ('qwen3.5-4b-grpo', 'results/kaggle/original-resume-v2/qwen3.5-4b-grpo/results'),
    ('qwen3.5-4b-staged-grpo', 'results/kaggle/staged-final-v1/qwen3.5-4b-staged-grpo/results'),
    ('qwen3.5-4b-staged-routed', 'results/final/staged-routed'),
    (newest['label'], newest['test_root']),
]
record = {'provenance': receipt({'purpose': 'report every complete registered frozen candidate', 'candidate_family': family}),
          'source': Path(__file__).read_text(), 'selection_sha256': file_sha256(selection_path),
          'newest_selection_sha256': file_sha256(newest_path),
          'identity_scope': 'Recorded episode adapter hashes and model revisions match both frozen choices; numerical replay does not load weights.',
          'candidates_registered': family, 'commands': [], 'results': {}, 'unavailable': {
              'qwen3.5-4b-staged-sft': 'Dice 200 complete; cards 100 saved, other worker failed its development batching gate. Cards not scored.',
              'qwen3.8-27b-unselected': 'The other newest-family adapter remains validation-only under the registered selection rule; its trial still counts in the conservative family correction.'}}
if family == 11:
    record['unavailable'].pop('qwen3.8-27b-unselected')
    record['unavailable'].update({
        'qwen3.8-27b-sft': 'Original serial SFT trial retained on validation only; cards incomplete.',
        'qwen3.8-27b-grpo': 'Original serial GRPO trial retained on validation only.',
        'qwen3.8-27b-batch4-unselected': 'Unselected batch-four adapter remains validation-only; its trial counts.'})
for label, root in candidates:
    out = f'results/final/scores/{label}.json'
    command = ['.venv/bin/python', 'scripts/score_final.py', '--local', root, '--label', label,
               '--candidates', str(family), '--out', out]
    if label == newest['label']:
        command += ['--games', *newest_games]
    record['commands'].append(command)
    subprocess.run(command, check=True)
    record['results'][label] = {'file': out, 'sha256': file_sha256(out)}
if len(newest_games) < 2:
    record['unavailable'][newest['label'] + '-missing-games'] = {
        'complete_games_scored': newest_games,
        'scope': 'Only complete 200-episode games scored; missing-game targets unknown. Partial raw episodes remain in download receipts.'}

# Preserve the complete dice-only candidate without scoring its partial cards.
root = Path('results/kaggle/staged-sft-complete-v1/qwen3.5-4b-staged-sft/results/raw/dice')
players, hashes = {}, {}
for path in root.glob('*.jsonl'):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(r['seed'] for r in rows) == list(range(200, 400))
    players[rows[0]['player']] = rows
    hashes[str(path)] = file_sha256(path)
partial = {'provenance': receipt({'purpose': 'descriptive complete dice cohort only; partial cards deliberately unscored'},
                                 players['qwen3.5-4b-staged-sft'][0]['provenance']['model_revision']),
           'source': Path(__file__).read_text(), 'seeds': [200, 399], 'input_sha256': hashes,
           'game': 'dice', 'rows': summarize(players), 'cards_episodes_saved': 100,
           'limitation': 'Marginal descriptive intervals only; this partial candidate cannot satisfy both-game targets.'}
path = Path('results/final/scores/qwen3.5-4b-staged-sft-dice-only.json')
path.write_text(json.dumps(partial, indent=2) + '\n')
record['partial_result'] = {'file': str(path), 'sha256': file_sha256(path)}
Path('results/final/summary.json').write_text(json.dumps(record, indent=2) + '\n')
print('Scored frozen candidates with all available complete API cohorts; partial cards remain unscored.')
