"""Score registered complete candidates after freezing local and 27B selections."""

import json
import subprocess
from pathlib import Path

from makemarket.provenance import file_sha256, receipt
from makemarket.scoring import summarize

selection_path = Path('results/training/selection.json')
selection = json.loads(selection_path.read_text())
assert file_sha256(Path(selection['adapter']) / 'adapters.safetensors') == selection['adapter_sha256']
newest_path = Path('results/training/qwen38_selection.json')
newest = json.loads(newest_path.read_text())
assert newest['status'] == 'frozen' and newest['final_candidate_family'] == 9
assert file_sha256(newest['adapter_file']) == newest['adapter_sha256']
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
record = {'provenance': receipt({'purpose': 'report every complete registered frozen candidate with the nine-candidate correction'}),
          'source': Path(__file__).read_text(), 'selection_sha256': file_sha256(selection_path),
          'newest_selection_sha256': file_sha256(newest_path),
          'candidates_registered': 9, 'commands': [], 'results': {}, 'unavailable': {
              'qwen3.5-4b-staged-sft': 'Dice 200 complete; cards 100 saved, other worker failed its development batching gate. Cards not scored.',
              'qwen3.8-27b-unselected': 'The other newest-family adapter remains validation-only under the registered selection rule; its trial still counts in the conservative family correction.'}}
for label, root in candidates:
    out = f'results/final/scores/{label}.json'
    command = ['.venv/bin/python', 'scripts/score_final.py', '--local', root, '--label', label,
               '--candidates', '9', '--out', out]
    record['commands'].append(command)
    subprocess.run(command, check=True)
    record['results'][label] = {'file': out, 'sha256': file_sha256(out)}

# Preserve the complete dice-only candidate without scoring its partial cards.
root = Path('results/kaggle/staged-sft-complete-v1/qwen3.5-4b-staged-sft/results/raw/dice')
players, hashes = {}, {}
for path in root.glob('*.jsonl'):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(r['seed'] for r in rows) == list(range(200, 400))
    players[rows[0]['player']] = rows
    hashes[str(path)] = file_sha256(path)
partial = {'provenance': receipt({'purpose': 'descriptive complete dice cohort only; partial cards deliberately unscored'}),
           'source': Path(__file__).read_text(), 'seeds': [200, 399], 'input_sha256': hashes,
           'game': 'dice', 'rows': summarize(players), 'cards_episodes_saved': 100,
           'limitation': 'Marginal descriptive intervals only; this partial candidate cannot satisfy both-game targets.'}
path = Path('results/final/scores/qwen3.5-4b-staged-sft-dice-only.json')
path.write_text(json.dumps(partial, indent=2) + '\n')
record['partial_result'] = {'file': str(path), 'sha256': file_sha256(path)}
Path('results/final/summary.json').write_text(json.dumps(record, indent=2) + '\n')
print('Scored frozen candidates with all available complete API cohorts; partial cards remain unscored.')
