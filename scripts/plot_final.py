"""Plot complete frozen cohorts and mark missing games from verified scores."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from makemarket.provenance import file_sha256, receipt

manifest_path = Path('results/final/summary.json')
manifest = json.loads(manifest_path.read_text())
rows = {'dice': {}, 'cards': {}}
inputs = {str(manifest_path): file_sha256(manifest_path)}
for entry in manifest['results'].values():
    path = Path(entry['file'])
    assert file_sha256(path) == entry['sha256']
    inputs[str(path)] = file_sha256(path)
    data = json.loads(path.read_text())
    for game, result in data['games'].items():
        for row in result['rows']:
            if row['player'] in rows[game]:
                assert row == rows[game][row['player']], 'inconsistent duplicate cohort scores'
            rows[game][row['player']] = row
entry = manifest['partial_result']
path = Path(entry['file'])
assert file_sha256(path) == entry['sha256']
inputs[str(path)] = file_sha256(path)
for row in json.loads(path.read_text())['rows']:
    rows['dice'][row['player']] = row
names = list(dict.fromkeys(['bayes', 'bayes-public', 'mean-sd', 'fixed-prior', 'random',
                           *manifest['results'], 'qwen3.5-4b-staged-sft',
                           *sorted(set(rows['dice']) | set(rows['cards']))]))
names = [name for name in names if any(name in d for d in rows.values())]
aliases = {'bayes': 'Bayes reference', 'bayes-public': 'Public-posterior Bayes',
           'mean-sd': 'Mean ± standard deviation', 'fixed-prior': 'Fixed prior', 'random': 'Random',
           'qwen3-1.7b-sft800': 'Qwen3-1.7B · SFT 800',
           'qwen3.5-4b-grpo-warmstart': 'Qwen3.5-4B · original SFT',
           'qwen3.5-4b-grpo': 'Qwen3.5-4B · original GRPO',
           'qwen3.5-4b-staged-sft': 'Qwen3.5-4B · staged SFT',
           'qwen3.5-4b-staged-grpo': 'Qwen3.5-4B · staged GRPO',
           'qwen3.5-4b-staged-routed': 'Qwen3.5-4B · router (two adapters)'}
for method in ('sft', 'dagger', 'dagger-last', 'rloo', 'rloo96', 'rloo192'):
    aliases['qwen3.5-4b-' + method] = 'Qwen3.5-4B · local ' + method.upper()
aliases.update({
    'nvidia/nemotron-3-ultra-550b-a55b (no-think)': 'Nemotron-3-Ultra · thinking off',
    'nvidia/nemotron-3-ultra-550b-a55b (thinking-medium)': 'Nemotron-3-Ultra · thinking medium',
    'nvidia/nemotron-3.5-lightning-30b-a3b (no-think)': 'Nemotron-3.5-Lightning · thinking off',
    'nvidia/nemotron-3.5-bolt-e4b': 'Nemotron-3.5-Bolt',
    'openai/gpt-oss-20b': 'GPT-OSS-20B · low reasoning',
    'meta/llama-3.2-11b-vision-instruct (text-only)': 'Llama-3.2-11B · text only',
    'meta/llama-3.2-90b-vision-instruct (text-only)': 'Llama-3.2-90B · text only',
    'mistralai/mistral-large-4-1t-a52b (server-default)': 'Mistral Large 4 · server default',
    'mistralai/mistral-large-4-1t-a52b (no-reasoning)': 'Mistral Large 4 · explicit none',
})
labels = [aliases.get(name, name.split('/')[-1]) for name in names]
intervals = [r['pct_of_bayes_ci'] for game in rows.values() for r in game.values()]
low, high = min(-100, min(v[0] for v in intervals)), max(120, max(v[1] for v in intervals))
fig, axes = plt.subplots(1, 2, figsize=(13, 3 + .45 * len(names)), sharey=True, sharex=True)
for ax, game in zip(axes, ('dice', 'cards')):
    for i, name in enumerate(names):
        row = rows[game].get(name)
        if row is None:
            ax.text(.98, i, 'incomplete', ha='right', va='center', color='#65717c',
                    transform=ax.get_yaxis_transform(), fontsize=10)
            continue
        value, (lower, upper) = row['pct_of_bayes'], row['pct_of_bayes_ci']
        analytical = name in ('bayes', 'bayes-public', 'mean-sd', 'fixed-prior', 'random')
        trained = name.startswith('qwen3')
        color = '#59636f' if analytical else '#1d63a3' if trained else '#b87518'
        marker = 'D' if analytical else 'o' if trained else 's'
        ax.errorbar(value, i, xerr=[[value - lower], [upper - value]], fmt=marker,
                    color=color, markersize=4.5, capsize=2, linewidth=1.3)
    ax.set_title(game.capitalize(), fontsize=13, loc='left')
    ax.axvline(0, color='#59636f', linewidth=.9)
    ax.axvline(100, color='#59636f', linewidth=.9, linestyle='--')
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlim(low * 1.15, high * 1.15)
    ticks = [v for v in (-100000, -10000, -1000, -100, 0, 100, 1000, 10000) if low * 1.15 <= v <= high * 1.15]
    ax.set_xticks(ticks, [f'{v:,}' for v in ticks])
    ax.grid(axis='x', color='#dde1e5', linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('% of Bayes expected PnL · symmetric log scale', fontsize=10)
axes[0].set_yticks(range(len(names)), labels, fontsize=14)
axes[0].invert_yaxis()
fig.suptitle('Fresh held-out evaluation', x=.03, ha='left', fontsize=17, fontweight='bold')
fig.text(.03, .945, '200 episodes per complete cohort · 95% paired bootstrap intervals', fontsize=11)
fig.text(.03, .025, 'Linear scale within ±100%. Blue circles: trained policies; gold squares: APIs; grey diamonds: analytical baselines.\nIntervals condition on frozen weights and recorded responses; corrected and inference-group intervals remain in the source JSON.', fontsize=9)
fig.subplots_adjust(left=.34, right=.98, top=.89, bottom=.14, wspace=.12)
out = Path('docs/figures/leaderboard_final.png')
fig.savefig(out, dpi=160, facecolor='white')
plt.close(fig)
record = {'provenance': receipt({'purpose': 'plot every available complete final cohort and mark missing games'}),
          'source_sha256': file_sha256(__file__), 'source': Path(__file__).read_text(),
          'input_sha256': inputs, 'output': str(out), 'output_sha256': file_sha256(out),
          'plot': {'interval': '95% paired episode bootstrap', 'scale': 'symlog, linear between -100% and +100%',
                   'names': names, 'missing_is_not_zero': True}, 'visual_inspection': 'pending'}
Path('results/final/figure.json').write_text(json.dumps(record, indent=2) + '\n')
print(out)
