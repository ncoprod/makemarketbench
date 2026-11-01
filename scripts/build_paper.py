"""Build a short research note directly from verified final score files."""

import json
import subprocess
from pathlib import Path

from makemarket.provenance import file_sha256, receipt

manifest_path = Path('results/final/summary.json')
manifest = json.loads(manifest_path.read_text())
inputs = {str(manifest_path): file_sha256(manifest_path)}
scores = {}
for label, entry in manifest['results'].items():
    assert file_sha256(entry['file']) == entry['sha256']
    inputs[entry['file']] = entry['sha256']
    scores[label] = json.loads(Path(entry['file']).read_text())


def interval(row):
    low, high = row['pct_of_bayes_ci']
    return f"{row['pct_of_bayes']:.1f} [{low:.1f}, {high:.1f}]"


def policy_row(label, game):
    data = scores[label]['games'].get(game)
    return next(r for r in data['rows'] if r['player'] == label) if data else None


aliases = {
    'qwen3-1.7b-sft800': 'Qwen3-1.7B SFT800',
    'qwen3.5-4b-rloo192': 'Qwen3.5-4B local RLOO192',
    'qwen3.5-4b-grpo-warmstart': 'Qwen3.5-4B original SFT',
    'qwen3.5-4b-grpo': 'Qwen3.5-4B original GRPO',
    'qwen3.5-4b-staged-grpo': 'Qwen3.5-4B staged GRPO',
    'qwen3.5-4b-staged-routed': 'Qwen3.5-4B router (two adapters)',
    'qwen3.8-27b-batch4-sft': 'Qwen3.8-27B SFT, batch four',
    'nvidia/nemotron-3-ultra-550b-a55b (no-think)': 'Nemotron-3-Ultra, thinking off',
    'openai/gpt-oss-20b': 'GPT-OSS-20B, low reasoning',
    'nvidia/nemotron-3.5-bolt-e4b': 'Nemotron-3.5-Bolt',
    'nvidia/nemotron-3.5-lightning-30b-a3b (no-think)': 'Nemotron-3.5-Lightning, thinking off',
}
newest = 'qwen3.8-27b-batch4-sft'
dice = interval(policy_row(newest, 'dice'))
cards = policy_row(newest, 'cards')
card_text = interval(cards) if cards else 'incomplete (192/200 episodes), unscored'
lines = [
    '---', 'title: "MakeMarketBench: synthetic market making under informed flow"',
    'author: Nicolas Coquelet', 'date: 1 November 2026', '---',
    '## Research note',
    f'Qwen3.8-27B SFT reaches **{dice}%** of the Bayes dealer expected PnL on dice '
    f'and **{card_text}** on cards. Complete test games use 200 common episodes; '
    'intervals are marginal 95% paired bootstrap intervals. This is a synthetic '
    'benchmark, not evidence of profitability on an exchange. All four complete '
    'NIM API baselines have negative mean expected PnL in both games.',
    '## Results',
    'TEST uses seeds 200-399. VAL uses 80000-80099 and selects the frozen primary '
    '27B adapter. The table reports percentage of Bayes expected PnL [95% CI].',
    '| Dealer | Split, n/game | Dice | Cards |', '|:---|:---|---:|---:|',
    '| Bayes reference | TEST, 200 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |',
]
for label in scores:
    values = [policy_row(label, game) for game in ('dice', 'cards')]
    n = '200' if all(r is not None for r in values) else '200 dice only'
    cells = [interval(r) if r is not None else 'Incomplete; unscored' for r in values]
    lines.append(f"| {aliases[label]} | TEST, {n} | {' | '.join(cells)} |")
first = next(iter(scores.values()))
for name in first['games']['dice']['api_cohorts']:
    values = [next(r for r in first['games'][g]['rows'] if r['player'] == name) for g in ('dice', 'cards')]
    lines.append(f"| {aliases[name]} | TEST, 200 | {' | '.join(interval(r) for r in values)} |")
selection_path = Path('results/training/qwen38_selection.json')
inputs[str(selection_path)] = file_sha256(selection_path)
selected = next(t for t in json.loads(selection_path.read_text())['trials'] if t['label'] == newest)
lines.append('| Qwen3.8-27B SFT, batch four | VAL, 100 | ' +
             ' | '.join(interval(selected['scores'][g]) for g in ('dice', 'cards')) + ' |')
lines += [
    'The staged 4B SFT candidate additionally has 200 dice episodes (identical to '
    'the router dice row), but only 100 cards episodes, unscored. The router uses '
    'SFT on dice and GRPO on cards, frozen using validation. The raw logs retain '
    'every failed or incomplete run. Full tables, sources and limitations: '
    '[repository README](https://github.com/ncoprod/makemarketbench).',
    r'\newpage',
    '## Protocol and limitations',
    'A dealer posts prices and sizes for a hidden terminal value. Public reveals '
    'and its own fills update its information. An informed trader knows the value; '
    'noise flow supplies spread revenue. The Bayes reference updates its exact '
    'posterior and optimizes one-round expected PnL. It is myopic, not a proven '
    'upper bound on the sequential game.',
    'Training, validation and test seeds are disjoint. Both local and newest-family '
    'choices freeze before test aggregates are opened. The conservative final '
    'family counts eleven candidates, including unavailable trials. Registered '
    'claims require corrected paired API contrasts and positive cards lower bounds '
    'under both episode and whole-inference-group bootstraps. A uniform tenfold '
    'bootstrap draw audit retains the primary verdicts; it cannot upgrade failures.',
    'The 27B completion gate has an explicit disclosed deviation: complete SFT '
    'validation and the GRPO dice score mathematically fix the unchanged max-min '
    'winner, whatever missing GRPO cards would score. This certifies the selection '
    'outcome, not statistical superiority. Partial games are never extrapolated.',
    'Intervals condition on frozen weights and recorded responses; training-seed '
    'variation and new API samples need replications. Quantizers, adapters and '
    'training recipes differ across backends. GRPO/RLOO rewards use teacher-visited '
    'states; learned-policy fills can induce a different state distribution. '
    'The earlier exploratory 50-episode results influenced development and are '
    'retained separately rather than treated as a controlled before/after experiment.',
    '## Compute and engine engineering',
    'A private saved batch completes 800 SFT and 96 GRPO updates of '
    'Apache-2.0 Qwen3.8-27B on two T4 GPUs in 17,901.346 seconds. '
    'Peak allocations are 12.238 / 12.261 GB. Periodic checkpoints retain optimizer '
    'and RNG state. A second training seed completes but remains validation-only. '
    'The three deadline-stopped evaluation versions preserve downloaded raw episodes '
    'and logs. The single bounded continuation fails before inference because Kaggle '
    'omits the ERROR source kernel from the saved input mounts. No new episodes '
    'are acquired; cards remains at 192/200, unscored. No second rerun is submitted.',
    'The C++20 book uses flat price levels, pooled orders, intrusive FIFO links '
    'and branch-light inlined matching. In 200 alternating pairs, marginal median '
    'throughput improves from 20.14M to 20.81M events/s. Median paired speedup is '
    '1.0511 [1.0318, 1.0694] after trial correction. Checksums match '
    '(6638620577960879557). The marginal-median ratio and median paired ratio '
    'are different estimands. Stateful Python/C++ parity, Release and ASan/UBSan '
    'checks cover the engine; the stream measures in-memory throughput, not network latency.',
    '## Reproduction and evidence',
    'The README reproduction commands rebuild the scores, bootstrap audit and '
    'figure from committed raw episodes without model weights or new API calls. '
    'Run `bash scripts/ci_local.sh` for the local fallback and '
    '`python scripts/build_paper.py` to rebuild this note.',
    'The [final score manifest](https://github.com/ncoprod/makemarketbench/blob/main/results/final/summary.json) '
    'lists every score path and raw hash. The '
    '[selection receipt](https://github.com/ncoprod/makemarketbench/blob/main/results/training/qwen38_selection.json), '
    '[error receipts](https://github.com/ncoprod/makemarketbench/blob/main/results/training/qwen38_terminal_errors.json), '
    '[continuation receipt](https://github.com/ncoprod/makemarketbench/blob/main/results/training/qwen38_test_finish.json), '
    '[training receipt](https://github.com/ncoprod/makemarketbench/blob/main/results/training/qwen38_seed0_completion.json) '
    'and [engine samples](https://github.com/ncoprod/makemarketbench/blob/main/results/engine_trials/inline_matching_confirmation.json) '
    'retain commands, configs, revisions, source commits and runtimes. The '
    '[paper receipt](https://github.com/ncoprod/makemarketbench/blob/main/results/final/paper.json) '
    'records the exact build and input hashes.',
]
for source in ('results/training/qwen38_seed0_completion.json',
               'results/engine_trials/inline_matching_confirmation.json',
               'results/training/qwen38_test_finish_completion.json'):
    inputs[source] = file_sha256(source)
folder = Path('paper')
folder.mkdir(exist_ok=True)
markdown = '\n'.join(lines[:5]) + '\n\n'
previous = ''
for line in lines[5:]:
    if line.startswith('|') and not previous.startswith('|'):
        markdown += '\n'
    markdown += ('' if line.startswith('|') else '\n') + line + '\n'
    previous = line
(folder / 'main.md').write_text(markdown)
command = ['pandoc', 'paper/main.md', '--pdf-engine=tectonic', '-V', 'geometry:margin=18mm',
           '-V', 'papersize=a4', '-V', 'fontsize=10pt', '-V', 'colorlinks=true', '-o', 'paper/main.pdf']
subprocess.run(command, check=True)
record = {'provenance': receipt({'purpose': 'build a research note from final cached scores'}),
          'source_sha256': file_sha256(__file__), 'command': command, 'input_sha256': inputs,
          'markdown_sha256': file_sha256(folder / 'main.md'),
          'pdf_sha256': file_sha256(folder / 'main.pdf'), 'visual_inspection': 'pending',
          'note': 'No prior paper source existed in the checkout; this is a new concise research note, not a recovered older paper.'}
Path('results/final/paper.json').write_text(json.dumps(record, indent=2) + '\n')
print(folder / 'main.pdf')
