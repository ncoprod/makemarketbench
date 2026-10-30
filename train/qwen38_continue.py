"""Resume the registered newest-family recipe from retained Kaggle checkpoints.

Run after downloading the preceding saved batch outputs. Keep both original
optimizer schedules and resume only a complete checkpoint.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def latest_checkpoint(directory):
    required = ['adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json', 'optimizer.pt',
                'scheduler.pt', 'rng_state.pth']
    manifest = directory.parent / 'checkpoints.jsonl'
    complete = []
    for line in manifest.read_text().splitlines() if manifest.exists() else []:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # A killed save can leave its final manifest line incomplete.
        if entry['phase'] != directory.name:
            continue
        path = directory / f"checkpoint-{entry['step']}"
        if all((path / name).is_file() and
               hashlib.sha256((path / name).read_bytes()).hexdigest() == entry['sha256'].get(name)
               for name in required):
            complete.append(path)
    return max(complete, key=lambda p: int(p.name.split('-')[-1]), default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core', default='/kaggle/working/train_core.py')
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--budget-seconds', type=int, default=37800)
    args = ap.parse_args()
    work = Path(args.core).parent
    previous = json.loads((work / 'runtime.json').read_text())
    assert previous['config']['revision'] == '8aa5f05d26b7205477066e1449e0af13f762a299'
    assert not ((work / 'lora_adapter').exists() and previous['status'] == 'completed'), \
        'The original GRPO recipe already completed; do not continue it'
    core_sha256 = hashlib.sha256(Path(args.core).read_bytes()).hexdigest()
    assert core_sha256 == previous['source_file_sha256'], 'The core differs from the preceding run'
    shutil.copyfile(work / 'runtime.json', work / 'preceding_runtime.json')
    started = time.monotonic()
    deadline = started + args.budget_seconds
    record = {'source_commit': args.source_commit, 'command': [sys.executable, *sys.argv],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'preceding_core_sha256': core_sha256,
              'core_change': 'None. Run the exact retained core; this receipt identifies the continuation producer.',
              'model_revision': previous['model_revision'],
              'seed': previous['seed'], 'config': vars(args),
              'preceding_runtime_sha256': hashlib.sha256((work / 'preceding_runtime.json').read_bytes()).hexdigest(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'stages': []}

    def run(name, flags):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError('Continuation wall budget exhausted')
        command = [sys.executable, args.core, '--train-only', '--budget-seconds', str(remaining), *flags]
        record['stages'].append({'stage': name, 'command': command})
        with (work / f'continuation_{name}.log').open('w') as log:
            try:
                subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                               timeout=remaining + 30, check=True)
            finally:
                shutil.copyfile(work / 'runtime.json', work / f'continuation_{name}_runtime.json')

    try:
        warm = work / 'warmstart_adapter'
        if not all((warm / name).is_file() for name in ['adapter_model.safetensors', 'adapter_config.json']):
            checkpoint = latest_checkpoint(work / 'sft')
            assert checkpoint is not None, 'No complete SFT optimizer checkpoint'
            run('sft', ['--adapter', str(checkpoint), '--sft-resume', str(checkpoint),
                        '--grpo-steps', '0', '--save-steps', '50'])
        checkpoint = latest_checkpoint(work / 'grpo')
        if checkpoint:
            step = json.loads((checkpoint / 'trainer_state.json').read_text())['global_step']
            flags = ['--resume', str(checkpoint), '--adapter', str(checkpoint)]
        else:
            step, flags = 0, ['--adapter', str(warm)]
        record['grpo_start_step'] = step
        record['grpo_total_ceiling'] = 96
        if step < 96:
            run('grpo', [*flags, '--grpo-steps', '96', '--grpo-save-steps', '5'])
        else:
            assert step == 96, 'Checkpoint exceeds the registered GRPO ceiling'
            final = work / 'lora_adapter'
            final.mkdir(exist_ok=True)
            for name in ['adapter_model.safetensors', 'adapter_config.json']:
                shutil.copyfile(checkpoint / name, final / name)
            record['recovered_final_adapter_from_checkpoint'] = str(checkpoint)
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(wall_seconds=time.monotonic() - started,
                      ended_at=datetime.now(timezone.utc).isoformat())
        (work / 'continuation.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
