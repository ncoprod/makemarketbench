"""Resume the registered newest-family recipe from retained Kaggle checkpoints.

Run only after downloading and preserving the preceding attempt. SFT keeps its
original validation cadence; GRPO saves more often within the remaining budget.
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
    required = ['adapter_model.safetensors', 'trainer_state.json', 'optimizer.pt',
                'scheduler.pt', 'rng_state.pth']
    complete = [p for p in directory.glob('checkpoint-*')
                if all((p / name).is_file() for name in required)]
    return max(complete, key=lambda p: int(p.name.split('-')[-1]), default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core', default='/kaggle/working/train_core.py')
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--budget-seconds', type=int, default=6300)
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
              'core_sha256': core_sha256, 'model_revision': previous['model_revision'],
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
            run('sft', ['--sft-resume', str(checkpoint), '--grpo-steps', '0', '--save-steps', '50'])
        checkpoint = latest_checkpoint(work / 'grpo')
        if checkpoint:
            step = json.loads((checkpoint / 'trainer_state.json').read_text())['global_step']
            flags = ['--resume', str(checkpoint), '--adapter', str(checkpoint)]
        else:
            step, flags = 0, ['--adapter', str(warm)]
        assert step < 96, 'The original GRPO checkpoint already reached its ceiling'
        record['grpo_start_step'] = step
        record['grpo_total_ceiling'] = min(96, step + 32)
        run('grpo', [*flags, '--grpo-steps', str(record['grpo_total_ceiling']), '--save-steps', '5'])
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
