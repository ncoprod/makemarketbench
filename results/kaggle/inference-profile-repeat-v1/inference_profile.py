"""Measure batching time, memory and exact quote agreement on development states."""
import hashlib
import json
import random
import time
from pathlib import Path


def run(model, tokenizer, args, quoter_class, provenance):
    import torch

    from makemarket.games import make_game
    from makemarket.market import episode_steps
    from makemarket.players import BayesQuoter
    from makemarket.prompts import messages_for

    observations, states = [], []
    for game in ('dice', 'cards'):
        for i, t in enumerate((0, 1, 3, 5, 0, 1, 3, 5)):
            teacher = BayesQuoter()
            teacher.reset(20000 + i)
            steps = episode_steps(make_game(game), 20000 + i, 'teacher', 'py')
            obs = next(steps)
            for _ in range(t):
                obs = steps.send(teacher.quote(obs))
            observations.append(obs)
            states.append({'game': game, 'seed': 20000 + i, 'round': t})
    prompts = [json.dumps(messages_for(obs), sort_keys=True) for obs in observations]
    output = Path(provenance['profile_output'])
    record = {**provenance, 'states': states, 'unique_prompts': len(set(prompts)),
              'prompts_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
              'profile_driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'measurements': [], 'status': 'running',
              'interpretation': 'Development-state latency and numerical agreement only. No held-out episodes, PnL or model selection. Fresh quoter per case; intra-case prompt caching retained. Two repeats are descriptive; allocator/compiler history can affect timing.'}

    def measure(size):
        quoter = quoter_class(model, tokenizer, args.label)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        answers = []
        for offset in range(0, len(observations), size):
            answers.extend(quoter.quote_many(observations[offset:offset + size]))
        torch.cuda.synchronize()
        return {'batch_size': size, 'wall_seconds': time.perf_counter() - started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'quotes': [q.to_dict() if q is not None else None for q, _ in answers],
                'attempts': [info['attempts'] for _, info in answers],
                'cached_observations': sum(info['cached'] for _, info in answers)}

    def save():
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(output)

    try:
        baseline = measure(1)
        record['initial_serial_baseline_including_warmup'] = baseline
        save()
        rng = random.Random(0)
        for repeat in range(2):
            order = [1, 2, 4, 8, 16]
            rng.shuffle(order)
            for size in order:
                measurement = measure(size)
                measurement['repeat'] = repeat
                measurement['quote_mismatch_indices'] = [i for i, (a, b) in enumerate(zip(baseline['quotes'], measurement['quotes'])) if a != b]
                record['measurements'].append(measurement)
                save()
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
