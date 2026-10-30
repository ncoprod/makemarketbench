"""Test whether batch companions and prompt lengths change greedy development quotes."""
import hashlib
import json
import time
from pathlib import Path


def run(model, tokenizer, args, quoter_class, provenance):
    import torch

    from makemarket.games import make_game
    from makemarket.llm import parse_quote
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
    messages = [messages_for(obs) for obs in observations]
    prompts = [json.dumps(m, sort_keys=True) for m in messages]
    lengths = [len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False)) for m in messages]
    output = Path(provenance['profile_output'])
    record = {**provenance, 'states': states, 'prompt_tokens': lengths,
              'prompts_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'compiler_environment': {'UNSLOTH_COMPILE_DISABLE': '1'},
              'cases': [], 'status': 'running',
              'interpretation': 'Fixed development states only; no held-out policy episodes or PnL. Direct first greedy generation bypasses prompt deduplication and repair so identical-prompt batches truly contain four sequences. Composition differences do not alone prove a padding or numerical-kernel bug.'}

    def save():
        tmp = output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(output)

    def measure(name, indices, size):
        actor = quoter_class(model, tokenizer, args.label)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        answers = []
        for offset in range(0, len(indices), size):
            batch = indices[offset:offset + size]
            texts = actor._gen_many([messages[i] for i in batch])
            for i, text in zip(batch, texts):
                try:
                    quote, error = parse_quote(text, observations[i].game.params.max_size).to_dict(), None
                except ValueError as exc:
                    quote, error = None, str(exc)
                answers.append({'state_index': i, 'text': text, 'quote': quote, 'parse_error': error})
        torch.cuda.synchronize()
        return {'name': name, 'batch_size': size, 'indices': indices, 'answers': answers,
                'wall_seconds': time.perf_counter() - started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}

    try:
        serial = measure('serial_including_warmup', list(range(16)), 1)
        record['serial'] = serial
        baseline = [a['quote'] for a in serial['answers']]
        save()
        orders = {'natural': list(range(16)), 'reverse': list(reversed(range(16))),
                  'length_ascending': sorted(range(16), key=lambda i: (lengths[i], i)),
                  'length_descending': sorted(range(16), key=lambda i: (-lengths[i], i)),
                  'identical_known_discrepancies': [2] * 4 + [13] * 4}
        for name, indices in orders.items():
            case = measure(name, indices, 4)
            case['mismatch_positions'] = [j for j, a in enumerate(case['answers'])
                                          if a['quote'] != baseline[a['state_index']]]
            record['cases'].append(case)
            save()
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
