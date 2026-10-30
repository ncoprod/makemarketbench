"""Compare native and explicitly left-padded copies of discrepant development prompts."""
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
    rendered = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for m in messages]
    lengths = [len(tokenizer(s)['input_ids']) for s in rendered]
    indices = [2, 13] if args.label.endswith('sft') else [7]
    output = Path(provenance['profile_output'])
    record = {**provenance, 'states': states, 'prompt_tokens': lengths, 'target_indices': indices,
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'rendered_prompts_sha256': hashlib.sha256(json.dumps(rendered).encode()).hexdigest(),
              'compiler_environment': {'UNSLOTH_COMPILE_DISABLE': '1'}, 'cases': [], 'status': 'running',
              'interpretation': 'Discrepant development prompts only, no held-out episodes or policy tuning. Direct first greedy generation, no deduplication/repair. Padding uses each target original mixed-batch maximum prompt length. Native versus padded single sequence isolates padding length within this execution; four identical padded copies also vary batch size.'}

    def save():
        tmp = output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(output)

    try:
        tokenizer.padding_side = 'left'
        for i in indices:
            pad_length = max(lengths[(i // 4) * 4:(i // 4 + 1) * 4])
            for name, copies, padding in [('native_single', 1, False), ('padded_single', 1, True), ('padded_identical_four', 4, True)]:
                kwargs = {'padding': 'max_length', 'max_length': pad_length} if padding else {}
                inputs = tokenizer([rendered[i]] * copies, return_tensors='pt', **kwargs).to(model.device)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.no_grad():
                    tokens = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                torch.cuda.synchronize()
                answers = []
                for text in tokenizer.batch_decode(tokens[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True):
                    try:
                        quote, error = parse_quote(text, observations[i].game.params.max_size).to_dict(), None
                    except ValueError as exc:
                        quote, error = None, str(exc)
                    answers.append({'text': text, 'quote': quote, 'parse_error': error})
                record['cases'].append({'name': name, 'state_index': i, 'copies': copies,
                                        'input_shape': list(inputs['input_ids'].shape),
                                        'left_pad_tokens': (inputs['attention_mask'] == 0).sum(1).tolist(),
                                        'wall_seconds': time.perf_counter() - started,
                                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'answers': answers})
                save()
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
