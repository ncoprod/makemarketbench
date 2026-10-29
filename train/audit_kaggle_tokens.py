"""Audit the fixed newest-family training context without loading model weights."""

import json
from pathlib import Path

from grpo_unsloth_kaggle import build_prompt_set, render, sft_target
from transformers import AutoTokenizer

from makemarket.provenance import file_sha256, receipt


def main():
    model = 'unsloth/Qwen3.8-27B-unsloth-bnb-4bit'
    revision = '8aa5f05d26b7205477066e1449e0af13f762a299'
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    result = {'provenance': receipt({'model': model, 'seed': 0}, revision),
              'source_sha256': file_sha256(__file__), 'splits': {}}
    rows = build_prompt_set(400, ['dice', 'cards'])
    for name, examples in [('sft', rows[:800]), ('grpo', rows[800:]),
                           ('teacher_validation', build_prompt_set(4, ['dice', 'cards'], 60000))]:
        prompts, targets = [], []
        for row in examples:
            prompt = render(tokenizer, row['messages'])
            prompts.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
            targets.append(len(tokenizer.encode(prompt + sft_target(row) + tokenizer.eos_token,
                                                 add_special_tokens=False)))
        result['splits'][name] = {
            'states': len(examples), 'max_prompt_tokens': max(prompts),
            'max_target_tokens': max(targets),
            'prompts_over_grpo_928': sum(n > 928 for n in prompts),
            'targets_over_sft_1024': sum(n > 1024 for n in targets)}
    Path('results/training/qwen38_token_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['splits'], indent=2))


if __name__ == '__main__':
    main()
