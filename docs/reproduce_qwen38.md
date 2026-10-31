# Reproduce the saved Qwen3.8-27B experiment

The primary run starts from the pinned 4-bit base, rather than the lost
interactive draft. Its exact submitted source, package versions, data seeds,
adapter hashes and optimizer states are indexed by the
[completion receipt](../results/training/qwen38_seed0_completion.json).
The [protocol](../results/protocol.json) separates training, validation and test.

Use an authenticated Kaggle CLI and keep credentials outside the repository.
Build a private **script kernel** and push it as a saved version:

```bash
python train/build_kaggle_kernel.py --qwen38-train --out runs/reproduce_train
# Check the account/slug in kernel-metadata.json and require is_private=true.
kaggle kernels push -p runs/reproduce_train --accelerator NvidiaTeslaT4 --timeout 42000
# Sleep at least 600 seconds between status requests.
kaggle kernels status nicolascoquelet/makemarketbench-qwen3-8-27b-training
kaggle kernels output nicolascoquelet/makemarketbench-qwen3-8-27b-training \
  -p runs/reproduced_training
```

This runs on both T4s. The checkpoint revision is
`8aa5f05d26b7205477066e1449e0af13f762a299`; `runtime.json` records the actual
revision, package versions, command, seeds and memory. SFT validates teacher
loss periodically and restores the best checkpoint. `sft/checkpoint-*` and
`grpo/checkpoint-*` include optimizer, scheduler and RNG state; the two exported
inference adapters are `warmstart_adapter` and `lora_adapter`. The rolling
checkpoint inventory is `checkpoints.jsonl`. An interrupted saved version can
feed `--qwen38-continue --checkpoint-kernel <saved-kernel>`; this resumes the
retained optimizer state rather than silently restarting its schedule.

Evaluate each adapter on validation before selecting a test candidate:

```bash
python train/build_kaggle_kernel.py --qwen38-evaluate --evaluate-adapter sft \
  --evaluation-budget-seconds 21300 --out runs/reproduce_sft_validation
# Push this private saved kernel, retain its returned reference, then pull outputs.
# Repeat with --evaluate-adapter grpo and a distinct kernel slug.
```

The default serial policy has its own label. `--newest-batch-size 4` builds a
separate `qwen3.8-27b-batch4-*` variant; use it only under the registered
[throughput-probe promotion rule](../results/training/qwen38_batch_probe_plan.json)
and amend the experiment family before acquiring its validation. A small quote
parity check does not establish numerical equivalence of these policies.

Put the two downloaded policy directories under one validation root. Freeze
the max-min validation choice, with the matching inference batch size:

```bash
python scripts/select_qwen38.py --training-root runs/reproduced_training \
  --validation-root runs/reproduced_validation --test-root runs/reproduced_test \
  --batch-size 1 --out runs/reproduced_selection.json
```

The selector checks the pinned revision, exact adapter hash, primary training
seed, inference batch size and complete validation seeds. Its output is
immutable. Use its `method` for `--evaluate-adapter` and the same batch size in
one `--qwen38-evaluate --evaluation-split test` build. Test seeds and cohort size
come from the registered protocol. Do not inspect aggregate test scores to pick
another adapter or training budget.

The actual saved campaign uses batch four and the explicit
`--certify-incomplete-loser` option. SFT's complete max-min score exceeds the
upper bound supplied by GRPO's complete dice score, so no possible cards score
can make GRPO win. The original both-cards completion gate is therefore waived,
with the exact [bound and protocol deviation](../results/training/qwen38_selection_bound_plan.json)
recorded before test acquisition. This option rejects an incomplete candidate
that could still win; partial cards are never scored. It certifies the original
selection outcome, rather than statistical superiority of SFT.

If an evaluation hits its wall cap, retain the raw episodes and pass
`--evaluation-resume-kernel <saved-evaluation-kernel>` to a new evaluation build.
The resume checks policy identity and canonical game states, preserves the
acquired prefix, restores recorded greedy prompt responses and acquires only
missing seeds. Batched resumes require complete original seed groups. Partial
game cohorts remain unscored.

The [budget ledger](../results/training/kaggle_budget.json) keeps script runtime
separate from official account billing. Session timeout flags are upper bounds,
not measurements. Check quota and reserve the complete primary evaluation
before submitting optional replications or ablations.
