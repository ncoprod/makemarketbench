# Optimisation trials

## Targets and stopping rule (registered 2026-10-29)

- Final evaluation: **200 complete episodes per player per game**, common seeds,
  episode-level paired bootstrap, 10,000 resamples, 95% marginal intervals.
- Best fine-tuned policy: **at least 80% of Bayes expected PnL on dice**;
  its lower interval must exceed every evaluated API policy's upper interval.
- Cards: positive expected PnL with the lower interval strictly above zero.
- Execute Kaggle GRPO on a GPU and record measured runtime and peak VRAM.
- Improve native engine events/s on the identical event stream and checksum;
  retain Python parity, property tests and ASan/UBSan CI.

Stop after all targets pass, or after two consecutive substantive validation
trials fail to improve and the remaining gap has an evidenced explanation.
External access failures remain unmet targets, not successful training trials.

## Splits, selection and statistical policy

- Original exploratory results on seeds 0–49 remain in the repository. Those
  seeds have already influenced development and are **not a fresh test**.
- Demonstrations: seeds 10,000–10,799 per game. Teacher validation examples:
  seeds 60,000–60,039. DAgger collection: seeds 30,000 and above.
- New policy selection: validation episodes **80,000–80,099**, separately from
  teacher-loss validation. No configuration is chosen from final test scores.
- Fresh final test: **200–399**, shared by all final policies and references.
  Evaluate each frozen final candidate once; retry missing episodes only.
- Count every training/configuration attempt, including interrupted/failed ones.
  Report all validation trials and all frozen final candidates, not only a winner.
- Marginal 95% CIs are descriptive. Any claim of superiority across K API
  comparisons must also pass family-wise Bonferroni intervals at level
  `1 - 0.05/(K + 2)` (K comparisons plus dice threshold and cards positivity).
  Do not claim a winner from overlapping or uncorrected intervals.
- Keep raw episodes, settings, seed ranges, model revisions, source commit,
  commands, failures and measured resource usage with each new result.

## Earlier exploratory trials

| Trial | Hypothesis / change | Command / evidence | Result |
|---|---|---|---|
| H1 | Qwen3-1.7B base | `train/mlx_pipeline.sh 0 0`; `results/raw/{dice,cards}/qwen3-1.7b-base.jsonl` | Negative in both games; historical 50-seed test |
| H2 | Clone exact Bayes decisions, 300/800 LoRA steps | `train/mlx_pipeline.sh 1 2`; `results/training/sft_qwen3-1.7b.log` | Dice 68% [38,89]; cards −80% [−161,−15]; historical test |
| H3 | 600 on-policy DAgger labels, 200 more updates | `train/mlx_pipeline.sh 3 4`; `results/training/dagger_qwen3-1.7b.log` | Dice regressed; cards improvement uncertain; historical test |

The earlier logs do not contain immutable model revisions or complete commands;
do not invent missing provenance. Historical adapter configs are preserved.

## Current trials

### T1 — private Kaggle GPU submission (failed before execution)

- Hypothesis: move GRPO and paired evaluation to a T4, starting from the BC adapter.
- Change: export MLX's 224 adapter tensors to PEFT, preserving scale; embed CPU
  benchmark package; evaluate and score inside Kaggle; save runtime/VRAM JSON.
- Commands: `kaggle datasets create -p runs/kaggle_dataset`;
  `kaggle kernels push -p train/kaggle --accelerator NvidiaTeslaT4`.
- Result: CLI says `Authentication required to call the Kaggle API.` Direct
  legacy-credential request returns **HTTP 401 Unauthorized**. No dataset/kernel
  created; GPU/internet allocation was never reached. No runtime/VRAM measured.
- Evidence: [kaggle_attempt.json](results/training/kaggle_attempt.json).
- Adapter export verification: maximum forward-delta error 1.6093254089355469e-6;
  CUDA NF4 and MLX affine quantization still produce different base weights.

### T2 — complete earlier Qwen3-1.7B BC (interrupted, continuation pending)

- Hypothesis: the truncated earlier run undertrained the teacher policy.
- Change: resume step-300 weights for 500 updates; validation 40 batches rather
  than 10; use a distinct output directory; optimizer state restarts explicitly.
- Command: `bash runs/complete_sft.sh` (full arguments will accompany its result).
- Interim result: validation loss 0.229 at start; 10 updates reported, then the
  owner interrupted the turn. No new checkpoint saved; prior step-300 weights
  remain intact. This is an interrupted trial, not 800 completed updates.
- Evidence: ignored `runs/sft800.log`; final machine-readable result pending.

### T3 — extended NIM probes (interrupted, continuation pending)

- Hypothesis: 600 s timeouts and 16,384 tokens for thinking make more models usable.
- Command: `bash scripts/with_resource.sh nim .venv/bin/python -u scripts/expand_nim.py`.
- Result: interrupted before a completed response; no new score claimed.
- Evidence: ignored `runs/nim_expansion.log`; protocol JSON is saved after each probe.

### T4 — Qwen3.5-4B LoRA (registered, not yet executed)

- Hypothesis: a stronger small model reduces posterior-inference errors on cards.
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`, revision
  `32f3e8ecf65426fc3306969496342d504bfa13f3` (Apache-2.0 verified from HF API).
- Initial recipe: same Bayes demonstrations, rank 8, 16 layers, batch 1,
  gradient accumulation 2, 1e-4 learning rate, 800-step ceiling, validation-only
  stopping/selection. Policy comparisons use validation seeds above.
- CUDA recipe: `unsloth/Qwen3.5-4B`, revision
  `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`; no 1.7B adapter is loaded into 4B.
- Result / commands / resource measurements: pending execution.

### E1 — engine baseline/profile (in progress)

- Hypothesis: profile the existing flat levels/pool before a focused change.
- Command: `cmake -S . -B build/native -DMMB_BUILD_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -g'`;
  `cmake --build build/native -j 2`; `python scripts/bench_engine.py`.
- Existing baseline: 20,207,599.5 events/s; checksum 6638620577960879557.
- Evidence: [engine_bench.json](results/engine_bench.json). Current rerun/profile
  was interrupted; no optimisation speedup claimed yet.
