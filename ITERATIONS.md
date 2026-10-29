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
  `1 - 0.05/(C * (K + 2))` (K comparisons plus dice threshold and cards
  positivity for each of C frozen final candidates). C is four: the completed
  1.7B baseline, the validation-selected MLX 4B candidate, CUDA teacher SFT,
  and CUDA GRPO. Failed/unavailable candidates remain counted conservatively.
  Do not claim a winner from overlapping or uncorrected intervals.
  The K corrected comparisons are paired differences in dice expected PnL;
  the other two claims are the dice ratio and cards positivity. Marginal
  individual-model intervals must also be separated as required by the target.
  Corrected intervals use enough resamples for at least 100 expected draws in
  each tail; marginal intervals retain 10,000. Block sampling bounds memory and
  reproduces the previous bootstrap draws exactly. This amendment precedes new
  final scores and increases conservatism rather than changing model selection.
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
- Authentication repair: official `kaggle auth login --force` completed with
  the existing account. A per-process `KAGGLE_CONFIG_DIR` omits the expired
  legacy key, which otherwise takes precedence over OAuth. The original key
  file remains untouched. `kaggle kernels list --mine` now succeeds.
- Retry: the private 1.7B checkpoint dataset was created. It is retained as an
  earlier baseline and is incompatible with the new 4B model. The frozen CUDA
  recipe therefore uses its own teacher SFT warm start (150 steps), then GRPO
  (300 steps, eight completions/group). Both predeclared checkpoints receive
  one final 200-episode evaluation per game; neither test score changes training.
  Report both candidates; any cross-candidate superiority claim also corrects
  for these two candidates. Training seed 0; exact defaults are in the script.
- Submission retry: accepted private version 1, T4 requested and internet
  enabled; console confirms actual SFT on one of two allocated T4s. Unsloth
  switches Qwen3.5 training to float32 on T4. GRPO completion is still pending.
  Receipt: [kaggle_retry.json](results/training/kaggle_retry.json).
- Future builds save SFT optimizer checkpoints every 50 steps and a warm-start
  adapter, expose SFT resume, skip redundant warm-start work on GRPO resume,
  and record memory for each allocated GPU. These source improvements do not
  modify the already running version 1; its submitted snapshot is preserved.

### T2 — complete earlier Qwen3-1.7B BC (completed after interruption)

- Hypothesis: the truncated earlier run undertrained the teacher policy.
- Change: resume step-300 weights for 500 updates; validation 40 batches rather
  than 10; use a distinct output directory; optimizer state restarts explicitly.
- Command: `bash runs/complete_sft.sh` (full arguments will accompany its result).
- Interim result: validation loss 0.229 at start; 10 updates reported, then the
  owner interrupted the turn. No new checkpoint saved; prior step-300 weights
  remain intact. This is an interrupted trial, not 800 completed updates.
- Restart: resumed the same step-300 adapter with the same continuation recipe.
  The fixed total-step-800 adapter is a predeclared final baseline; it will be
  evaluated once on the fresh 200-episode cohort, independently of T4–T6 selection.
- Evidence: ignored `runs/sft800.log`; `results/training/qwen17_sft800.json`
  will record completion, exact command, restarted optimiser and measured runtime.
  Completion: all 500 continuation minibatches finished, reaching 800 total,
  in 3,912.920 seconds after restart. The CLI printed validation loss 0.227
  before the last minibatch; it sampled 80 examples and is not directly
  comparable to the new fixed full-validation protocol. No policy improvement
  is inferred from this loss. Peak allocator memory was not persisted by the
  upstream CLI and remains unknown. Frozen final adapter SHA is in the receipt;
  its fresh 200-episode/game evaluation is queued under the heavy-ML lock.
  Local base weights/tokenizer match all checked immutable Hub blobs:
  [verification](results/training/qwen17_base_verification.json).

### T3 — extended NIM probes (resumed, running)

- Hypothesis: 600 s timeouts and 16,384 tokens for thinking make more models usable.
- Command: `bash scripts/with_resource.sh nim .venv/bin/python -u scripts/expand_nim.py`.
- Result: interrupted before a completed response; resumed under one shared slot.
  GET `/v1/models` succeeded and returned no Qwen model. The initial long-timeout
  worker is running. A second sequential batch is queued for catalogue-listed
  Mistral-7B, Mistral-NeMo-12B and Llama-3.2-11B (text only), with licences
  verified before calls. Each batch holds its own slot; total requests remain
  bounded by the two shared mutexes. GLM then failed after three requests with
  HTTP 504 in 922.903 seconds, despite the 600-second client timeout; no score
  is assigned. DeepSeek also failed after three HTTP 504 responses in 923.328
  seconds. Ultra with thinking is the next probe. No new policy score is claimed.
- Evidence: [catalogue](results/expansion/models.json), ignored
  `runs/nim_expansion.log` / `runs/nim_supplement.log`; protocol JSON after each probe.
- Focused follow-up: [NVIDIA's GLM guide](https://docs.nvidia.com/nim/large-language-models/2.1.4/get-started/advanced/get-started-glm-5.3-flash.html)
  documents default maximum reasoning and a low-effort control; the
  [upstream DeepSeek serving recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
  documents `enable_thinking=false`. Retry these explicit modes under a shared
  slot with `scripts/expand_nim.py --reduced-reasoning`. This changes decoding
  after availability failures, before any new held-out score is observed.
  Preserve the original failures and both configurations; completed cohorts
  join the API comparison family. Evidence: `results/expansion/protocol-reasoning.json`.

### T4 — Qwen3.5-4B LoRA (running)

- Hypothesis: a stronger small model reduces posterior-inference errors on cards.
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`, revision
  `32f3e8ecf65426fc3306969496342d504bfa13f3` (Apache-2.0 verified from HF API).
- Initial recipe: same Bayes demonstrations, rank 8, 16 layers, batch 1,
  gradient accumulation 2, 1e-4 learning rate, 800-step ceiling, validation-only
  stopping/selection. Policy comparisons use validation seeds above.
- Validation-loss checks cover **all 480 teacher-validation examples**, every
  200 microbatches, with three unsuccessful checks for stopping. This replaces
  the initially drafted random 80-example checks before any T4 execution:
  changing validation subsets could imitate a plateau. The ceiling is 800
  microbatches / 400 optimiser updates; receipts distinguish these counts.
- Prompt contract: explicitly disable thinking during both teacher tokenization
  and generation; audit every training/validation prefix before starting.
- CUDA recipe: `unsloth/Qwen3.5-4B`, revision
  `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`; no 1.7B adapter is loaded into 4B.
- Execution started after T2 released the heavy-ML mutex. Initial full teacher
  validation completed before the first update: loss 2.1129567623138428 in
  1,884.2940601250157 seconds, MLX allocator peak 4,012,511,272 bytes. This is
  an untrained adapter baseline, not a policy result; optimisation is running.
  Exact command and start receipt: `results/training/qwen35_sft.json`.
- Attempt 1 was interrupted by the owner-reported machine reboot. No trained
  adapter checkpoint survived: the saved best adapter is at step zero, and the
  number of unsaved microbatches is unknown (below the first report at ten).
  A pre-reboot `sample` report printed an 11.5G process footprint peak. The
  reboot's cause is not established. Preserve [the attempt receipt](results/training/qwen35_sft_attempt1.json)
  and [log](results/training/qwen35_sft_attempt1.log), rather than claiming completion.

### E1 — engine baseline/profile (in progress)

- Hypothesis: profile the existing flat levels/pool before a focused change.
- Command: `cmake -S . -B build/native -DMMB_BUILD_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -g'`;
  `cmake --build build/native -j 2`; `python scripts/bench_engine.py`.
- Existing baseline: 20,207,599.5 events/s; checksum 6638620577960879557.
- Evidence: [engine_bench.json](results/engine_bench.json). Current rerun/profile
  was interrupted; no optimisation speedup claimed yet.

### E2 — recycle pool nodes through an intrusive free head (confounded)

- Hypothesis: remove the second vector allocation/write when recycling nodes.
- Change: replace `free_` with `free_head_`, reusing a released node's next field.
- Command: 5 alternating pairs of `bench_book_{before,free_head} build/bench_stream.bin 10`.
- Result: extremely inconsistent separate-process wall times under shared load;
  a build/source-edit overlap also makes this unsuitable for code attribution.
  **No speedup claimed.** Counted as a trial and discarded.
- Evidence: [all samples](results/engine_trials/free_head_confounded.json).

### E3 — free head plus one index probe per cancellation (no improvement)

- Hypothesis: avoid repeated hashing/probing for successful cancellations.
- Change: locate an index slot once, validate its owner, then erase that slot.
- Command: `python scripts/compare_engine.py --pairs 50` (old/new books compiled
  into distinct namespaces in one executable, alternating order).
- Result: median wall throughput 22.93M → 22.77M events/s; paired median speedup
  0.990, 95% interval [0.982, 1.005]. **Discarded both speculative changes.**
- Evidence: [result and source hashes](results/engine_trials/free_head_single_probe.json),
  [exact candidate patch](results/engine_trials/free_head_single_probe.patch).

### E4 — inline the matching loop (confirmed improvement)

- Hypothesis: the sampling profile showed matching remained out of line; forcing
  inlining removes an inner call and lets the compiler specialise the hot path.
- Change: one `always_inline` attribute on the existing templated matcher;
  retain the existing flat levels and pool. Separately fix reserved-ID lookup:
  the empty index sentinel must never return a stale value.
- Exploratory command: `python scripts/compare_engine.py --pairs 50 --out results/engine_trials/inline_matching.json`.
- Confirmation command: `python scripts/compare_engine.py --pairs 200 --out results/engine_trials/inline_matching_confirmation.json`.
- Confirmation result: median wall throughput **20,136,331.421 → 20,814,208.545**
  events/s. Median paired speedup **1.0511**, 95% CI **[1.0347, 1.0649]**;
  Bonferroni-for-three-candidates CI **[1.0318, 1.0694]**. Ratios of marginal
  medians and medians of paired ratios are different estimands; both are reported.
  All checksums equal **6638620577960879557**. Machine shared with other work.
- Evidence: [200-pair confirmation](results/engine_trials/inline_matching_confirmation.json),
  [50-pair exploratory trial](results/engine_trials/inline_matching.json).
- Harness failure: naming the generated executable `compare` shadowed the C++20
  `<compare>` header on the next build. Renamed it `bench_comparison`; rerun
  succeeded. This was a build failure, not a performance measurement.
- Verification: 77 Python tests, including 10 stateful fuzz seeds; native
  50,000-event pool/index stress under ASan/UBSan passed locally. CI added.

### T5 — larger Qwen3.5 DAgger round (registered, queued)

- Hypothesis: posterior errors after the policy's own fills require on-policy labels.
- Change: collect 150 training episodes per game (1,800 states), triple the earlier
  round's 600 states, on seeds 30,000–30,149; combine with 4,000 original teacher
  examples. Resume the selected T4 adapter, seed 4, same LoRA recipe and stopping.
- Commands: `train/mlx_dagger.py --episodes 150`; `train/mlx_train.py --iters 800 --seed 4`;
  complete commands, dataset hashes and measurements accompany the receipts.
- Validation: same 100 episodes per game; final test remains untouched.
- Result: pending; `results/training/qwen35_dagger.json` and
  `results/validation/qwen35-dagger/leaderboard_{dice,cards}.json`.

### T6 — local RLOO-style fallback (registered, queued)

- Hypothesis: optimise the exact economic reward rather than only imitate text.
- Change: start from T4; four categorical completions per dice training state,
  leave-one-out baseline, 24 Adam updates at 1e-6, seed 5. No critic, PPO clipping
  or reference KL; this is a small REINFORCE fallback, not an executed Kaggle GRPO.
- Command: `train/mlx_rloo.py --steps 24 --lr 1e-6 --seed 5` under the heavy-ML lock.
- Validation: both games, same validation seeds; no test-dependent adjustment.
- Result: pending; `results/training/qwen35_rloo.json` and
  `results/validation/qwen35-rloo/leaderboard_{dice,cards}.json`.

### T7 — CUDA GRPO compatibility smoke (registered)

- Hypothesis: exercise GRPO before the full SFT job reaches it, so a trainer or
  memory defect is found without waiting for the long warm start.
- Change: separate private kernel, same pinned model, no SFT, one GRPO update
  with two completions on dice training states; one validation episode before
  and after on seed 80,000. This is an infrastructure trial, not a candidate
  selected for final testing or evidence of policy quality.
- Commands: `python train/build_kaggle_kernel.py --smoke`;
  `kaggle kernels push -p runs/kaggle_smoke --accelerator NvidiaTeslaT4`.
- Result: private version 1 accepted, T4 and internet enabled; remote privacy
  verified in [the receipt](results/training/kaggle_smoke.json). It completed:
  **827.4385588169098 seconds** whole-run wall time; **4,484,432,896 bytes** peak
  allocated CUDA memory, 5,110,759,424 reserved. One GRPO update took 243.7 seconds
  (printed trainer precision). Both completions hit the 96-token ceiling and
  received identical invalid-format rewards: reward std and training loss zero.
  This proves execution but provides **no learning signal or quality result**.
  [Runtime](results/kaggle/smoke-v1/runtime.json),
  [complete log](results/kaggle/smoke-v1/makemarketbench-grpo-qwen3-smoke.log).
  The full SFT/GRPO kernel remained running remotely across the local reboot.

### T8 — reduce MLX recurrent-state retention (registered, smoke pending)

- Hypothesis: Qwen3.5's training-only sequential gated-delta fallback retains
  per-token recurrent states; layer checkpointing alone leaves a large peak.
- Change: checkpoint that same recurrence in 64-token chunks. Value and gradient
  parity passed on CPU, including masks and a partial last chunk; inference's
  existing Metal kernel is unchanged. This is rematerialisation, not truncation
  or a changed objective. Limit our training child's sampled physical footprint
  to 7 GiB using a macOS `top` guard, releasing the shared lock on exit.
- Command: two-microbatch smoke on two fixed training examples, followed by the
  original T4 recipe only if the smoke completes within the memory guard.
- Results: pending `results/training/qwen35_memory_smoke.json` and
  `results/training/qwen35_memory_guard.json`. This infrastructure trial does
  not satisfy the two unsuccessful substantive validation trials stopping rule.

Selection registered before T4–T6 validation: maximise the smaller of the two
validation margins, `min(dice_percent_of_Bayes - 80, cards_percent_of_Bayes)`;
ties choose the earlier trial. Report every trial. Freeze the selected adapter
before one final 200-episode test per game. This validation selection is separate
from validation-loss checkpoint selection. If T5 and T6 both fail to improve,
inspect reward, inference and data diagnostics before applying the stopping rule.

### Export correction — unavailable diagnostics

- The completed CUDA smoke produced no valid quotes. Its exact saved JSON
  contains `NaN` for unavailable quote/flow diagnostics, which is not standard
  JSON. Preserve those original outputs unchanged. Future exports use `null`
  and Markdown uses `n/a`; expected PnL, regret and all rewards are unchanged.
- Verification: `pytest -q tests/test_games.py` passed 13 tests, including a
  no-quote episode checked with `json.dumps(..., allow_nan=False)`. This is an
  export fix, not a model-quality iteration.

### Reboot recovery receipts

- The Qwen3-1.7B frozen test was paused after 57 complete dice episodes to yield
  the heavy-ML lock for the short registered memory probe. No scores were
  inspected and neither weights nor decoding changed; only missing episodes
  will resume. [Yield receipt](results/training/qwen17_evaluation_yield.json).
- The memory probe's subset is audited against the original training and
  validation files, with exact selected row indices and hashes:
  [subset audit](results/training/qwen35_memory_data.json).
- The physical-memory guard now persists samples while running. Its small
  process self-check exceeded the intentionally tiny 0.001-GiB limit and exited
  137, as intended: [receipt](results/training/memory_guard_selftest.json).
- The remote full kernel finished its SFT warm start (150 steps). This interim
  UI observation is transcribed at the console's printed precision and is not
  a validation or final policy result:
  [progress](results/training/kaggle_live_progress.json).
