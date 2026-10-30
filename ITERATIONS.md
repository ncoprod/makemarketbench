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
  positivity for each of C frozen final candidates). C is nine: the completed
  1.7B baseline, the validation-selected MLX 4B candidate, CUDA teacher SFT,
  and CUDA GRPO, plus a separately staged CUDA warm start and GRPO pair,
  and the newest-family Qwen3.8 SFT/GRPO pair, plus the validation-selected
  two-adapter staged CUDA game router.
  Failed/unavailable candidates remain counted conservatively.
  Do not claim a winner from overlapping or uncorrected intervals.
  The K corrected comparisons are paired differences in dice expected PnL;
  the other two claims are the dice ratio and cards positivity. Marginal
  individual-model intervals must also be separated as required by the target,
  in both expected-PnL units and the leaderboard's percentage-of-Bayes units.
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

### T8 — reduce MLX recurrent-state retention (failed memory smoke)

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

### Shared-resource preparation — CPU references

- Add `mmb eval/score --results-dir` and baseline/score receipts, so the
  mathematical references can be computed outside the heavy-ML mutex and
  resumed by the existing policy evaluator. No game or model setting changes.
- End-to-end check: generate and score Bayes on dice seed 80,000 in an ignored
  output directory; verify the provenance command, exact seed, 100% self-ratio
  and standard JSON. This is workflow preparation, not a policy trial.

### CUDA checkpoint/session separation (not yet submitted)

- Future builds accept `--train-only` to save warm-start and GRPO adapters before
  a separate `--eval-only --adapter DIR` session. Explicit adapters are checked
  against the requested base and hashed. GRPO resume preserves the original
  SFT/data partition. The already running version 1 is unchanged.
- Verification: CPU reward dry run and paired JSON export test pass. CUDA
  execution of these new stage flags is not yet verified; it is not claimed.

### T8 outcome — checkpointing alone was insufficient

- The two-minibatch probe exceeded its 7-GiB physical-footprint guard after
  82.85770724999975 seconds. Maximum sampled footprint was approximately
  8,478,785,536 bytes; periodic samples can overshoot the guard. Only our child
  was killed, with no verified optimiser update. Unsaved minibatch progress is
  unknown. [Training/termination receipt](results/training/qwen35_memory_smoke.json),
  [guard](results/training/qwen35_memory_guard.json),
  [log](results/training/qwen35_memory_smoke.log). The original T4 recipe was
  not launched. This failure does not count as substantive non-improvement.

### T9 — chunk-parallel MLX training recurrence (registered, smoke queued)

- Hypothesis: replace sequential per-token training states with the algebraic
  chunkwise formulation, reducing both retained states and dispatch overhead.
- Change: port the two algorithm functions from MLX-LM PR #1870, pinned to
  `3b04b9f1992fb83c4210228f7eb1c9dc72827e49`, under its verified MIT licence.
  Function-body ASTs match upstream; only annotations/import style changed.
  Scalar unmasked training uses the port; masks/vector gates retain the
  checkpointed reference. Inference's Metal path is unchanged.
- Verification: seven CPU value/gradient parity cases pass, covering grouped
  heads, nonzero initial state, masks, vector gates, full/partial chunks,
  strong decay, float32 and bfloat16. GPU memory feasibility is still unverified.
- Command: same two-minibatch subset and initial adapter as T8, new output
  `train/runs/qwen35-parallel-smoke`, under the same 7-GiB guard and heavy lock.
- Results pending: `results/training/qwen35_parallel_smoke.json` and
  `results/training/qwen35_parallel_memory_guard.json`. Do not start full
  training until this probe passes. This remains an infrastructure trial.

### Completed reference preparation and monitoring recovery

- All 20 reference files are complete: five mathematical baselines, two games,
  200 fresh-test episodes and 100 validation episodes per player/game, 3,000
  episodes total. [Manifest and hashes](results/reference_cohorts.json). No
  learned-policy test score was inspected.
- Two scheduled private-kernel status polls failed after the SDK reused an
  expired OAuth token. Refreshing the existing credentials restored private
  kernel listing; the original account/scopes are unchanged and no token was
  printed. Exact failures and recovery are in
  [the submission receipt](results/training/kaggle_retry.json). This is a
  monitoring/authentication failure, not evidence that remote training failed.

### Evaluation observability

- Sequential evaluation now prints flushed progress every five new complete
  episodes, matching the existing threaded cadence. Quotes, seeds and scoring
  are unchanged. The submitted CUDA version 1 is unchanged.
- End-to-end check: a six-episode Bayes run prints `5/6`; resuming the same run
  skips all six and prints no new-episode progress.

### API batch provenance and disjoint cohorts

- Freeze source commit and file hashes once when each sequential worker starts;
  retain served model/fingerprint and token metadata already present in caches.
  Remote immutable deployment revisions are unavailable and remain labelled so.
- Separate the explicit three-model supplement from the default batch, avoiding
  simultaneous writes to the same cohorts. Commands are in the README.
- The resumed default worker started before this patch and saved seven complete
  GPT-OSS-20B dice episodes. Restart it with the existing cache/raw episodes;
  completed episodes are skipped. No aggregate test score was inspected.
- Verification: Ruff passes; 16 parsing/training tests pass (seven recurrence
  cases excluded from this metadata-only check). The first test command used a
  nonexistent filename and ran no tests; the corrected command passed.

### T10 — staged CUDA run (registered before submission)

- Hypothesis: a separate training session reaches GRPO sooner and preserves
  adapters/checkpoints even if the combined training/evaluation session expires.
  The original session finished 150 SFT steps but still had no first complete
  200-episode game after 12,831 seconds. This is elapsed progress, not a score.
- Change: build a private `-train-only` kernel with the existing stage flag;
  retain the original running kernel and its frozen source. Save both adapters
  and optimiser checkpoints, then evaluate each in separate Kaggle sessions.
  Architecture, data, learning rates and step budgets remain predeclared.
- Explicitly pass trainer/data seeds zero. The original receipt's seed zero
  described torch/adapter initialisation; original trainer seeds must be read
  from the exported TRL configs rather than inferred from that receipt.
- Count the two staged checkpoints as two additional final candidates (C=6),
  including unavailable candidates, before inspecting any new aggregate test
  score. Neither test evaluation chooses a CUDA checkpoint.
- Commands: `python train/build_kaggle_kernel.py --train-only`;
  `kaggle kernels push -p runs/kaggle_train_only --accelerator NvidiaTeslaT4`.
  Submission, privacy and runtime receipts will be kept in
  `results/training/kaggle_staged.json` and the pulled outputs.

### API availability scheduling

- Hypothesis: evaluating a slow first model delays discovering other responsive
  endpoints. The first resumed batch saved seven GPT-OSS-20B dice episodes over
  approximately fifteen minutes before its provenance restart. This is
  throughput evidence, not a comparison of policy scores.
- Change: finish the registered availability probes first, then evaluate every
  responsive model in increasing observed probe latency. Models, sample sizes,
  seeds, prompts and decoding stay fixed; selection never uses test PnL.
- Command: the existing three disjoint `scripts/expand_nim.py` batches.
  Per-model timings and outcomes remain in `results/expansion/protocol*.json`.

### Private checkpoint attribution

- Added the full Apache-2.0 licence from the pinned Qwen3-1.7B revision and a
  modification/quantisation notice to the earlier private PEFT dataset. Its
  adapter SHA-256 is unchanged; it remains the historical 300-iteration model.
- `kaggle datasets version -p runs/kaggle_dataset -m "Retain upstream Apache
  licence, attribution and checkpoint provenance" --quiet` succeeded.
  The official SDK confirms that the dataset remains private.
- [Upload, privacy and file-hash receipt](results/training/kaggle_dataset_notice.json).

### Dataset metadata correction and resource waiting

- Kaggle had stored the initial dataset licence display name as Unknown.
  Corrected it to its documented `apache-2.0` identifier, explicitly retaining
  `isPrivate: true`. The initial failure and corrected SDK checks are retained
  in the existing dataset notice receipt; the licence file/weights are unchanged.
- The heavy lock was released and immediately reacquired by another repo while
  our wrappers waited between 30-second checks. Retry acquisition every second
  instead; the atomic mkdir mutex and exclusive execution remain unchanged.
  Shell syntax verification passes. This is scheduling, not a policy trial.

- Apply the same one-second acquisition retry to NIM: both supplemental
  wrappers have waited over an hour, including observed intervening owner
  changes. This changes no API concurrency, request throttle or retry backoff.

### API latency diagnostic (no held-out scores)

- The initial partial GPT-OSS-20B diagnostic pooled 42 attempts across seven complete
  dice episodes, including nine cache replays. Its attempt-weighted median
  recorded latency was 12.006267070770264 seconds; p95 was
  77.82202217578886 seconds. These exclude in-flight/unsaved requests and are
  not batch wall time. No PnL was computed or inspected.
- [Self-contained metadata and calculations](results/expansion/latency_partial.json)
  support establishing availability before a slow model's full evaluation.

### Validation reference scores

- Commands: `mmb score --game dice --seeds 80000-80099 --results-dir
  results/validation/qwen35-sft` and the same command for cards. The first
  invocation omitted the seed argument, found no default-split episodes and
  wrote no result; the corrected commands use only the registered validation.
- Public-posterior Bayes reaches approximately 75% of Bayes on dice and 95% on
  cards. These are reference validation scores, not new learned-policy results.
  [Dice](results/validation/qwen35-sft/leaderboard_dice.json),
  [cards](results/validation/qwen35-sft/leaderboard_cards.json).

### Engine timing dependence check (registered before calculation)

- Hypothesis: adjacent timing pairs may share short-term machine load.
- Change: circular block bootstrap of the existing 200 paired ratios, retaining
  every sample and reporting both execution orders. Block lengths 1, 5, 10 and
  20; 50,000 resamples each, seed zero. Correct simultaneously for the three
  engine candidates and four block lengths. This is a robustness analysis, not
  a new performance optimisation or a new timed benchmark.
- Command: `python scripts/analyse_engine.py`. Result pending at
  `results/engine_trials/inline_matching_block_intervals.json`.

- Timing-dependence result: every registered block length retains a corrected
  lower bound above one. The lowest is 1.028462112694586 (block length five);
  old-first and new-first median ratios are 1.0511294533532733 and
  1.0495269513125054. Lag-one ratio correlation is −0.014392495517376416.
  All results, including the wider intervals, are retained in the result file.

### API latency replay correction

- Diagnosis: cache replays retain original server latency; pooling them with
  live attempts is unsuitable for estimating newly executed request time.
- Change: retain original pooled statistics, then filter `cached is False`.
  There are 33 live requests and nine cached attempts. Live-request median is
  13.250767230987549 seconds, p95 91.69077033996577 seconds. Failed/in-flight
  requests and other overhead remain excluded, so the sum is not batch runtime.
- Result and exact per-attempt metadata: `results/expansion/latency_partial.json`.
  No model configuration or PnL calculation changed.

### Future CUDA runtime persistence

- Diagnosis: a hard session timeout may bypass Python's final report.
- Change: future builds atomically persist elapsed time, phase/step and peak
  allocated/reserved memory at trainer log callbacks as well as shutdown.
  Include CUDA and additional training-stack package versions. Current private
  kernel snapshots are unchanged; these hooks have not yet executed on CUDA.
- Verification: Ruff, five CPU training/export checks and the reward dry run
  pass. CUDA callback execution remains to be verified on a future resume.

### Reference reuse outside the heavy lock

- Copy the exact completed reference cohorts into the remaining DAgger/RLOO
  validation folders and frozen-final destination. Verify seeds, complete rounds
  and identical SHA-256 values. Thirty destination files reuse existing data;
  zero new episodes are executed or counted.
- This prevents deterministic baseline recomputation while a model holds the
  heavy-ML lock. [Source/destination manifest](results/reference_reuse.json).

### Preserve API evaluation retries

- Diagnosis: a resumed API batch replaced a previous game's error receipt with
  its latest outcome, losing the failure and its elapsed time.
- Change: append every future game attempt to the protocol's history, including
  elapsed time on failure, while retaining the latest outcome for compatibility.
  Existing cached responses and completed episodes remain reusable. Already
  running Python processes retain their original recorded source revision.
- Verification: Ruff and Python compilation pass. This changes provenance only,
  not prompts, seeds, API concurrency or candidate selection.

### Reuse identical greedy CUDA evaluation prompts

- Hypothesis: repeated public states across episodes cause redundant sequential
  decoding. Cache exact prompts, quotes and repair attempts within one frozen
  adapter's evaluation. The cache is discarded before evaluating other weights;
  every episode and its original state transition still executes.
- Change: share the stateless CUDA quoter across episodes and tag replayed
  quotes. Greedy decoding, token limits, parser and repair policy are unchanged.
  Running private kernel snapshots are unchanged.
- Verification: CPU evaluation/export and cached repair isolation checks pass
  ([receipt](results/training/cuda_cache_verification.json)).
  No GPU runtime improvement is claimed before a future CUDA measurement.

### Clarify reward-training state distribution

- Source review confirms that both reward scripts use contexts collected along
  Bayes-teacher trajectories. Their sampled completions come from the learned
  policy, but subsequent training states are not its own trajectories. Document
  this limitation explicitly; DAgger separately collects on-policy contexts.
  No experiment or outcome changes.

### GLM low-effort availability result

- The separately registered low-effort GLM-5.3-Flash probe also failed after
  three HTTP 504 responses, with 923.1078577041626 seconds elapsed. Configuration,
  exact error and source hashes are retained in
  [protocol-reasoning.json](results/expansion/protocol-reasoning.json).
- Rechecked [NVIDIA's reasoning controls](https://docs.nvidia.com/nim/vision-language-models/latest/get-started/advanced/get-started-glm-5-3-flash.html):
  thinking is always enabled and `low` is the smallest documented effort.
  The failure therefore does not establish poor quoting quality. The separate
  DeepSeek non-thinking probe remains pending.

### Back up incomplete evaluation cohorts

- Commit the 57 complete 1.7B dice episodes and seven complete GPT-OSS-20B
  dice episodes already written before the current queues. Verify unique seeds
  and six rounds per episode without aggregating scores. These remain incomplete
  cohorts and do not satisfy the final sample requirement.
- [Recovery checkpoint](results/recovery_checkpoint.json) records each snapshot's
  prefix hash and producer commits. Resumers skip these seeds; later appends
  change the full-file hash without invalidating this historical snapshot.

### DeepSeek non-thinking availability result

- The registered DeepSeek-v4.1-Flash retry with `enable_thinking: false` also
  failed after three HTTP 504 responses, taking 921.8056712150574 seconds.
  [Exact configuration/error](results/expansion/protocol-reasoning.json).
  No score or negative-quality conclusion is assigned to this endpoint.
- The reduced-reasoning worker exited and released its slot. Remaining API
  workers continue waiting for the shared slots; no concurrency limit changed.

### Raw-data exposure audit correction

- A broad reasoning-parameter search printed raw API episode lines while
  locating configuration metadata. No aggregate policy score was calculated;
  no policy, seed choice or selection rule changed in response.
- Correct the recovery receipt's imprecise `scores_inspected: false` field to
  distinguish uncomputed aggregate scores from exposed raw episode fields.
  Preserve the earlier commit. Further metadata searches name protocol/receipt
  files explicitly. The registered training recipes remain unchanged.

### Recovered live CUDA progress

- Chrome accessibility reads recovered on a fresh private-kernel tab. The
  original session has completed 200 warm-start dice episodes, with a printed
  evaluation time of 9,760 seconds. Filter its mean PnL before tool output;
  aggregate policy scores remain withheld during validation selection.
- The staged run is executing SFT, with its latest visible log at epoch 1.467,
  training loss 0.05815 and 5,455.0 seconds. These are training diagnostics, not
  validation performance. The UI reports two allocated T4s and one trainer GPU;
  its effective SFT batch is 16 (device batch eight, accumulation two).
- [Original observation](results/training/kaggle_retry.json) and
  [staged observation](results/training/kaggle_staged.json) retain displayed
  precision and distinguish log visibility from actual execution progress.

### New quota directive and newest-family fit probe

- Owner now requests useful GPU work until the available quota is consumed or
  its 2026-10-31 UTC reset. This overrides the policy stopping rule only for
  Kaggle. Existing useful sessions continue. The official account quota and
  per-session receipts are recorded in `results/training/kaggle_budget.json`;
  unknown billing timestamps remain null rather than inferred from submission.
- HF Qwen catalogue and pinned configs identify Qwen3.8 as the newest open
  general-purpose family, with 27B its smallest size. The Apache-2.0 Unsloth
  NF4 file is 22,344,833,801 bytes. Two-T4 aggregate capacity may permit it;
  weight size alone does not establish training fit. Register one SFT update
  followed by one GRPO update, sequence 1024, rank eight, balanced model split,
  no final-test evaluation, capped at one hour. See `qwen38_fit_plan.json`.
- Focused implementation: expose training seed, SFT accumulation and device map;
  retain previous seed zero and accumulation two defaults. Record physical GPU
  capacity in runtime receipts. Command: `.venv/bin/python -m pytest
  tests/test_training.py -q`; result: 13 passed in 4.80 seconds. Ruff passed.
- Research errors retained: the unsharded checkpoint has no index JSON (404);
  an initially unauthenticated SDK quota call returned 401, corrected by using
  the official CLI authentication path. Metadata `last_run_time` predates known
  submissions by three days and changes with query time, so it is excluded from
  runtime accounting.
- Audit correction: an automatic browser snapshot displayed the fixed CUDA
  warm-start dice aggregate early. No training or selection choice changed in
  response. Further notebook snapshots redact URL queries before output; no
  session query value is stored in repository artifacts. See recovery receipt.

### Dual-T4 fit probe submission and concurrency check

- First push failed with HTTP 409 (title/slug conflict). Correcting the slug
  reached the account limit: `Maximum batch GPU session count of 2 reached.`
  CLI exit zero does not imply submission. Both errors are retained in
  `results/training/qwen38_fit_submission.json`; neither consumed a new run.
- Imported the same bounded source through a private interactive notebook.
  Verified Private selected, internet on, GPU T4 x2. Run All was granted:
  UI reports three active events and a running draft with two GPU indicators.
  This confirms two batch plus one interactive session concurrently.
- The staged 4B session has saved GRPO checkpoint 100. Training reward variance
  is nonzero, unlike the cold one-update smoke. These UI diagnostics are
  provisional; downloaded logs and frozen paired evaluation determine results.

### Prepare staged SFT versus GRPO validation

- Hypothesis: SFT provides formatting and scale; nonzero-variance exact-EV GRPO
  should improve decisions beyond that warm start. Before any recipe adjustment,
  compare both saved policies on the fixed 100 validation episodes per game.
- Command: `train/build_kaggle_kernel.py --evaluate-staged`. The generated private
  kernel mounts the completed training output and evaluates one frozen adapter
  per T4, using identical seeds 80,000–80,099. No held-out-test selection.
  Plan: `results/training/kaggle_staged_validation_plan.json`.
- Correct automatic adapter discovery to accept Unsloth's actual base-model ID
  as well as the upstream ID. Require an explicit path when several compatible
  adapters exist, avoiding arbitrary checkpoint selection. The prepared worker
  paths select `warmstart_adapter` and `lora_adapter` explicitly.

### Qwen3.8 fit probe 1 — failed loader placement

- Result downloaded from private Quick Save version one:
  `results/kaggle/qwen38-fit-v1/runtime.json`. Script wall time was
  245.16056537628174 seconds; two T4s each expose 15,636,037,632 bytes.
  No SFT or GRPO update executed. VRAM peaks are loader-preflight allocations,
  not the memory required by a loaded model.
- Exact failure: `does not fit in GPU memory ... transformers placed part of it
  on the CPU, which bitsandbytes cannot quantize`. The log also reports
  `Using float16 precision for qwen3_5 won't work! Using float32.` The script
  failed before substantive weight allocation, so this is a loader placement
  failure, not a measured training OOM. Stop the idle interactive allocation
  after preserving its files. Account quota includes its startup/idle interval;
  script runtime alone does not represent the billed duration.
- One focused repair is registered: use the loader's suggested
  `offload_layers="auto"` to stream layers from host RAM. Keep all other probe
  settings and its 3,300-second cap. `qwen38_offload_plan.json` records the trial.
- Conditional fallback research also records the Apache-2.0 Qwen3.6 35B-A3B
  upstream/Unsloth catalogue. Its available checkpoint is 71,903,776,776 bytes
  before runtime quantization, not a prequantized fit claim. No fallback run
  has been selected from a policy score.

### Qwen3.8 fit probe 2 — host streaming started

- Change only the loader placement policy with `--offload-layers auto`.
  The private draft retained T4 x2 and internet settings; import the new bounded
  notebook and request Run All. Exact source and notebook hashes, revision,
  seed and launch time are in `qwen38_offload_plan.json`.
- Preserve the first output snapshot before restarting. The retry writes a
  durable combined `probe.log`; the first snapshot contains its exact exception
  in runtime JSON but its console-only precision warning is a UI observation.
- No held-out evaluation runs in either feasibility probe. Both are training
  infrastructure trials, not evidence of policy improvement or exhaustion.

### Preserve data partition when staging GPU training

- A staged SFT adapter must skip SFT execution without resetting the registered
  SFT state count: that count also defines the disjoint GRPO training partition.
  Remove the argument mutation and test the execution condition directly.
- `--grpo-steps 0` now saves the SFT adapter and returns, enabling bounded SFT
  stages before a later GRPO session. This avoids TRL interpreting zero maximum
  steps as a request to run its default number of epochs.
- Verification: 13 CPU training checks passed in 3.38 seconds; Ruff passed.
  No active immutable Kaggle source changes and no new quality trial starts.

### Qwen3.8 fit probe 2 — failed offload compatibility

- Downloaded runtime and durable log: `results/kaggle/qwen38-fit-v2/`.
  Script wall time: 212.26065254211426 seconds. No weights/update completed.
  Exact diagnostic says automatic load-to-host requires every layer on one GPU.
  Preserve Quick Save version two, then stop the idle interactive allocation.
- Inspect the exact PyPI wheel used by Kaggle, verified by SHA-256. The
  multimodal loader's single-device guard precedes native block-swap planning;
  generic `balanced` plus load-to-host cannot reach the streaming planner.
  `qwen38_loader_inspection.json` records source locations and wheel hash.
- Focused next check: `device_map=unsloth_balanced`, without layer offload,
  retaining the same model, revision, rank, sequence and one-update objectives.
  This native planner can diagnose the head/embedding memory requirements
  precisely. It is not a policy-quality trial or a declaration that every
  possible implementation of 27B QLoRA is infeasible.

### Qwen3.8 fit probe 3 — native planner allocated

- Hypothesis: the native head-aware device mapper may place the large untied
  embedding and output head where generic balanced placement failed. Change
  only `--device-map unsloth_balanced`, remove the unsupported load-time offload
  flag, retain the one-SFT/one-GRPO-update objective and 3,300-second cap.
- Command: `.venv/bin/python runs/build_qwen38_probe.py`, then import its private
  notebook and Run All on T4 x2. Source/notebook hashes, pinned revision, seed
  and the approximate launch window are in `qwen38_native_plan.json`. Exact
  allocation start is unknown; UI confirms a running two-GPU draft at 18:08 UTC.
- Result: running; no fit or policy-quality claim yet. The account snapshot
  records 46,317.750142 of 108,000 GPU seconds used at 18:09 UTC. Script runtimes
  remain distinct from account billing in `kaggle_budget.json`.

### Freeze completed Lightning dice cohort and prepare paired validation

- `scripts/expand_nim.py` completed 200 unique dice episodes, seeds 200–399,
  six rounds each. Preserve raw episodes plus a hash/provenance receipt in
  `results/expansion/lightning_dice_cohort.json`. No aggregate score is used for
  tuning or selection; cards and the remaining API cohorts continue.
- The next staged-adapter validation launcher installs its shared dependencies
  before starting its two workers, avoiding concurrent package mutations.
  Workers still record their exact installed versions and individual runtime.

### Qwen3.8 fit probe 3 — successful SFT and GRPO updates

- Authoritative downloaded runtime and log contradict the stale prior error
  initially shown in the UI. Both registered updates completed in
  840.3186721801758 seconds; allocated peaks were 11,994,392,064 and
  12,017,976,832 bytes. Files: `results/kaggle/qwen38-fit-v3/`.
- The native planner splits about 20.487 GiB of loaded weights evenly and
  explicitly places the output head on GPU 1. Some layers remain bfloat16,
  despite float32 training computation. The earlier conditional all-F32 memory
  arithmetic is a bound, not the measured footprint. GRPO reward standard
  deviation is 0.7225 and no completion was clipped; this is a real update,
  not a policy-quality evaluation.
- Preserve private Quick Save version three before stopping the completed
  allocation. Its UI title mistakenly says failure; source-hashed JSON/logs
  establish success. Record that mistake explicitly in the receipt.

### T10 — newest-family Qwen3.8 SFT → GRPO (registered)

- Hypothesis: clone Bayes quotes with completion-only loss, select by teacher
  validation loss, then improve exact expected-value rewards with GRPO.
  Earlier CUDA SFT trained on full prompt text; the new objective supervises
  the answer directly and does not spend its loss on copying the prompt.
- Command: `train/build_kaggle_kernel.py --qwen38-train`; private interactive
  notebook, two T4s, pinned Qwen3.8-27B NF4 revision, rank 8, seed 0, 1,024-token
  context. Up to 800 SFT updates; 48 fixed teacher validation states, evaluated
  every 50 updates, patience two, restore minimum validation-loss checkpoint.
  Then 96 GRPO updates with two completions per group. No final evaluation
  runs in this training allocation. Plan: `qwen38_pipeline_plan.json`.
- A four-hour wall cap and checkpoints every 50 updates bound quota exposure.
  Resume optimizer state if the cap interrupts a useful stage; reaching the
  cap is not quality exhaustion. Policy validation remains 100 episodes per
  game on seeds 80,000–80,099, independently of teacher-loss selection.
- Before execution or new final scores, increase the candidate family from
  six to eight, counting both new SFT and GRPO checkpoints. Infrastructure
  attempts remain logged separately; they do not justify a superiority claim.

### T10 launch and complete Lightning API pair

- Import the source built at commit `4e9eda3` into the private interactive
  notebook. Run All requested at 18:39 UTC; the UI confirms two T4s and internet.
  Runtime and source hashes are in `qwen38_pipeline_plan.json`. No policy-test
  data is evaluated in this allocation. All three permitted GPU slots are active.
- The Lightning API batch now has 200 complete episodes in each game, seeds
  200–399, six rounds each. Preserve both raw files and their hashes in
  `results/expansion/lightning_cohorts.json`; aggregation remains deferred.
  The sequential NIM worker continues with Ultra, thinking disabled.

### Keep explicit SFT checkpoint resumes executable

- Inspection before any wall-cap continuation found that loading a compatible
  adapter also skipped SFT, even with an explicit `--sft-resume` checkpoint.
  Let an explicit optimizer-state resume take precedence over the warm-adapter
  skip. Retain the exact SFT data partition and total update budget.
- This change affects future resumptions only. The active newest-family run
  starts from the base model, and the two original batch snapshots are immutable.

### Make quota accounting directly reproducible

- Replace the repeated ignored quota-inspection helper with the committed
  `scripts/kaggle_quota.py`. It calls the official SDK, records the exact
  command/source hash/commit, and appends the raw quota snapshot to the ledger.
  It never serializes credentials, signed output URLs or full SDK objects.
- Account debit remains authoritative. Unknown per-session allocation start/end
  is kept unknown, rather than substituting unreliable `last_run_time` metadata
  or treating downloaded script duration as billed session duration.

### Isolate paired validation compiler caches

- Give each evaluation worker its own working directory as well as its own
  visible T4 and result directory. Unsloth generates Python/compiler cache
  files relative to the working directory; two workers must not write the same
  generated modules concurrently. Shared dependency installation remains serial.
- Rebuild and compile-check the prepared validation launcher. No policy, seeds,
  training recipe or active immutable source changes.

### Staged Qwen3.5-4B — full SFT and GRPO completed

- Downloaded primary results, configs, checkpoint histories and complete log
  from private version one into `results/kaggle/staged-v1/`. SFT reaches 150
  updates; GRPO reaches 300. Script wall time 17,983.032301425934 seconds;
  peak allocated 6,326,962,176 bytes on GPU 0; GPU 1 allocation remains zero.
  Completion receipt: `results/training/kaggle_staged_completion.json`.
- GRPO's last logging interval has reward standard deviation
  0.12598861753940582 and nonzero gradient norm. This establishes substantive
  execution, not held-out policy improvement. Frozen paired validation is next.
- The bulk optimizer/adapter download continues resumably in ignored runs/;
  slow weight transfer does not block remote evaluation of immutable outputs.
  Primary receipts/results are downloaded and committed before next submission.
- Download a live Qwen3.8 runtime through the notebook's output-file action:
  update 70, SFT phase, 1,724.92853140831 seconds, no memory failure. Retain it
  explicitly as a progress sample, not a completed run or a policy score.

### Paired validation launched; newest-family teacher loss improving

- Private validation version one accepted, T4 requested, internet on and
  remote privacy verified. Execution receipt: `kaggle_staged_validation.json`;
  its monitor waits 610 seconds between status requests. The source references
  immutable staged-training outputs, and runs one frozen policy on each T4.
- Download the active newest-family training log through Kaggle's file action.
  The first two printed teacher validation losses decrease from 0.3883 to
  0.3473. Retain the complete unedited progress snapshot and download receipt in
  `results/kaggle/qwen38-training-v4/progress/`. These are teacher-loss diagnostics,
  not policy PnL; exact final checkpoint history remains pending.
- Both local MLX queues still wait for the live foreign heavy-ML owner. No
  competing local model training or inference starts while memory is occupied.

### T9 local chunk-parallel memory probe — guard stopped before update

- After the shared lock became free, the queued probe ran at 19:16 UTC.
  The physical-footprint guard stopped it after 60.59448858399992 seconds,
  sampling approximately 7,633,633,280 bytes. No optimizer update completed.
  Preserve `qwen35_parallel_memory_guard.json`, the last training callback
  plus post-run guard outcome in `qwen35_parallel_smoke.json`, and the full log.
- The conditional full-SFT launcher correctly did not start. The 1.7B frozen
  final evaluation acquired the heavy-ML lock and resumed missing episodes.
  This failed infrastructure attempt is not a policy nonimprovement.

### T11 — avoid the masked vocabulary-head allocation (registered)

- Hypothesis: projecting hundreds of masked prompt positions to a 248k-word
  vocabulary retains large, unnecessary training tensors. Keep the full causal
  backbone, then project only the final 128 positions. Preserve MLX-LM's exact
  loss mask, including its boundary convention.
- A batch guard rejects any supervised token outside the retained window.
  CPU checks compare loss and every parameter gradient against the full causal
  model, with tied and untied heads; another test rejects dropping one label.
  Sixteen training tests pass. Source: `train/mlx_answer_loss.py`.
- Add `--answer-head-tokens 128` to the otherwise unchanged two-microbatch
  probe, under the heavy-ML lock and 7 GiB physical guard. Require one verified
  optimizer update before enabling the full 800-iteration run. Registration:
  `results/training/qwen35_answer_probe_plan.json`. No held-out policy tuning.

### Local memory repair queued; staged adapters retained

- The complete local suite now passes 95 tests. T11 waits under the shared
  heavy-ML mutex behind the fixed 1.7B final evaluation; the full 4B run is
  conditional on a completed, guarded optimizer update.
- The large optimizer download failed with an incomplete-read error after its
  retries. A targeted resumable download succeeded for both staged adapters.
  Safetensors headers contain 256 tensors each; exact byte counts and SHA-256
  hashes are retained in `kaggle_staged_adapter_download.json`. Immutable Kaggle
  outputs retain the optimizer states. No training or evaluation is repeated
  because of the transport failure.

### Newest-family context audit — no truncated training state

- Run `python train/audit_kaggle_tokens.py` with the exact pinned tokenizer,
  without loading weights or consuming GPU quota. The fixed 800 SFT states,
  4,000 disjoint GRPO states and 48 teacher-validation states all fit their
  context budgets. Maximum GRPO prompt: 902 tokens against 928; maximum SFT
  target: 965 against 1,024. No training recipe changes.
- Result: `results/training/qwen38_token_audit.json`, including exact command,
  source hash, model revision and producing commit. This rules out context
  truncation as a cause of a later quality gap; it does not establish quality.

### B1 — fixed-group GPU evaluation (registered, GPU measurement pending)

- Bottleneck: the downloaded newest-family fit log takes 128.4 seconds for a
  two-completion GRPO update. Serial token generation threatens the remaining
  evaluation budget. Batch four independent episodes at each round instead.
- Share the original market mechanics through an observation/quote generator;
  do not duplicate fill, ledger or expectation calculations. Fixed seed groups
  and atomic whole-group writes preserve deterministic ordering and resumability.
  Greedy decoding, the prompt, parser and one repair retry remain the protocol.
- Before GPU use, all 100 tests pass. Separate comparison with the previous
  runner gives exact equality of every episode field for 100 baseline episodes,
  covering both games and all five baselines. Evidence and the complete check
  source: `results/training/batch_mechanics_parity.json`.
- Command: `python train/build_kaggle_kernel.py --qwen38-evaluate`; then private
  validation allocation. `--evaluation-split test` generates the later frozen
  200-episode cohort. Qwen3.8 adapters are evaluated serially across both T4s;
  staged 4B adapters use separate T4s. Each allocation has a three-hour cap.
- No GPU throughput, numerical equivalence or memory improvement is claimed
  from CPU tests. Validate batch size four on seeds 80,000–80,099 before the
  held-out test; register the actual inference configuration with its receipts.
  The eight-candidate family remains conservative; no extra test candidate is
  selected by these infrastructure checks.

### Newest-family progress sample at update 300

- Downloaded live runtime and complete log at 20:12 UTC: SFT update 300,
  5,386.978399038315 seconds elapsed, peak allocations 12,005,684,736 and
  12,040,838,656 bytes. The latest completed teacher-validation loss in this
  snapshot is 0.2949. This is still an active training diagnostic, not policy PnL.
- Preserve the unedited files and hashes in
  `results/kaggle/qwen38-training-v4/progress/step300-download.json`.
  The official account snapshot reports 65,537.912747 seconds used; remaining
  quota is 42,462.087253 seconds. Account debit and script duration stay distinct.

### Future GPU receipts include execution timestamps

- Add UTC execution start, snapshot and finish timestamps to future runtime
  JSON. These describe the script process, not Kaggle allocation start/end.
  Existing source snapshots and unknown allocation times remain unchanged.
- Verify with the 17 CPU training tests and lint. The scheduled active-run
  diagnostic still reports SFT update 350 with no error; it predates this change.

### Protect batching validity and reserve useful newest-family evaluation

- Before any batched validation/test cohort, compare serial and batched parsed
  quotes on eight fixed training states: dice/cards, seeds 20,000–20,003, rounds
  0/1/3/5. Preserve responses and timings in `batch_parity.json`; differing
  quotes stop evaluation for diagnosis. Serial timing includes warm-up and is
  explicitly not a steady-state speedup estimate. All 102 CPU tests pass.
- Prepared conditional continuation plan: `qwen38_continuation_plan.json`.
  Only if the current 96-update GRPO stage is unfinished, admit up to 32 further
  updates subject to actual quota and evaluation reservations. Resume optimizer
  state when available; otherwise explicitly restart from frozen SFT weights.
  The single-update 128.4-second probe includes compilation, so it is only a
  conservative scheduling diagnostic. This is not a quality-stopping trial.
- Future GRPO jobs retain two rolling optimizer checkpoints. This permits
  frequent checkpointing without retaining every large optimizer snapshot;
  it changes neither the reward nor the active immutable training run.

### T3 follow-up — Ultra medium-effort thinking (registered)

- The original thinking-on Ultra probe failed with HTTP 504. The official
  [Ultra API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b)
  documents `medium_effort=true` alongside `enable_thinking=true`, and recommends
  that mode before explicit reasoning-budget tuning. Its budget example is a
  separate client-side continuation; do not assume an undocumented hosted flag.
- Queue `bash scripts/with_resource.sh nim python scripts/expand_nim.py --ultra-medium`.
  Keep the original timeout, output ceiling, sampling temperature and retry count;
  this is a separate labelled configuration and protocol file. Run 200 episodes
  per game only if the availability probe returns a usable quote.
- Plan and source: `results/expansion/ultra_medium_plan.json`. Configuration is
  chosen from timeout evidence and official docs, without inspecting new test
  scores. The shared two-slot bound and one request/client remain unchanged.

### T2 frozen final cohorts completed

- Both games now contain 200 unique seeds, 200–399, with six rounds each.
  The frozen model revision and adapter hash are constant across all resumed
  records. Exact producing commands, commits and output hashes are retained in
  `results/final/qwen17-sft800/cohorts.json`.
- Commit every episode and descriptive per-game leaderboard. Defer the final
  corrected API comparison until its cohorts are complete; no held-out score
  changes a subsequent training recipe. The heavy-ML mutex is released and
  T11 begins its registered two-microbatch memory probe.
- New availability probes use validation seed 80,000, rather than historical
  seed zero, and record their probe seed/round. Already recorded probes and
  active workers retain their original source snapshot and outcomes.

### T11 completed; full local SFT restarted

- Two microbatches complete one optimizer update in 100.53891158400074 seconds;
  guard wall time 103.18615537499863 seconds. Peak MLX allocation is
  4,538,984,408 bytes; sampled approximate physical footprint reaches
  7,319,060,480 bytes, below the 7 GiB guard. These are different memory measures.
- The probe uses the longest training examples (987 and 975 tokens, rounded to
  the trainer's bucket). Compare all 248 saved LoRA tensors with the initial
  adapter: 124 change, confirming a nonzero update. Preserve the completed
  runtime, memory guard, full log and `qwen35_answer_probe_verification.json`.
- The conditional full run starts from the unchanged initial step-zero adapter,
  not the probe's trained weights. It uses 800 microbatches, accumulation two,
  all 480 validation examples, seed three and the same 7 GiB guard. Its eventual
  best checkpoint is selected by validation only. No policy improvement is yet
  claimed, and the probe's update is not added to its iteration count.

### T3 expansion — second complete API pair retained

- Ultra with thinking disabled now has 200 unique seeds, 200–399, and six
  rounds in each game. Preserve every episode and cohort hashes in
  `results/expansion/ultra_cohorts.json`; compute no aggregate during tuning.
- Correct the unexecuted medium-effort draft's temperature from zero to the
  existing client value 0.3, and make that value explicit in future batches.
  Active requests already used 0.3. The separate medium-effort retry remains
  queued under the shared slot rule. These CIs will be conditional on the
  recorded API samples and trained checkpoints, rather than uncertainty across
  training seeds or repeated stochastic generations.

### Preserve the renamed newest-family kernel identity

- Kaggle changed the training kernel slug when its title changed. The native
  editor, official owner listing and pulled private metadata agree on
  `makemarketbench-qwen3-8-27b-training`, numeric ID 141529807. Historical probe
  references remain historical; future mounts use the verified current slug.
- Record the alias and verification commands in `qwen38_kernel_rename.json`.
  The saved version still predates the active training draft: save its completed
  outputs and verify their hashes before any continuation or evaluation mount.

### T12 conditional continuation is executable

- `train/build_kaggle_kernel.py --qwen38-continue` prepares a private kernel;
  it remains unsubmitted while the useful original training session continues.
- Copy saved checkpoints into their original working paths, preserving the
  trainer's absolute best-checkpoint references. Reject incomplete optimizer
  checkpoints and any mismatch between the saved runtime and core source hash.
- SFT continuation keeps its original 50-update validation/save cadence. A
  separate GRPO command saves every five updates, with at most 32 additional
  updates and a combined 6,300-second launcher cap. Record an optimizer restart
  from SFT when no complete GRPO checkpoint exists. A resumed cosine schedule
  uses the explicitly bounded new ceiling; it is not labelled uninterrupted.
- Preserve the original runtime and every continuation stage's runtime/log.
  CPU checks cover both SFT-resume and already-completed-SFT paths, and reject
  partial checkpoints. GPU behavior remains unmeasured; admission still requires
  a fresh quota check and retained evaluation budget.
- Pre-submission review also found that a mounted training run contains several
  compatible checkpoint adapters. Select the same complete SFT checkpoint
  explicitly for both PEFT loading and optimizer resumption, avoiding ambiguous
  auto-discovery. The targeted continuation checks pass before any GPU use.

### Newest-family progress sample at update 500

- Downloaded runtime reports SFT update 500 and 8,484.44287776947 seconds elapsed,
  without an error. Preserve the raw log and source-matched runtime under
  `results/kaggle/qwen38-training-v4/progress/step500-*`.
- The log's completed teacher-validation checks improve to 0.2818 at update 400,
  then 0.2832 at update 450. These are training diagnostics, not policy PnL;
  neither a policy improvement nor a stopping decision is inferred from them.
- One native progress check at 20:53 UTC was scheduled less than ten minutes
  after the 20:45 check. Retain that error in the download receipt and enforce
  the interval from the actual UTC download timestamp thereafter. CLI batch
  monitors continue at 610-second intervals.
- Audit the deployed source, rather than the current repository alone: its
  GRPO configuration predates rolling retention. The launcher now preserves
  that exact core and records an asserted two-line patch: retain two GRPO
  checkpoints and identify the continuation's producing commit. Both complete
  source files and hashes survive in outputs. The added test covers resuming
  GRPO optimizer step 20 to the registered ceiling 52, as well as starting from
  SFT; no GPU result is inferred from it.

### T11 full-model validation parity

- The full 480-example validation pass at the unchanged initial adapter reports
  2.1129567623138428, exactly the same loss as the pre-reboot implementation.
  The model revision and both dataset hashes match. The new pass takes
  1,493.2317288750019 seconds; this is not a controlled timing comparison.
- Retain the complete initial validation snapshot and calculation source in
  `results/training/qwen35_answer_full_validation.json`. This strengthens the
  loss-preservation evidence beyond the small gradient tests; it is not a
  trained-policy improvement. The 800-microbatch training stage continues.

### T11 full training interrupted; T13 bounds unused MLX cache

- After the full initial validation, the guard observes 7,522,484,224 bytes,
  approximately 6,291,456 bytes above its 7 GiB threshold, and terminates the
  process. Preserve the guard, raw final report, corrected failure receipt and
  full log under `qwen35_sft_answer_attempt1*`. Training began, but no ten-step
  report exists: the actual completed update count is unknown, not zero.
- The saved best adapter remains byte-identical to the initial adapter. This
  infrastructure failure does not count as a policy nonimprovement.
- MLX documents that its free-cache limit defaults to the memory limit. Bound
  unused cached allocations to 256 MiB, without altering active tensors, the
  loss or the 7 GiB physical guard. Report active/cache memory separately.
- Register an eight-microbatch, four-update probe on the same longest examples,
  followed only on success by the original full 800-microbatch recipe from
  step-zero weights. `qwen35_cache_probe_plan.json` retains settings and host
  diagnostics. The foreign live heavy-ML owner has the mutex; wait normally.

### T10 staged CUDA validation completed

- Both frozen adapters complete 100 unique episodes per game, validation seeds
  80,000–80,099. Pair wall time is 6,555.098810173 seconds; both exit codes are
  zero. Preserve raw episodes, runtime, source and full log under
  `results/kaggle/staged-validation-v1/`, with hashes in
  `kaggle_staged_validation_completion.json`.
- SFT dice: 86.16444508138923% of Bayes, descriptive 95% CI
  [77.15045957210653, 94.2791714178511]. Cards expected PnL: 2.6100787313559777,
  CI [-3.6356169405684278, 8.270299340381493].
- GRPO dice: 76.76487646171897%, CI
  [64.72015836780598, 86.65049749516886]. Cards expected PnL: 7.568413040676241,
  CI [1.955406324895731, 12.777514395329854]. This is a tradeoff, not evidence
  that one method dominates both games. No test score informed this diagnosis.
- Queue the already registered frozen SFT/GRPO test pair, 200–399, with batched
  inference only after its train-state parity guard passes. Use a two-hour
  evaluation cap, reserving remaining quota for the newest-family pipeline.
- Private held-out version one is accepted and remotely verified. Exact launch,
  source hash, settings, and conservative quota reservation are in
  `results/training/kaggle_staged_final.json`. The status monitor sleeps
  610 seconds between checks. Both frozen adapter identities are unchanged.

### T13 cache-bounded probe passed; full training restarted

- Eight microbatches / four optimizer updates complete in 172.9254369170012
  seconds. Sampled approximate physical peak is 6,607,077,376 bytes, below the
  unchanged 7 GiB guard; MLX allocator peak is 4,058,164,994 bytes. The prior
  default free-cache limit was 16,320,875,724 bytes, now bounded to 268,435,456.
- All 248 adapter tensors change after four updates. Retain runtime, guard,
  complete log and weight-check source in `qwen35_cache_*` results. These are
  infrastructure measurements on the longest training examples, not policy PnL
  or a controlled throughput comparison with the earlier two-microbatch probe.
- The conditional launcher restarts full training from the unchanged step-zero
  adapter, discarding probe updates. Full training is not yet verified complete.
- The complete local suite passes 104 tests with no skips; exact producing
  commit, command, output and hash are in `results/verification.json`.

### T14 — exact greedy MLX response reuse

- Hypothesis: repeated states make serial local validation and DAgger collection
  needlessly regenerate the same greedy answer. On teacher-training data only,
  9,600 examples contain 5,579 unique prompts and 4,021 repeated occurrences.
  This is not a measured hit rate on future actor states.
- Change: keep one response cache per loaded frozen adapter and reuse its actor
  across episodes. Preserve the complete repair trace, including failures;
  mark reused responses explicitly. Bound unused inference allocator cache to
  the same 256 MiB used by the successful training probe.
- Commands: `.venv/bin/python runs/measure_mlx_cache.py`;
  `.venv/bin/python -m pytest tests/test_training.py -q`.
- Result: 23 targeted tests pass. CPU fake-generator checks cover successful
  repairs, repeated failures and adapter isolation. No local inference speedup
  or policy improvement is claimed. Source, data hash and calculation are in
  `results/training/mlx_greedy_cache.json`; this infrastructure change does not
  count as a substantive policy nonimprovement.

### T15 — completion-only local RLOO loss

- Hypothesis: the unexecuted reward trainer would waste memory projecting
  masked prompt positions, just as the original SFT implementation did.
- Change: project only completion positions, preserving float32 log-probability
  computation and the summed, advantage-weighted REINFORCE objective. Bound
  unused allocator cache to 256 MiB before model loading.
- The first parity check also exposes a pre-existing call to nonexistent
  `mlx.core.log_softmax`. The reference check now uses `mlx.nn.log_softmax`;
  the implementation uses cross entropy. No previous RLOO execution is claimed.
- Command: `.venv/bin/python -m pytest tests/test_training.py -q`.
- Result: the initial two reference checks fail on the nonexistent function;
  after correction, all 23 targeted tests pass. Tied and untied output heads
  match the full-projection objective and every parameter gradient. GPU-memory
  behavior and trained-policy performance remain unmeasured. This repair is
  not a policy nonimprovement.

### T16 — validation-selected game routing (registered)

- Hypothesis: the staged GRPO dice/cards tradeoff can be handled by routing the
  known game to a separately frozen LoRA adapter on the same 4B base model.
- Change: choose the larger validation mean percentage of Bayes independently
  by game; the completed 100-episode validation chooses SFT for dice and GRPO
  for cards. SFT wins an exact tie. No held-out aggregate has been inspected.
- Command: `.venv/bin/python runs/register_routed_policy.py`.
- Result: registration only. `results/training/staged_routing_plan.json`
  freezes both adapter hashes, validation hashes, routes and full registration
  source. Reuse the corresponding deterministic frozen-policy test episodes
  once complete, with explicit underlying-policy provenance. This is a
  two-adapter policy, not a single-checkpoint result. Increase the correction
  family from eight to nine before any new final score is inspected. Final
  PnL, CIs and target outcomes remain pending.
- `scripts/stitch_routed_policy.py` now materializes the registered cohorts
  without model calls or aggregate scoring. It checks all 200 seeds, complete
  episodes, both adapter hashes and the pinned revision; preserves original
  files and records underlying-policy provenance. CPU tests reject changed
  weights or a changed registration and verify idempotent reuse. All 27
  training tests pass after this and the uniform-padding checks.

### Computer-use outage during newest-family training

- After the downloaded update-500 progress sample, computer use fails with
  `Sky Computer Use native pipe startup failed`; browser access separately
  reports `The browser access host is unavailable`. Resetting and reconnecting
  do not restore access. The official SDK's draft-output request returns 404.
- Preserve these exact errors and the recovery attempts in
  `results/training/kaggle_ui_outage.json`. The latest phase and live draft
  outputs remain unknown. Do not replace the useful session or present its
  last progress sample as a completed run. Ask the owner to restore computer
  use while CLI batch monitors, local training and API evaluations continue.
- The complete local suite now passes 106 tests, with no skips, after the
  greedy-cache and RLOO repairs (`results/verification.json`).

### T17 — uniform training shapes (registered)

- The T13 full run also hits the unchanged guard: sampled process footprint
  8,065,646,592 bytes, versus the last reported MLX allocator peak
  4,032,126,054 bytes. Ten microbatches / five updates were reported; the
  actual final count after that report is unknown. Preserve the log, guard,
  untouched raw report and corrected failure under `qwen35_sft_cache_attempt1*`.
- Hypothesis: mlx-lm's compiled gradient step accumulates process memory across
  its many 32-token length buckets. The repeated-longest-example probe did
  not exercise those varied shapes. This explanation is not yet measured.
- Change: right-pad training only to 1,025 tokens, giving 1,024 model inputs.
  The audited maximum is 987 tokens, so no example is truncated. Gather the
  128-position vocabulary projection by each row's actual end position;
  padding must not discard an earlier answer. Keep the 7 GiB guard, seed,
  optimizer, training data and 256 MiB free-cache limit unchanged.
- CPU checks compare loss and every gradient with full causal projection on
  ragged examples, with and without extra trailing padding, for tied and
  untied heads. Padding preserves every input token and label offset, and
  rejects insufficient space. All 26 training tests pass.
- `results/training/qwen35_uniform_shape_plan.json` registers a 32-microbatch,
  16-update probe on the full unchanged training set, with only two validation
  examples. Discard its updates before the full validation-selected recipe.
  Neither this infrastructure failure nor its repair is a policy
  nonimprovement.

### T17 probe completion and full restart

- Command: the exact guarded command and source commit are retained in
  `results/training/qwen35_uniform_smoke.json`. The unchanged training order
  completes 32 microbatches / 16 optimizer updates in 394.0657104999991 seconds.
- Sampled process footprint peaks at 6,653,214,720 bytes; the MLX allocator peak
  is 4,145,553,086 bytes. The two-example validation loss falls from
  2.064978837966919 to 0.3982905149459839. This subset is an infrastructure
  check, not a policy-quality estimate or a controlled memory comparison with
  the previous full-validation failure.
- All 248 LoRA tensors change; maximum absolute delta is
  0.00855996459722519. Guard and weight-verification files are
  `results/training/qwen35_uniform_memory_guard.json` and
  `results/training/qwen35_uniform_probe_verification.json`.
- At 21:58:35 UTC, discard probe weights and restart the full 800-microbatch
  recipe from its original adapter, with all 480 validation examples and the
  same guard. Full completion and policy validation remain pending. The
  launcher source and command are retained in the uniform-shape plan.
- The routed-cohort addition initially fails CI on import ordering. Correct
  that single lint error; lint and all 110 local tests pass, with exact command
  and output in `results/verification.json` and `results/verification.log`.
- Queue the remaining registered local stages after full SFT and its complete
  100-episode validation in both games: collect 150 DAgger episodes per game,
  audit token lengths, train for up to 800 microbatches, validate, then run the
  24-group dice RLOO recipe and validate. Freeze the winner before fresh test.
  Every heavy stage takes the shared lock. DAgger retains answer-only projection
  and fixed training shapes, with the same 7 GiB guard; stop if its audited
  inputs exceed the fixed shape. The exact queued launcher is retained in
  `results/training/qwen35_followup_plan.json`; no follow-up outcome is claimed.

### Original combined CUDA run: platform limit and frozen-policy continuation

- The scheduled 22:21:51 UTC poll returns `CANCEL_ACKNOWLEDGED`; download its
  outputs with the exact command in `results/training/kaggle_retry.json`.
- The checkpoint verifies all 300 GRPO updates after 150 SFT updates. Both
  warm-start cohorts contain all 200 seeds; the GRPO dice cohort contains
  seeds 200–319 (120 complete episodes), with cards not started. Preserve all
  raw episodes without inspecting aggregate test scores.
- The platform termination prevents `runtime.json` from being written. The
  last downloaded log timestamp is 37,147.368898703 seconds; this is a lower
  bound on script duration, not an exact GPU billing time or VRAM measurement.
- Retain the immutable outputs, checkpoint histories, hashes and exact
  extraction source in `results/training/kaggle_original_completion.json`.
  Resume only missing episodes from the saved final GRPO adapter, with the
  same serial greedy decoding and 128-token limit, one independent game per
  T4. This continues a frozen candidate; it does not select a new policy.

### Original continuation input-source failure (retained)

- Kaggle accepts the push but reports the cancelled parent as an invalid kernel
  source. Its mismatched title also produces a different slug. The mounted-input
  assertion then fails before model loading or inference; the run reaches
  `ERROR`. Preserve its full downloaded log and submission in
  `results/training/kaggle_original_final_resume_attempt1.json` and
  `results/kaggle/original-resume-failure-v1/full.log`.
- Download the original final adapter directly, hash it, and create a private
  dataset containing that adapter, the 300-update trainer state and original
  partial cohorts. Include the pinned upstream Apache-2.0 licence, derivative
  attribution and a separate MIT licence for synthetic episodes. Verify the
  dataset is ready and private before retrying. Exact upload commands, hashes
  and privacy verification are in
  `results/training/kaggle_original_adapter_dataset.json`.
- The retry uses the actual slug and a dataset input. Verify every mounted file
  against the uploaded manifest before model loading. This is an infrastructure
  failure, not a policy nonimprovement or a reason to change the frozen weights.

### Original trainer-seed correction

- Inspecting the downloaded original SFT and GRPO configs reveals trainer seed
  3407, while their outer metadata hard-codes 0. LoRA initialization and prompt
  shuffling use 0; the trainer-specific seed was left at Unsloth's default.
  The separate staged run explicitly configures trainer/data seeds 0.
- Retain all raw files unchanged. Correct their interpretation in
  `results/training/original_seed_correction.json`, which records both config
  hashes, the original source hash and exact extraction command/source. The
  frozen-adapter continuation seed is an inference-process seed and does not
  imply retraining with seed 0.

### Staged held-out evaluation: GRPO complete, SFT parity safeguard triggered

- Download the terminal kernel outputs at the scheduled 22:44 UTC poll; retain
  the exact download command in `results/training/kaggle_staged_final.json`.
- GRPO passes serial/batched parsed-quote parity on eight training states and
  completes all 200 held-out episodes in each game. SFT fails that training-state
  check and stops before test evaluation. Keep the failure rather than treating
  batched inference as equivalent for this checkpoint.
- Parent wall time is 4,528.369066715 seconds; worker exit codes are `[1, 0]`.
  Raw cohorts, both parity files, both runtime files and the complete downloaded
  log are retained in `results/kaggle/staged-final-v1/`, with hashes and exact
  extraction source in `results/training/kaggle_staged_final_completion.json`.
- No held-out aggregate scores are inspected to change a policy. The focused
  fallback is the unchanged SFT adapter with serial greedy decoding. Check
  remaining quota before admitting it; prioritize the registered dice SFT route,
  since the registered cards GRPO route is already complete. This infrastructure
  safeguard is not a substantive policy nonimprovement.

### Frozen SFT serial fallback admitted within the remaining quota

- Do not change weights, decoding limits or selection rules. Split all 200 dice
  seeds into two disjoint 100-episode serial workers, one per T4. This preserves
  the registered dice SFT route while its cards GRPO route is already complete.
- The downloaded validation log measured the first 100 serial SFT dice episodes
  finishing at script timestamp 3,257.641215616 seconds. Request 3,350 seconds
  for this session, with a 3,300-second parent cap; retain partial outputs if a
  cohort takes longer rather than weakening the 200-episode criterion.
- Official quota before submission leaves 16,446.531049 seconds. The new session
  plus the original session's full remaining requested limit and continued
  interactive allocation through that deadline total 16,396.543836 seconds.
  Do not admit any further run until completed jobs release that allowance and
  a fresh quota snapshot establishes sufficient headroom.
- Command: `python runs/submit_staged_sft_serial.py`; its entire builder and
  submitted source, exact CLI commands, frozen adapter hash, admission
  arithmetic and private remote metadata are retained in
  `results/training/kaggle_staged_sft_serial.json`. The kernel is private and
  accepted. Status polling is every 610 seconds. Execution remains pending.

### Serial SFT wrapper syntax failure (retained)

- The first scheduled poll reports `ERROR`. The downloaded log shows
  `SyntaxError: unterminated string literal (detected at line 38)`: the local
  builder interpreted a newline inside the generated wrapper's string literal.
  No model loaded and no test episode ran.
- Preserve the exact submitted source, failed download and retention source in
  `results/training/kaggle_staged_sft_serial_attempt1.json`, with the full log
  in `results/kaggle/staged-sft-serial-failure-v1/`.
- Correct the one escaping error and compile the generated source locally
  before any retry. Recheck quota and use the identical frozen adapter and
  seed partitions. This is an implementation failure, not a policy outcome.

- Serial retry version 2 compiles locally before submission and is accepted
  with private input metadata verified. Its fresh admission check covers
  15,045.364986 seconds against 15,088.960270 seconds remaining. The corrected
  builder and submitted source are retained in
  `results/training/kaggle_staged_sft_serial.json`; the failed version remains
  separate. Status polling resumes every 610 seconds.

### Cross-backend held-out game comparability check

- Verify the completed CUDA, MLX and API cohorts against freshly generated
  canonical games using only seed, hidden value, round indices and public
  reveals. Do not inspect model scores during this check.
- Command: `.venv/bin/python scripts/check_cohort_games.py results/kaggle/staged-final-v1 results/kaggle/original-v1 results/final/qwen17-sft800 results/expansion/raw`.
- All 24 complete cohorts pass: each has the 200 distinct registered test seeds
  and six rounds per episode, with canonical hidden values and public reveals.
  Incomplete API/GRPO files remain listed separately rather than counted as
  completed. Exact command, source hashes and input hashes are retained in
  `results/final/game_comparability.json`. Ruff passes for the new script.

### Paired validation ablation diagnosis

- Compute episode-paired GRPO-minus-SFT differences on all 100 validation seeds
  in each game. Confirm their Bayes reference arrays are identical first.
- Command: `.venv/bin/python runs/staged_validation_ablation.py`. Exact source,
  raw-input hashes, bootstrap seed 0 and 10,000 resamples are retained in
  `results/validation/staged_ablation.json`.
- Cards expected PnL changes by +4.958334309320262, with descriptive paired 95%
  CI [2.0498365187466008, 8.044547142644168]. Noise edge rises 1.7099343093202617
  and adverse-selection cost falls 3.2484.
- Dice changes by -1.3725765147304103, CI [-2.290160461493702,
  -0.5298819195898281]. Noise edge falls 1.2243765147304095 and adverse cost
  rises 0.1482. Both stages have zero no-quote rates in both games.
- These are the same validation cohorts used for router selection, conditional
  on one training pipeline. Retain both directions; do not present these CIs
  as held-out confirmation or training-seed uncertainty. No new policy is
  selected from test results and this analysis consumes no GPU budget.

### GPU ledger distinguishes verified progress from live allocation

- The 23:11 UTC official quota snapshot records 95,207.853819 seconds used of
  108,000. Retain the reported reservation independently; it is not additional
  completed compute or free quota.
- The newest-family live draft remains inaccessible. Record its downloaded
  8,484.44287776947-second, step-500 SFT sample as a script-duration lower
  bound. Its current phase, final runtime and actual allocation release remain
  unknown. Correct the ledger's ambiguous four-hour session label: four hours
  is the registered script cap; Kaggle's platform session maximum is 12 hours.
- Record available script start/end timestamps separately from unknown GPU
  allocation boundaries. Retain exact maintenance source in the ledger.
  No extra kernel is admitted while the active-run allowance consumes the
  remaining quota. Latest completed audit/ablation CI remains green.

### T17 full-run progress: full validation before iteration 200

- The full 480-example validation pass scheduled before iteration 200 completes
  in 1,259.6607853330024 seconds. Validation loss falls from
  2.1129567623138428 to 0.4035864770412445.
- Inspect the upstream trainer's convention: it validates before the next
  training microbatch, displays 200, and passes callback iteration 199. These
  weights have 199 consumed microbatches and 99 optimizer updates; preserve
  those exact values rather than relabelling the callback as 200.
- The sampled physical-footprint peak remains 6,712,983,552 bytes under the
  unchanged 7 GiB guard. Preserve training/guard snapshots and the current
  best-adapter hash in `results/training/qwen35_uniform_validation199.json`.
- Command: `.venv/bin/python runs/retain_mlx_step200.py`; its complete source is
  embedded in that receipt. The registered recipe continues toward 800
  microbatches with validation early stopping. This is training progress and
  full-validation memory evidence, not episode-level PnL.

### Inference-group bootstrap sensitivity registered before final scoring

- Inspect only the training-state parity file. One SFT dice bid changes from
  22.0 to 22.3 under batching, while the other seven tested quotes match.
  The failed gate already prevented its test execution. This demonstrates
  that numerical batching cannot be assumed globally identical to serial
  decoding, even when a small GRPO parity sample passes.
- Retain the required episode bootstrap. Also resample complete inference
  groups for CUDA cohorts, using group means and common indices for paired
  reference/API comparisons. Serial cohorts retain their original intervals.
- Require cards positivity and paired API superiority to pass both the primary
  and inference-group family intervals, without choosing a method after seeing
  scores. Keep the nine-candidate correction and both interval sets. The
  sensitivity remains conditional on the frozen response cache and does not
  estimate fresh-cache or training-seed variation.
- Register this amendment in `results/protocol.json` before opening any new
  held-out aggregate. No policy or seed changes. A synthetic perfectly
  correlated four-episode-group example verifies that preserving dependence
  widens the bootstrap interval while retaining the same mean.
- Ruff and all 31 training/scoring tests pass; exact commands, source hashes and
  outputs are in `results/verification_inference_groups.json`.

### Frozen router accepts independently downloaded evaluation roots

- The registered router uses SFT dice and GRPO cards. The SFT batching gate
  forces a separate serial kernel, so its two source cohorts now live in
  different downloaded output roots.
- Allow multiple explicit input roots and require exactly one source for each
  registered route. Keep all adapter, revision, seed and source-hash checks;
  preserve raw source bytes and reuse episodes without new model calls.
- Verification: `ruff check scripts/stitch_routed_policy.py tests/test_training.py`
  and `pytest -q tests/test_training.py`; the existing frozen-source and
  changed-weight checks now exercise two distinct input roots.
- Correct the README reproduction command to include the fixed 1025-token
  training shape actually required by the bounded-memory local recipe.

### Audit the complete projected supervision window

- Command: `.venv/bin/python runs/audit_answer_width.py`; tokenizer only, no
  model weights or heavy-ML work. Retain exact source, model revision, data
  hashes and upstream dataset/trainer source hashes in
  `results/training/qwen35_answer_width.json`.
- All 9,600 training and 480 validation targets contain at most 68 answer
  tokens. The upstream inclusive mask also includes one boundary padding
  target, so its maximum required projected window is 69, below the configured
  128. Existing loss/gradient parity tests intentionally preserve that upstream
  convention. Do not change the objective or hyperparameters of the active run.
- The quota snapshot at 23:41 UTC reports 99,553.244329 GPU seconds used of
  108,000. Clarify the ledger admission rule: bounded batch limits and an
  explicit allowance for unknown continued interactive allocation are separate
  from billing reservations and the platform's 12-hour maximum.
- The official SDK's separate draft-status route also returns HTTP 404. Retain
  that exact failure. Direct browser selection still fails with the native
  pipe error. Exact timestamps of the two earlier CUA calls were not persisted;
  replace an approximate timestamp with the surrounding clock-read bounds.

### GPT-OSS completes the fresh dice cohort

- GPT-OSS-20B, low reasoning effort, completes all 200 dice episodes on seeds
  200–399. All seven decoding/endpoint settings agree across resumed rows.
  The canonical hidden values, public reveals and six round indices also pass.
- Commands: `python scripts/check_cohort_games.py results/expansion/raw/dice/openai_gpt-oss-20b.jsonl --out results/expansion/gpt_dice_game_check.json`
  and `python runs/retain_gpt_dice.py`. Exact source and hashes are retained in
  `results/expansion/gpt_dice_cohort.json`. Cards remains in progress; no new
  aggregate score is opened or counted as a complete two-game API comparison.
- The comparability command initially accepted only directories and found no
  files for a single-file input. Allow explicit files as well as directories;
  the real 200-episode cohort and Ruff now pass. Preserve this failed command
  as a utility-interface failure, not a policy-quality iteration.

### Serial SFT dice reaches its wall cap with 187 saved episodes

- The 3300-second script budget stops both serial workers: 92 episodes on
  seeds 200–291 and 95 on seeds 300–394. Retain every byte, both exact
  `TimeoutError: Registered training wall budget exhausted` errors, and the
  3300.733895038-second pair runtime. Peak allocated memory is 4,219,314,688
  bytes per worker, as recorded in the downloaded runtime files.
- The runtime estimate from validation was too tight for these test states;
  no score was inspected. This is an incomplete infrastructure outcome,
  not a substantive validation nonimprovement.
- Command: `.venv/bin/python runs/retain_sft_serial_partial.py`. Exact source
  and downloaded-file hashes are retained in
  `results/training/kaggle_staged_sft_serial_completion.json`.
- Next priority: preserve and reuse these 187 episodes and evaluate only the
  missing 13. The planned SFT cards ablation is secondary to completing the
  already-registered dice/router cohort within the remaining GPU allowance.

### Frozen SFT missing-only continuation, then cards ablation

- Preserve all 187 saved dice episode bytes as compressed, hash-verified kernel
  inputs. No extra dataset or rejected cancelled/error-kernel input is needed.
  Each T4 loads the identical frozen SFT adapter once and first evaluates its
  missing dice seeds only (eight on one GPU, five on the other).
- Only after dice completes, compare serial/batch4 quotes on eight fixed cards
  training states, seeds 20000–20007 and rounds 0,1,3,5 repeated. Cards test
  starts only if that expanded gate passes. Two disjoint 100-seed cards shards
  complete the registered SFT-only ablation; no policy/model/decoder changes.
- Command: `.venv/bin/python runs/submit_sft_missing_then_cards.py`.
  Accepted private version 1, T4 x2 requested, source compiles before submission.
  The first local compile caught a placeholder collision with an embedded
  source identifier; corrected before allocation. No GPU time spent on that
  preflight failure.
- Fresh quota has 6578.239610000004 seconds left. The 2150-second session bound,
  remaining original batch allowance 2042.9443798065186 seconds and a further
  2150-second allowance for unknown continued interactive allocation total
  6342.944379806519, which fits. Script cap is 2100 seconds, below the platform
  maximum and before the reset. Do not submit more unless another fresh check
  admits useful work after these outputs are retained.
- Full source, scheduling/parity patches, embedded-input hashes, producer
  wrapper commit, inherited core provenance note, private remote metadata and
  quota arithmetic: `results/training/kaggle_staged_sft_complete.json`.
  Monitor sleeps 610 seconds between status checks and retrieves terminal
  outputs into `runs/kaggle_staged_sft_complete_output`.

### T17 full-run progress: full validation before iteration 400

- The next complete 480-example validation pass improves loss from
  0.4035864770412445 to 0.32049041986465454, taking 1253.209341665999 seconds.
  The upstream callback is 399 consumed microbatches and 199 optimizer updates;
  the trainer then reports microbatch 400. Keep those distinct counts.
- Preserve the unchanged 800-microbatch recipe's training/guard snapshots and
  best-adapter hash in `results/training/qwen35_uniform_validation399.json`.
  Command: `.venv/bin/python runs/retain_mlx_step400.py`; full source is embedded.
- This is teacher-loss progress under the same shared lock and memory guard,
  not evidence of market-making PnL. No held-out aggregate is opened.

### Complete local and hosted verification

- Command: `.venv/bin/pytest -q` reports **111 passed in 6.24 seconds**, including
  MLX loss/gradient and bounded-shape checks. These small tests do not load
  heavy model weights or bypass the training lock.
- Hosted run 40977684968 is green at `7af1272487d3edaf3e9125d12c7c573014890f14`:
  each Python interpreter has 98 passed and 13 skipped, with native C++ and
  ASan/UBSan also passing. Distinguish hosted skips from local execution.
- Exact local output, source hashes, command and receipt-generation source:
  `results/verification_full_111.json`.
