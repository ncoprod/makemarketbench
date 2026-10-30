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

### Correct terminal-status matching before collecting live outputs

- The new kernel slug contains “complete”. The old monitor searched the entire
  CLI output for that substring and treated its first RUNNING poll as terminal.
  The premature output pull returned no files; it did not stop the GPU job.
- Match only the `KernelWorkerStatus` value. Centralize the actual active
  monitors in `scripts/watch_kaggle.py`; preserve the premature pull and exact
  correction in `results/training/kaggle_staged_sft_complete.json`.
- Resume monitoring from the existing recorded poll plus 610 seconds; do not
  send an early status request. The other batch uses the same corrected script.
- Regression checks reject RUNNING/QUEUED even when the slug contains complete
  and accept actual terminal statuses. Ruff passes and the full local suite
  reports **112 passed in 6.31 seconds**; exact output and hashes are in
  `results/verification_full_112.json`.

### Resume the incomplete GPT-OSS cards cohort

- The completed 200-episode dice cohort is unchanged. Cards saves three episodes,
  then stops after three HTTP 504 retries in 1239.676654100418 seconds. Preserve
  the exact error and attempt duration in `results/expansion/protocol.json`.
- Queue `runs/resume_gpt_cards.py` after both existing protocol writers finish,
  then acquire a global NIM slot before `scripts/expand_nim.py --models
  openai/gpt-oss-20b`. The endpoint, decoding and game settings are unchanged;
  cache replay and existing raw rows avoid repeating completed episodes.
- Exact launcher and queue dependencies: `results/expansion/gpt_resume_plan.json`.
  Availability failures do not count as substantive policy nonimprovements.
- Computer-use inventory still returns the native-pipe startup and browser-host
  errors. Preserve this check in `results/training/kaggle_ui_outage.json`; the
  newest-family draft's later training phase and outputs remain unknown.

### Official Qwen catalogue recheck after the UTC date change

- Command: `.venv/bin/python runs/recheck_qwen_catalogue.py`. Query the Qwen
  organisation's latest-created models and all Qwen3.8 model IDs directly from
  the Hugging Face API. The general-purpose family still lists 27B, Flash-Next
  and 2.4T-A95B with their FP8 variants; no smaller Qwen3.8 is listed.
- Retain the exact query URLs, IDs, revisions, tags and source in
  `results/training/qwen_latest_recheck_20261030.json`. Image, speech and driving
  specialists are distinct releases; they do not replace the general-purpose
  quote-policy family selection. No training recipe changes or extra trial.

### Frozen CUDA continuations retained; registered router complete

- Commands: `.venv/bin/python runs/retain_original_final_resume.py`;
  `.venv/bin/python runs/retain_sft_complete.py`;
  `.venv/bin/python scripts/stitch_routed_policy.py --input
  results/kaggle/staged-sft-complete-v1 results/kaggle/staged-final-v1`.
- Original GRPO: 200 dice and 200 cards, pair runtime 6754.009554178 seconds,
  both workers completed. Original 120 dice bytes remain identical. Legacy rows
  lack an adapter hash; preserve that limitation rather than retrofit metadata.
- Staged SFT: both dice shards complete, 200 episodes total, prior 187 bytes
  preserved. Cards has 100 episodes on seeds 300–399. Worker 200 fails its
  eight-development-state serial/batch4 gate; worker 300 passes the same states.
  For development seed 20005, round 1, its failed batched quote is 31/45 versus
  serial 33/47 (sizes 4); the other seven quotes agree. Record both outcomes.
- Staged pair runtime 1753.499030631 seconds; exit codes [1,0]. This is a
  partial secondary ablation, not a complete SFT cards result. Do not score it
  or bypass the failed gate to consume quota. The registered SFT-dice/GRPO-cards
  router now has two complete 200-episode cohorts without a policy change.
- Retained sources, hashes, runtimes and raw episodes:
  `results/training/kaggle_original_final_resume_completion.json`,
  `results/training/kaggle_staged_sft_complete_completion.json`,
  `results/final/staged-routed/cohorts.json`. Canonical seeded-game checks:
  `results/final/resumed_game_comparability.json`. No new held-out aggregates opened.

### Development inference profiles in two fresh allocations

- Hypothesis: the observed one-quote cards discrepancy depends on batching and
  execution history rather than a changed adapter. Measure this directly before
  claiming serial/batch numerical equivalence. The partial SFT cards cohort
  remains incomplete; do not bypass its failed development-state gate.
- Commands: `.venv/bin/python runs/submit_inference_profile.py` and the same
  command with `--replicate`. Both accepted private version 1, requesting T4 x2,
  with identical fixed sources and frozen SFT/GRPO adapter hashes. One stage per
  GPU; two fresh allocations test cold-session reproducibility.
- Sixteen fixed development states cover both games. Record an initial serial
  baseline, then randomized batch sizes 1,2,4,8,16 twice each, exact quotes and
  attempts, latency, cache counts and peak allocated/reserved VRAM. These are
  descriptive systems measurements, not new training seeds or policy scores.
- Each script is capped at 1050 seconds and each session at 1100. At the second
  admission, 3507.709486000007 seconds remain; the first profile's remaining
  bound is 1088.8721990585327 seconds. Both new/existing batch bounds plus 1100
  seconds for unknown continued interactive allocation fit below that quota.
- Full sources, private metadata, admission arithmetic, commands and later polls:
  `results/training/kaggle_inference_profile.json` and
  `results/training/kaggle_inference_profile_repeat.json`. Monitors sleep 610
  seconds between status checks. Preserve outputs before any further admission.

### Remove a receipt-writer dependency from the GPT cards retry

- Diagnosis: the initial retry launcher waited for both the default and every
  supplementary API batch because explicit-model batches shared a receipt file.
  That could delay finishing an original API cohort behind optional coverage.
- Add the single `--protocol-file` option to `scripts/expand_nim.py`. The queued
  GPT retry now writes `protocol-gpt-resume.json`, allowing it to acquire a
  global slot independently while retaining the same cache/raw paths and model
  settings. No active request or shared slot was interrupted.
- Stop only the old waiting retry launcher; retain its original plan as
  `results/expansion/gpt_resume_plan_initial.json`. The new launcher/source is
  in `results/expansion/gpt_resume_plan.json`. Other queued batches continue.
- Ruff and CLI help pass; `.venv/bin/pytest -q` reports 112 passed in 6.05 seconds.
  Verification receipt: `results/verification_nim_resume.json`. No new policy or
  test score inspected.

### Development profiles complete: speed and numerical agreement differ

- Command: `.venv/bin/python runs/retain_inference_profiles.py`. Both private
  kernels complete; pair runtimes are 879.2537151480001 and 974.645657264 seconds.
  Full downloaded logs, per-case timings/VRAM/quotes and hashes are retained in
  `results/training/kaggle_inference_profile_completion.json`.
- Sixteen observations yield fourteen distinct prompts. Across two cold
  allocations, serial baselines agree; four randomized measurements per batch
  size show consistent within-stage quote changes, not evidence of training-seed
  uncertainty. SFT batch4 changes two quotes each time; GRPO batch4 changes none.
  SFT batch2 matches these serial quotes; GRPO batch2 changes one quote each time.
- Median serial → batch4 times: SFT 112.09542951949999 → 46.16061960600001
  seconds; GRPO 114.54942646699993 → 48.046202304500014 seconds. Treat these as
  descriptive development timings with cache/compiler/allocator history, no
  significance or general-policy equivalence claim.
- Official quota after retrieval: 106349.971345 / 108000 GPU seconds used,
  leaving 1650.028655 seconds. A subsequent settled quota read with both batch
  kernels complete is retained before considering one bounded compiler ablation.
  The newest-family live draft remains inaccessible; do not infer its end time
  or final training state from this account-wide billing observation.

### Make the official quota reset timezone explicit

- The SDK returns a naive reset timestamp. The owner specifies UTC. Attach UTC
  when absent before serializing `refresh_at`; avoid local-time interpretation
  in future admission arithmetic. Existing numeric quota measurements do not
  change. Command: the isolated Kaggle Python running `scripts/kaggle_quota.py`.
- Ruff passes; the real API snapshot now explicitly reads
  `2026-10-31T00:00:00+00:00`. This is a timestamp-provenance correction.

### T17 full-run progress: full validation before iteration 600

- The next complete 480-example validation pass improves loss to
  0.28865939378738403 from 0.32049041986465454, in 1253.5278436669978 seconds.
  The callback is at 599 consumed microbatches and 299 optimizer updates; the
  trainer subsequently reports microbatch 600. Sampled physical peak remains
  6712983552 bytes, below the unchanged seven-GiB guard.
- Command: `.venv/bin/python runs/retain_mlx_step600.py`. Snapshot, best-adapter
  hash, full source and counts: `results/training/qwen35_uniform_validation599.json`.
  Same registered recipe; no policy-quality result inferred from teacher loss.

### T18 — focused compiler ablation on development states

- Hypothesis: Unsloth auto compilation contributes to the repeatable batch-shape
  quote differences. Set only `UNSLOTH_COMPILE_DISABLE=1` before imports, retaining
  pinned bases/adapters, decoder, parser and sixteen development observations.
  Repeat batch sizes 1 and 4 twice. No held-out scores or new policy candidate.
- Commands: `.venv/bin/python runs/prepare_compile_ablation.py`;
  `.venv/bin/python runs/submit_compile_ablation.py`. Accepted private version 1.
  Script cap 1450 seconds, session cap 1500, below 1650.028655 seconds remaining.
- Both prior batch sessions are complete; official account debit is unchanged
  over the receipt's settled interval. Explicitly label the inferred no-current-
  overlap assumption. The inaccessible draft's exact end remains unknown.
- The new title resolved to `makemarketbench-compiler-ablation`, differing from
  the requested slug. The initial metadata pull therefore failed locally with
  FileNotFoundError; do not invent its unrecorded HTTP status. Pull the success
  URL's actual slug, verify private metadata, inputs and the identical source
  hash, correct receipt/ledger/monitor, and let the same useful job continue.
- Full sources, admission, exact local error, verified remote metadata and
  documented upstream flag: `results/training/kaggle_compile_ablation.json`.
  Monitor sleeps 610 seconds; no resubmission or allocation restart.

### T18 outcome — compilation explains only part of the discrepancy

- Command: `.venv/bin/python runs/retain_compile_ablation.py`. Both workers
  complete; downloaded pair runtime is 692.7815337 seconds. Serial baselines
  match the compiled profiles. SFT batch4 changes state index 2 in both repeats,
  compared with two changed states under compilation. GRPO batch4 still matches
  its serial baseline. Full quotes, attempts, timings, memory and hashes are in
  `results/training/kaggle_compile_ablation_completion.json`.
- The flag removes one discrepancy, but compilation is not the sole cause.
  Numerical batch equivalence remains unestablished. No frozen policy, held-out
  cohort or test score changes in response to this development-only ablation.
- Official quota after collection is 107048.112345 / 108000 seconds, leaving
  951.887655 seconds. Record the exact account debit separately from the
  measured script wall time in `results/training/kaggle_budget.json`.
- Refresh the existing OAuth access token with unchanged scopes before expiry;
  retain only expiration, source and permissions metadata, never credentials:
  `results/training/kaggle_oauth_refresh.json`.

### T19 — batch-composition diagnostic within the final quota minutes

- Hypothesis: companion sequences and padding lengths contribute to the
  remaining SFT quote discrepancy. Keep compilation disabled, frozen adapters,
  decoder and sixteen development observations. Compare serial with natural,
  reversed, increasing/decreasing prompt-length orders, plus true fourfold
  identical-prompt batches for the two previously discrepant states.
- Direct first-generation calls bypass cache deduplication and repair, making
  the identical-prompt case a real four-sequence batch. Record that configuration
  difference explicitly; no held-out episodes, rewards or policy selection.
- Commands: `.venv/bin/python runs/prepare_composition_ablation.py`;
  `.venv/bin/python runs/submit_composition_ablation.py`. Initial local admission
  correctly refused adjacent quota reads less than sixty seconds apart. Use an
  earlier settled read after the prior kernel completed; the account debit is
  unchanged. Remaining 951.887655 seconds covers the 850-second script and
  900-second session bounds. Private version 1 is accepted.
- The post-push source hash check assumes the upload filename and fails locally;
  the remote metadata instead names the pulled source after its slug. Verify
  that actual code_file hash, private flag and inputs without resubmitting or
  restarting the useful job. Recovery source, full notebook and admission:
  `results/training/kaggle_composition_ablation.json`. Monitor sleeps 610 seconds.

### Local DAgger preflight and newest-family output recovery recheck

- Command: `.venv/bin/python runs/audit_dagger_prefix.py`. The original 9,600
  demonstrations contain 4,800 decisions per game. The registered 4,000-row
  replay prefix contains 2,033 dice and 1,967 cards decisions, retaining both
  games before adding 1,800 on-policy states. Exact source, hashes and source-
  defined disjoint seed ranges: `results/training/dagger_prefix_audit.json`.
  Demonstration rows have no per-row seed metadata; do not claim a per-row
  reconstruction from this audit. No recipe or policy changes.
- Recheck computer-use inventory between 01:27:40 and 01:28:00 UTC on October
  30. It still returns empty apps/browsers and the same native-pipe/browser-host
  failures. No newest-family final weights retrieved or allocation restarted.
  Keep the exact observation in `results/training/kaggle_ui_outage.json`.

### T19 outcome — mixed companions matter; small parity gates are insufficient

- Command: `.venv/bin/python runs/retain_composition_ablation.py`. Both workers
  complete in 541.0333677 seconds. Serial quotes match the earlier profiles.
  SFT natural/reverse batches change development states 2 and 13; length-sorted
  batches change state 2. Four identical copies of each target match serial.
  GRPO natural/reverse changes state 7; sorted orders and identical SFT-target
  copies match serial. All cases parse. Exact quotes, timing, VRAM and hashes:
  `results/training/kaggle_composition_ablation_completion.json`.
- Cache/deduplication is explicitly bypassed in this experiment; do not compare
  its grouping to cached profiles as though their execution were identical.
  Mixed-companion sensitivity remains real in this configuration, but the probe
  does not distinguish masked padding from floating-point kernel effects. No
  frozen test policy is changed, and no held-out score is inspected.
- Official account debit is 107594.130345 / 108000 seconds, leaving 405.869655
  seconds. A final focused padding-invariance check can fit a 350-second script
  and 380-second session bound. Remaining time cannot support another full
  policy cohort, training seed or newest-family pipeline.

### T20 — final bounded padding check on discrepant development prompts

- Hypothesis: left padding alone can explain the mixed-batch changes. For SFT
  states 2 and 13 and GRPO state 7, compare native single generation, explicitly
  left-padded single generation to the original mixed-batch maximum length, and
  four identical padded copies. Keep the frozen weights, greedy decoder and
  disabled compilation. Retain input shapes, mask counts, texts, parsed quotes,
  latency and peak VRAM. No policy/test changes or held-out aggregation.
- Commands: `.venv/bin/python runs/prepare_padding_ablation.py`;
  `.venv/bin/python runs/submit_padding_ablation.py`. Private version 1 accepted,
  remote source hash verified using its actual metadata code_file. The settled
  405.86965500000224-second quota covers a 350-second script / 380-second
  session bound. Monitor sleeps 610 seconds, including when a short job could
  finish sooner. Full sources/admission: `results/training/kaggle_padding_ablation.json`.

### T20 version 1 failed — builder corrupted the guard's expected hash

- The global numeric replacement from 850 to 350 also changed a substring in
  the GRPO adapter digest. The checksum guard raises AssertionError before
  comparison. The adapter itself is unchanged. SFT child started, but the main
  loop failed before its wait/finally block, so pair runtime is missing.
- Command: `.venv/bin/python runs/retain_padding_failure.py`. Retain all outputs
  and exact error in `results/training/kaggle_padding_ablation_failure.json`.
  Official account debit is 35.80000000000291 seconds during this isolated
  failed allocation. This is billing, not an invented complete script runtime.
- Restore the pinned hash and use exact deadline/budget replacements. Check all
  frozen hash constants before submitting a shorter 320-second script /
  350-second session retry within the remaining 370.069655 seconds. No quality
  result or stopping-rule nonimprovement is inferred from this failed probe.

### T17 completes — full local SFT under the unchanged memory guard

- Command: `.venv/bin/python runs/retain_mlx_sft_completion.py`. The registered
  run completes 800 consumed microbatches / 400 optimizer updates in
  14021.710419917003 seconds. Full 480-example validation loss improves from
  2.1129567623138428 initially to 0.2795637547969818 at callback 799, whose
  complete pass takes 1254.1336007499995 seconds. The best adapter is copied
  for frozen policy validation, with its hash checked in the receipt.
- Sampled physical footprint remains 6712983552 bytes, below seven GiB; the
  guard completes successfully. This is an approximate process-footprint peak,
  distinct from MLX allocator measurements. Warm-start/restarted-optimizer
  provenance is retained, not represented as a fresh-base seed replicate.
- Source, exact command/config/model revision and snapshots:
  `results/training/qwen35_sft_completion.json`. The queued launcher proceeds
  to 100 validation episodes per game; DAgger/RLOO recipes remain unchanged.
  No held-out aggregate or policy-quality conclusion is drawn from teacher loss.

### Fresh release and API discovery checks

- Command: `.venv/bin/python runs/check_qwen_task_catalogues.py`. Task-filtered
  official Qwen catalogues confirm the newest general-purpose family remains
  Qwen3.8; inspect both text and multimodal tags rather than only the newest
  unfiltered uploads. Full responses/source:
  `results/training/qwen_task_catalogues_20261030.json`. Existing pinned weights,
  permissive licence and measured two-T4 fit evidence remain authoritative.
- Queue a dated GET /v1/models refresh under the global NIM mutex, with three
  sequential attempts and transient-error backoff. Plan/source:
  `results/expansion/catalogue_recheck_plan.json`. It does not replace the
  original batch's catalogue. Evaluate a newly listed Qwen3.8 only if discovered.

### T20 version 2 — corrected guard, same padding diagnostic

- Correct the expected frozen hash using an exact-string replacement and verify
  both adapter constants before building. Replace only the literal deadline and
  recorded budget, then submit private version 2 of the same kernel. Commands:
  `.venv/bin/python runs/prepare_padding_ablation.py`;
  `.venv/bin/python runs/submit_padding_retry.py`. Remote private metadata, input
  source and source hash verified. Session 350 seconds / script 320 seconds fit
  the observed 370.06965499999933 seconds left.
- Keep the failed version-one receipt unchanged. Retry source, admission and
  polls: `results/training/kaggle_padding_retry.json`. This remains a systems
  diagnostic, not a policy trial or a change to held-out evaluation.

### T20 version 2 completes; weekly GPU submission loop closes

- Command: `.venv/bin/python runs/retain_padding_retry.py`. Both workers
  complete; downloaded pair wall time 283.59922675099995 seconds. SFT state 2
  has 86 left-pad tokens: single and four identical padded copies match native.
  SFT state 13 has 195 left-pad tokens: padded single matches native, while all
  four identical padded copies change their quotes. GRPO state 7 is a zero-pad
  control and all cases match native. Every answer parses. Retain all cases,
  shapes, masks, raw text, VRAM and timings in
  `results/training/kaggle_padding_retry_completion.json`.
- This exposes inference-configuration sensitivity; do not claim that the
  previous eight-state gate proves equivalence or that this identifies a
  unique kernel defect. Frozen policy configurations remain unchanged.
- Command: `.venv/bin/python runs/close_kaggle_budget.py`. Official weekly
  account debit is 107918.349345 / 108000 seconds (29.9773192625 hours), with
  81.65065500000492 seconds remaining and no reservation. Even the first
  observed native generation including warmup takes at least 97.324324784
  seconds, before counting environment/model loading. Another useful cold
  allocation cannot fit. No filler job or automatic next-week allocation.
- Close the submission loop, preserving the incomplete newest-family run and
  unknown exact allocation boundaries honestly. Source/evidence:
  `results/training/kaggle_budget_closure.json`; per-kernel ledger remains
  `results/training/kaggle_budget.json`. Continue the registered normal local
  improvement loop, API coverage, final frozen scoring, figures and CI.

### T17 policy validation: complete dice cohort is weak despite lower loss

- The frozen local SFT adapter completes all 100 dice validation episodes,
  seeds 80000–80099, with no format/failure cases. It reaches
  23.134629375436102% of Bayes [−4.187352895527057, 48.005063619931335],
  expected PnL 3.3782453474795444 [−0.6220414708649687, 6.953221967179836].
  These are descriptive 95% paired-bootstrap validation intervals.
- Midpoint-error mean is 1.890888401085412; state-level regret increases from
  zero in the first round to 3.921408153369202 in the last. Exact evaluation
  command/config/revision and raw observations are retained in
  `results/validation/qwen35-sft/leaderboard_dice.json` and its raw directory.
- Command: `.venv/bin/python runs/diagnose_mlx_sft_dice.py`. All 600 rationale
  statements parse; public-value mean absolute error is 1.55, with only 179
  within 0.01. Posterior-value mean absolute error is 1.8276492617554572, with
  113 within 0.01. These round counts are dependent within 100 episodes, not
  600 independent trials; rationale numbers need not faithfully explain the
  model. Full source and per-state checks:
  `results/validation/qwen35-sft/dice_rationale_diagnosis.json`.
- Diagnosis: learned format/low teacher loss is insufficient for public-value
  arithmetic and sequential posterior conditioning. The registered next change
  remains DAgger on more actor-reached states; do not tune on new held-out
  scores or claim that backend/recipe differences isolate one causal factor.
  Cards validation and the queued DAgger/RLOO rounds continue.

### Bolt endpoint fails mid-cohort with HTTP 404

- Previously responsive Bolt stops after 95 complete dice episodes, seeds
  200–294, with `RuntimeError: NotFoundError(404): Error code: 404`; cards
  immediately returns the same error. Dice attempt wall time is
  8304.971775054932 seconds; cards 0.09904932975769043 seconds. Preserve the
  final protocol attempt, cached responses and every completed episode byte.
- Command: `.venv/bin/python runs/retain_bolt_failure.py`. Exact structured
  failure, provenance and raw hash: `results/expansion/bolt_partial_failure.json`.
  No partial-cohort score or permanent-retirement claim.
- Queue one missing-only retry conditional on successful fresh catalogue
  discovery still listing the model. Same endpoint/decoder/cache settings,
  independent protocol-bolt-resume receipt and global NIM mutex. Source/plan:
  `results/expansion/bolt_resume_plan.json`. If absent, record that limitation.
  Both shared NIM slots are now held by live trial-ledger wrappers; leave them
  untouched and continue local work while our API jobs queue.

### T17 policy validation completes; larger DAgger collection starts

- Command: `.venv/bin/python scripts/retain_local_validation.py sft`. Verify
  both 100-episode cohorts, all six rounds, seeds 80000–80099, one frozen
  adapter/base identity, reference cohorts and leaderboard hashes. Source and
  result: `scripts/retain_local_validation.py`;
  `results/validation/qwen35-sft/cohorts.json`.
- Cards reaches 8.87169554476217% of Bayes [−35.13311349022747,
  42.71311186729823], expected PnL 1.555911360939736
  [−5.485642180841259, 8.046854681414676]. Realised PnL is 9.12
  [1.5397500000000022, 16.52]; retain it, but do not substitute that more
  favourable statistic for the registered expected-PnL target. No format
  errors/failures. These validation intervals do not establish positive edge.
- Registered max-min selection score is −56.865370624563894. No new policy
  test is opened. The existing follow-up launcher acquires the heavy-ML mutex
  and starts collecting 150 actor episodes per game, targeting 1,800 new
  labels, before the unchanged DAgger training/validation and RLOO trial.
  The verifier CLI adds no new selection rule or experimental configuration.

### Register paired local validation contrasts before follow-up outcomes

- Add `scripts/compare_local_validation.py` for the two already-registered
  DAgger/RLOO trials against SFT on identical validation seeds. Retain expected-
  PnL mean differences, marginal 95% intervals and 98.75% Bonferroni intervals
  over four method/game contrasts (16,000 resamples, seed zero). The max-min
  selection score and nine-candidate held-out correction are unchanged.
- CLI help and Ruff pass. Actual contrasts wait for complete 100-episode
  follow-up cohorts and their verified manifests; no placeholder numbers.
  These are validation diagnostics, not a claim of independent test inference.
- The 02:51:36–02:54:53 UTC computer-use recheck still returns empty surfaces
  and identical transport failures. Exact observation is appended to
  `results/training/kaggle_ui_outage.json`; GPU submissions remain closed.

### Larger DAgger collection completes; registered training begins

- Hypothesis and change remain the registered follow-up: exact labels on more
  actor-reached states may address the local SFT policy's sequential errors.
  Collect 150 episodes per game on seeds 30000–30149: 1,800 new states, mixed
  with the first 4,000 original teacher examples. Collection takes
  5164.821201791994 seconds; peak MLX allocated memory is 4065234612 bytes.
- Commands: the exact collection/audit/training commands are retained in
  `results/training/qwen35_followup_plan.json`;
  `.venv/bin/python runs/retain_dagger_collection.py` verifies every episode,
  seed, label count, actor identity and dataset hash. Result and source:
  `results/training/qwen35_dagger_collection.json`.
- The full token audit passes: 5,800 training and 480 validation examples,
  maximum lengths 976 and 966, no inference-prefix mismatches. The existing
  fixed padding of 1,025 tokens needs no change. Start the registered
  800-microbatch continuation from SFT under the shared heavy-ML lock and
  physical-memory guard. Policy validation remains pending; no improvement
  is inferred from successful collection or teacher loss.
- A 04:08 UTC computer-use check still returns empty inventories and the same
  native-pipe/browser-host errors. No new observation of the Qwen3.8 draft.

### T21 — validate the fixed last DAgger checkpoint as well as the teacher-selected one

- Diagnosis: DAgger teacher loss is 0.4232971966266632 at callback 199 and
  0.32471534609794617 at 399, versus 0.27956369519233704 initially. The current
  best checkpoint is therefore unchanged SFT. This does not test whether the
  trained DAgger weights improve policy PnL on actor-reached states.
- Focused change: register the last numbered saved DAgger checkpoint before
  any follow-up policy-validation cohort or new held-out aggregate is inspected.
  Its identity is fixed by training order, never by policy scores. Preserve the
  teacher-selected checkpoint and its planned validation as a separate trial.
- Command: the existing `train/final_candidate.py --model ... --revision ...`
  first prepares and evaluates that fixed checkpoint on 100 validation episodes
  per game, seeds 80000–80099, then selects among all four local validation
  trials using the unchanged max-min score. It refuses to freeze when any
  trial's validation is missing. No extra held-out candidate: one local winner
  is evaluated, so the nine-candidate correction remains unchanged.
- Widen the validation contrast family from four to six before any follow-up
  contrast is observed: 1−0.05/6 intervals, 24,000 resamples, seed zero. Both
  previous and amended registrations remain in this log. Plan, training snapshot
  and protocol amendment: `results/training/qwen35_dagger_last_plan.json`,
  `results/protocol.json`. Actual policy result is pending.
- Add regression checks for refusing incomplete validation and selecting the
  balanced two-game winner rather than the highest dice score. Targeted tests
  and Ruff pass; full CI follows the milestone commit.

### Hosted CI blocked by account billing before any job steps

- CI run 41017970170 for commit `1c8fecc` fails before assigning a runner or
  executing any steps. The exact annotation says: "The job was not started
  because recent account payments have failed or your spending limit needs to
  be increased. Please check the 'Billing & plans' section in your settings".
- Commands and complete primary API receipts:
  `.venv/bin/python runs/retain_ci_billing_block.py`;
  `results/ci_billing_block_20261030.json`. There is no existing self-hosted
  runner for this repository. The previous commit's hosted CI is green;
  the amended code passes 114 local tests in 5.47 seconds and Ruff
  (`results/verification_validation_amendment.json`). These are distinct facts.
- Notify the owner asynchronously; do not change payments or account spending
  limits. Continue experiments and rerun hosted CI after the restriction is
  resolved. An unstarted workflow is neither a code regression nor a passing CI.

### DAgger continuation stops on the registered validation plateau

- Run completes in 10750.590444499998 seconds. The validation callbacks are
  0, 199, 399 and 599, with losses 0.27956369519233704,
  0.4232971966266632, 0.32471534609794617 and 0.3136330544948578.
  Three checks fail to improve on the initial adapter, triggering the existing
  early-stopping rule at callback 599: 599 consumed microbatches and 299
  completed optimizer updates, not a rounded 600/300.
- Best step is zero; selected adapter hash is byte-identical to SFT. This
  teacher-loss-selected policy has no weight improvement. The distinct last
  numbered checkpoint is `0000500_adapters.safetensors`; its policy quality
  remains unknown until the separately registered validation runs.
- Memory guard completes normally: sampled physical footprint peak
  6739197952 bytes, MLX allocated peak 4202603070 bytes. No memory-limit failure.
  Commands, source, all validation points and identities:
  `.venv/bin/python runs/retain_dagger_training.py`;
  `results/training/qwen35_dagger_completion.json` and linked training/guard
  results. The original follow-up launcher proceeds to its registered policy
  validation, then RLOO and the amended final-selection sequence.

### Local native safety verification while hosted runners are unavailable

- Command: `.venv/bin/python runs/verify_ci_local.py`. Repository-wide Ruff,
  native Release rebuild/tests, ASan/UBSan rebuild/tests and C++ extension
  availability all pass. Actual outputs, elapsed times, source hashes and
  environment: `results/verification_ci_local_fallback.json`.
- This is macOS verification with local LeakSanitizer disabled, not a green
  hosted Ubuntu/Python-version matrix. The complete local Python suite remains
  the 114-pass run in `results/verification_validation_amendment.json`.
  Hosted CI still refuses jobs for account billing; the owner request is pending.
- The 07:18 UTC browser inventory check still has the same transport errors;
  retain it in `results/training/kaggle_ui_outage.json`. No new GPU submission.

### Fresh NIM discovery, retained partial failures and additional candidates

- The dated GET /v1/models completes under a global NIM slot: 85 listed models,
  no Qwen. Full response/source/provenance:
  `results/expansion/models_20261030.json`. Catalogue presence does not prove a
  generation endpoint works: both Mistral-7B and Nemo return HTTP 404; Llama
  3.2-11B yields one complete dice episode before HTTP 404 in both games.
  Preserve `results/expansion/protocol-supplement.json` and its partial raw row.
- GPT-OSS cards resumes from three to 125 complete episodes, then fails after
  three HTTP 504 attempts, with batch wall time 13835.842253923416 seconds.
  Preserve all bytes and no partial score:
  `results/expansion/gpt_cards_partial_failure.json`,
  `results/expansion/protocol-gpt-resume.json`. Dice replays its complete cache.
- Verify pinned HF model-card licences before adding Mistral Large 4, Llama
  3.2-90B text-only and Granite-3.0-8B as availability candidates. Source/full
  licence evidence: `results/licenses_nim_followup_20261030.json`;
  `THIRD_PARTY_NOTICES.md`. Mistral Large 4 has the custom Mistral Ecosystem
  Open-Weight License, not Apache or unrestricted MIT.
- The hosted Mistral schema accepts `reasoning_effort=none/high`, unlike its
  vendor low/medium/high examples. Fix `none` before any model evaluation:
  https://docs.api.nvidia.com/nim/reference/mistralai-mistral-large-4-1t-a52b-infer.
  Plan/source/commands: `results/expansion/additional_followup_plan.json`.
  Queue this batch then another identical-config, missing-only GPT-OSS retry,
  each under the global NIM mutex with concurrency one. No successful episode
  or response is discarded, and no held-out score chooses a model or decoder.

### Teacher-selected DAgger policy validation completes

- Commands: `.venv/bin/python scripts/retain_local_validation.py dagger` and
  `.venv/bin/python scripts/compare_local_validation.py dagger`.
- All 100 validation episodes per game, seeds 80000–80099, complete. The
  selected adapter is the unchanged SFT adapter. Dice remains
  23.134629375436102% of Bayes, cards 8.87169554476217%; paired expected-PnL
  changes and both marginal and six-contrast-family intervals are exactly
  zero in each game. This is no policy improvement under teacher-loss selection,
  not an evaluation of the distinct last trained DAgger checkpoint.
- Sources: `results/validation/qwen35-dagger/cohorts.json` and `vs_sft.json`,
  retaining exact commands, revision, input hashes and git commit. RLOO starts
  under the shared heavy-ML lock; last-checkpoint validation remains registered
  before final selection and held-out scoring.
- Browser inventory at 08:15 UTC still reports the native-pipe and browser-host
  errors. Preserve the exact result in `results/training/kaggle_ui_outage.json`;
  no new GPU job or claim about inaccessible Qwen3.8 progress.

### Local RLOO training completes before policy validation

- Registered command is retained verbatim in `results/training/qwen35_rloo.json`:
  `train/mlx_rloo.py` from the SFT adapter, 24 groups of four categorical
  samples, dice-only exact quote-EV/format rewards, learning rate 1e-6, seed 5.
  This is REINFORCE with a leave-one-out baseline, not TRL GRPO.
- All 24 groups complete in 1175.5073630419938 seconds; one group has zero
  advantages. No training-reward trend is claimed because states differ.
  MLX allocated peak is 4392258616 bytes; the physical-footprint guard samples
  5086642176 bytes and exits successfully. The saved adapter differs from SFT.
- Command to retain the completion receipt:
  `.venv/bin/python runs/retain_rloo_training.py`. Source, initial/final adapter
  hashes and linked training/guard files:
  `results/training/qwen35_rloo_completion.json`.
  Policy validation starts under the heavy-ML lock; no validation or held-out
  improvement is inferred from successful optimization steps.

### Intermittent NIM 404s: bounded transport retry, unchanged policies

- Evidence: Bolt first stops after 95 dice episodes with HTTP 404, resumes
  successfully to 138, hits 404 again, then generates valid cards episodes.
  `results/expansion/protocol-bolt-resume.json` records the failed dice attempt
  (4020.912531375885 seconds); its cards batch remains in progress. These
  observations rule out treating every NIM 404 as permanent model retirement.
- Focused change: retry NIM 404s with the existing bounded backoff, just like
  timeouts/429/5xx. Other providers' 404s and permanent 400/401/403/410 errors
  still fail immediately. No prompt, decoder, game seed or successful response
  changes. Existing live clients finish with their original imported code.
- Verification commands: `.venv/bin/ruff check src/makemarket/llm.py
  tests/test_llm_retries.py` and `.venv/bin/pytest -q`: 127 pass in 6.07 seconds.
  Regression tests verify identical retry requests, cached successful replay,
  a hard three-attempt limit and immediate permanent-client-error failure.
  Receipt: `results/verification_nim_404_retry.json`.
- Queue one missing-only Bolt/Llama-3.2-11B follow-up after the current Bolt
  and additional-model/GPT batches, under the same global NIM mutex. Both
  have produced valid episodes before 404s. Exact command/source and prefix
  preservation checks: `results/expansion/intermittent_resume_plan.json`.
  This is infrastructure recovery, not another policy-tuning trial or an
  improvement inferred from a partial test cohort.

### RLOO24 validation improves; continue the local self-improvement loop

- Commands: `scripts/retain_local_validation.py rloo` and
  `scripts/compare_local_validation.py rloo`, each via `.venv/bin/python`.
  All 100 episodes per game complete. Dice rises from
  23.134629375436102% to 48.695043276153044% of Bayes; cards from
  8.87169554476217% to 19.55496643819823%.
- Paired expected-PnL changes: dice +3.7324716959344295, marginal 95% CI
  [1.9463066225400596, 5.594895028624427], originally registered six-contrast
  CI [1.3507798115233602, 6.212015001267778]. Cards +1.8736240971330098,
  marginal [0.39508384179146094, 3.4522276034022834], six-contrast
  [-0.08127892992507649, 4.111049415684826]. The cards change does not survive
  that correction. Sources: `results/validation/qwen35-rloo/cohorts.json` and
  `vs_sft_6_contrasts.json`. Conditional on this one training seed and cache.
- Dice improves while remaining below target, so stopping would violate the
  requested loop. Stop only our old worker process group before local selection
  or test starts; the last-Dagger validation has four saved dice episodes and
  resumes missing-only. Its interrupted supervisor is explicitly superseded,
  not recorded as a model-training failure. No held-out aggregate is opened.

### T22 — extend the same RLOO recipe from 24 to 96 groups (registered)

- Hypothesis: the short RLOO trial is update-budget limited. Change only its
  fixed training budget; continue both adapter and Adam state, keeping seed 5,
  lr 1e-6, four categorical samples, 128-token limit and exact rewards.
- Command/source: `.venv/bin/python -u runs/qwen35_rloo_extensions.py`, retained
  in `results/training/qwen35_rloo_extensions_plan.json`. It first resumes the
  fixed last-Dagger validation, then runs the guarded 96-group continuation
  and 100/game policy validation on seeds 80000–80099. Preserve RLOO24 weights,
  history and cached validation. Never use test seeds for this decision.
- Measure paired increments versus RLOO24 and SFT on both games. A material
  improvement requires a max-min selection-score gain of at least five points
  and a positive corrected lower bound for at least one game's paired gain,
  without a corrected interval wholly below zero in the other game. Retain
  all results even when this criterion fails.

### T23 — extend the same RLOO recipe from 96 to 192 groups (registered)

- Same focused hypothesis and controls as T22; continue the exact preceding
  adapter/optimizer, measure against RLOO96 and SFT. Run both fixed budgets,
  without selecting a favorable intermediate training step from policy scores.
- Before either new outcome, enlarge the validation contrast family to fourteen:
  five follow-ups versus SFT in both games, plus the two consecutive-budget
  comparisons in both games. Bonferroni 1−0.05/14, 56,000 resamples, seed zero.
  Preserve earlier six-contrast files and report the conservative enlarged
  intervals too. Protocol amendment: `results/protocol.json`.
- One final local winner is still selected by the same max-min rule, so the
  final held-out family remains nine, including unavailable candidates. The
  extended supervisor ends at validation, allowing the requested loop to be
  assessed before freezing and testing once. Results are pending.
- Selection regression checks and full suite pass: 129 tests in 6.01 seconds,
  with Ruff clean (`results/verification_rloo_extension_selection.json`).

### Bound continuation checkpoint storage and preserve reboot recovery

- Measured RLOO checkpoint pair is about 93 MiB (31 MiB adapter, 62 MiB Adam
  state). Keeping every additional group in both extensions would use roughly
  15 GiB on a volume with 48 GiB available, shared with the other repositories.
  Keep each fixed budget's frozen adapter and latest resume checkpoint; retain
  every group's complete JSON history. Original RLOO24 artifacts stay intact.
- `train/mlx_rloo.py` atomically replaces its progress JSON before pruning the
  preceding optimizer checkpoint. Preserve loaded history immediately, so an
  exception before a continuation's first successful group cannot erase it.
  Rewards, sampler, gradients, learning rate, seeds and budgets do not change.
- Fault injection makes the JSON replacement fail and verifies both previous
  progress and optimizer checkpoint survive; retry then commits new progress
  before cleanup. Ruff and all 130 tests pass in 6.66 seconds. Exact commands,
  source hashes and checks: `results/verification_rloo_checkpoint_storage.json`.
- The 09:42 browser recheck still fails. Hosted CI run 41035351043 for `ceb5528`
  still has zero job steps and the same account-billing refusal; append primary
  receipts to `results/ci_billing_block_20261030.json`. No owner response yet.

### Bolt partials retained; one Mistral field-omission compatibility control

- The original-import Bolt resume stops at 138 dice and 39 cards episodes.
  Both errors are exactly `RuntimeError: NotFoundError(404): Error code: 404`.
  Dice attempt wall time is 4020.912531375885 seconds, cards
  5068.9419713020325 seconds. Preserve all valid episodes without scores:
  `.venv/bin/python runs/retain_bolt_resume_completion.py`;
  `results/expansion/bolt_resume_partial_completion.json`. The already queued
  missing-only recovery will use the new bounded NIM 404 retries.
- The additional batch begins. Llama-3.2-90B's text-only availability probe
  succeeds; Mistral Large 4 returns HTTP 400, `Invalid request`, with the
  documented `reasoning_effort=none`. Retain both in
  `results/expansion/protocol-additional.json`; no test-based prompt choice.
- Recheck the primary hosted schema at
  https://docs.api.nvidia.com/nim/reference/mistralai-mistral-large-4-1t-a52b-infer.
  It documents none/high and says omission is served as none. Register one
  field-omission compatibility control, preserving the failed explicit request
  and separate labels/caches. Do not assume server modes are identical solely
  from that documentation. Other request settings stay unchanged.
- Source and exact command: `results/expansion/mistral_default_plan.json`,
  `.venv/bin/python -u runs/mistral_default_followup.py`. It waits for the
  additional-model/GPT batch, takes a shared slot, then evaluates a usable
  configuration on the fixed 200/game cohort. Ruff passes for the one-row
  endpoint-settings addition; this is availability recovery, not model tuning.

### T21 outcome — distinct last DAgger weights do not solve the policy gap

- All 100 validation episodes per game complete, including the four dice
  episodes preserved across the intentional supervisor change. Exact commands,
  base/adapter identity and raw-file hashes:
  `results/validation/qwen35-dagger-last/cohorts.json`.
- Dice is 37.07467210510954% [7.38244956873757, 60.641507063954734] of Bayes;
  cards −39.7077063429703% [−92.01579810008809, 1.486122090191159]. No format
  failures in either game. These are numerical quote decisions, not invalid JSON.
- Paired expected-PnL changes versus SFT: dice +2.0356014237767592 with
  14-contrast CI [−3.678260986447036, 7.894486999248122]; cards
  −8.519819342696623 with CI [−18.31621571147791, 1.117073474115093]. Both
  corrected intervals include zero. Preserve marginal CIs too; do not claim
  corrected evidence of either dice improvement or cards degradation.
- Max-min selection score −42.92532789489046 remains worse than RLOO24's
  −31.304956723846956. Result: `results/validation/qwen35-dagger-last/vs_sft.json`.
  The registered RLOO96 continuation starts from the exact RLOO24 adapter and
  Adam-state hashes; no local selection or held-out test has started.
- Granite-3.0-8B's probe fails after three bounded HTTP 404 attempts in
  15.734764099121094 seconds; Llama-3.2-90B continues its complete-cohort attempt.
  Primary receipt: `results/expansion/protocol-additional.json`.

### T22 training completes; fixed 96-group policy validation starts

- Resume adds 72 groups to the exact original 24-group history, taking
  3952.6985441669967 seconds. Cumulative training wall time across the two
  separately initialized processes is 5128.20590720899 seconds, excluding their
  policy-validation time. Do not describe 96 cumulative groups as 96 new ones.
- MLX allocated peak 4403563576 bytes; sampled physical footprint peak
  5042601984 bytes. The guard completes normally. Frozen RLOO24 history is
  unchanged; all 24 earlier history entries match exactly.
- Bounded retention leaves the final adapter and latest adapter/optimizer pair,
  129985219 bytes in total. Exact command/source and input hashes:
  `.venv/bin/python runs/retain_rloo96_training.py`;
  `results/training/qwen35_rloo96_completion.json`. Validation is pending;
  successful training is not evidence that T22 improves the policy.

### Prioritize missing original API baselines, preserve the slow additional cohort

- Llama-90B's validation availability probe generates 304 output tokens and
  stops, taking 126.76465725898743 seconds. It is slow even well below its
  4096-token cap (`results/expansion/protocol-additional.json`), so lowering
  the cap alone is not an evidenced remedy. No held-out policy score is read.
- Register `.venv/bin/python -u runs/nim_baseline_priority.py` to wait for the
  next complete Llama-90B episode, preserve every completed row and cached
  response, then stop only that supervisor's process group. Acquire a shared
  slot for the missing-only GPT-OSS resume before releasing the already queued
  Bolt/Llama-11B and Mistral controls. Resume Llama-90B afterward.
- Exact source/commands, initial prefix identity and observed transition:
  `results/expansion/baseline_priority_plan.json`. All requested cohorts remain
  200/game on seeds 200–399, with unchanged prompts, decoder and timeout.
  The interrupted batch is explicitly marked rescheduled, never complete in
  the sense of a finished cohort. This is scheduling from validation runtime,
  not selection on test outcomes or discarded slow/negative episodes.

### API priority transition verified; output recovery remains unavailable

- Preserve Llama-3.2-90B's seven completed dice episodes and cache before
  rescheduling its original batch. The paused raw-file hash matches the
  registered transition receipt, `results/expansion/baseline_priority_plan.json`.
  Its unfinished 200/game cohort is unscored; GPT-OSS resumes its existing
  125-card prefix under the shared slot, ahead of the slower additional cohort.
- At the next browser inventory check after 11:56 UTC, both native-pipe startup
  and browser-host errors persist. Retain the exact errors and observation
  bounds in `results/training/kaggle_ui_outage.json`. No newest-model output
  recovery, GPU restart or new-quota submission occurred.

### T22 outcome — another real dice gain, cards uncertainty remains

- All 100 validation episodes per game finish. Dice: 66.25064949045698%
  [49.00663830949743, 80.6395849872454] of Bayes; cards:
  16.16959522093798% [−28.850103662534124, 50.03337429112936].
  No format failures. Exact raw/adapter hashes and commands:
  `results/validation/qwen35-rloo96/cohorts.json`.
- Versus RLOO24, paired expected-PnL increments are +2.563565815263349
  with 14-contrast CI [0.4496906368066986, 5.030294604695359] on dice,
  −0.5937238841614036 [−4.1196724311990325, 2.701148282975692] on cards.
  Source: `results/validation/qwen35-rloo96/vs_rloo.json`; versus-SFT
  contrasts remain in `vs_sft.json`.
- Max-min score rises −31.304956723846956 → −13.74935050954302. The
  17.555606214303936-point gain, positive corrected dice increment and
  non-significant cards contrast meet T22's registered material-improvement
  criterion. This is validation conditional on one training seed, not a test
  claim. Targets remain unmet; T23 continues to 192 groups as registered.
- The cards point estimate is positive but its lower bound remains negative.
  Dice-only reward training and remaining numerical/flow-inference errors
  remain plausible bottlenecks; no format/JSON failure explains the gap.

### Make the final scoring and figure reproducible before unsealing

- Promote the prepared orchestration to `scripts/score_frozen_candidates.py`
  and `scripts/plot_final.py`. The scorer requires the frozen local selection
  and verifies its adapter hash before opening any new held-out score.
  Complete candidates retain the nine-candidate correction; partial cards
  remain unscored and unavailable newest-model candidates remain counted.
- The figure uses complete 200-episode cohorts, marginal 95% intervals and
  explicit missing-game markers. Add concise model/decoder labels, including
  every registered local budget and the two-adapter router.
- Verification: `.venv/bin/ruff check scripts/score_frozen_candidates.py
  scripts/plot_final.py --fix` and `.venv/bin/python -m py_compile` on both
  scripts pass. Neither scoring nor plotting has run; all new held-out
  aggregates remain sealed. Visual verification follows actual result creation.

## Round 3 — saved 27B training, complete baselines, local CI, clear reporting

The owner reopens GPU work for the 30-hour allocation resetting at
2026-10-31T00:00Z. The preceding week's closure remains valid. Four priorities:

1. Finish the already registered RLOO96/192 validation and API cohorts. Verify
   complete seeds and preserved prefixes before scoring; keep partials unscored.
2. Rerun the newest-family recipe as a private **saved batch version**. Preserve
   the exact training core, runtime, trainer logs and complete checkpoints under
   `/kaggle/working`. SFT keeps its 800-update ceiling, 50-step validation and
   patience two; GRPO keeps 96 updates, with checkpoints every five. A 37,800-s
   core cap and 39,000-s wrapper cap leave room before the 12-hour platform cap.
   If interrupted, resume adapter, Adam, scheduler and RNG state from the latest
   complete checkpoint, retaining the original 96-step scheduler ceiling.
3. Use the same max-min validation criterion to select the newest-family
   checkpoint before its one 200/game sealed test. Keep all validation trials;
   the final family still conservatively counts nine candidate slots, including
   any untested/unavailable checkpoint. No new held-out aggregate has been read.
4. Add `scripts/ci_local.sh` and install the tracked pre-push hook, then reduce
   README to a TL;DR, current diagram and one main results table. The local
   macOS checks do not replace the unavailable hosted Ubuntu/Python matrix.

### Batch recovery implementation and local CI fallback

- Remove the unused interactive-notebook alternative from the newest training
  builder. The batch writes `train_core.py` and `train.log` into persisted
  working output, alongside checkpoints, configs and atomic runtime receipts.
  Each save records checkpoint file hashes. GRPO save cadence is independent
  of SFT validation cadence. This is persistence/recovery work, not a claimed
  model-quality improvement. GPU submission waits for the explicit reset.
- Simplify continuation: run the exact retained core instead of patching source
  strings, preserve the original 96-update schedule, reject incomplete optimizer
  checkpoints, and recover the final adapter if checkpoint 96 was saved just
  before interruption. Earlier unsaved-draft outputs remain lost; do not claim
  the rerun starts from its step-500 weights.
- Meaningful CPU regression checks cover incomplete/newest checkpoints, exact
  core reuse, original SFT cadence, GRPO ceiling and final-adapter recovery.
  `pytest -q tests/test_training.py`: 35 passed; Ruff passes. Full local CI
  (Ruff, native extension, Python suite, release native, ASan/UBSan, kernel
  reward/export dry run) is required by `.githooks/pre-push`. LeakSanitizer is
  disabled locally on macOS; hosted Linux coverage remains unavailable.

### Preserve early stopping and reject torn checkpoint saves

- The required pre-push fallback passes all 131 Python tests, Ruff, native
  release, ASan/UBSan and kernel dry run on `f374a5f`. Receipt:
  `results/ci_local_round3.json`; hosted Actions minutes remain exhausted.
- Restore the SFT early-stopping callback state on resume; Transformers
  v5.16.1 otherwise defaults to discarding it. Primary source:
  https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/training_args.py.
  Pin the 27B Python/CUDA stack to the versions in the successful downloaded
  fit/step-500 receipts, rather than silently upgrading between batch sessions.
- Require checkpoint file hashes from the post-save manifest. A present but
  truncated optimizer is rejected, and a partial final manifest line is ignored.
  CPU regression tests exercise both cases and final-step adapter recovery.
- Newest-family evaluation uses serial decoding to avoid the measured
  batch-composition sensitivity. The test builder requires an explicit
  validation-selected SFT/GRPO adapter; default evaluation remains validation.
  Sources: `train/build_kaggle_kernel.py`, `train/qwen38_continue.py`.

### T24 — registered saved-batch Qwen3.8-27B rerun (waiting for reset)

- Source-only Kaggle version 4 is saved with `--no-run`; pullback verifies
  exact source hash, private script type, internet and GPU settings. No new
  GPU allocation is requested. Receipt: `qwen38_round3_source_save.json`.
- Exact command, pinned model/revision, seed, 800/96 recipe, checkpoint
  persistence and admission bounds: `qwen38_round3_batch_plan.json`. Start
  after 2026-10-31T00:00Z only once official quota confirms the new allowance.
- Measure both SFT and GRPO on all 100 validation episodes/game, serially.
  Freeze the same max-min validation winner (SFT wins ties), then run its one
  200/game test. Both local and newest-family selection must precede opening
  any new aggregate test score. Nine final candidate slots remain counted.
- The prior week's 29.9773192625-hour account closure remains historical.
  `kaggle_budget.json` separately records the owner's new-window authorization;
  source-save time is not invented GPU runtime. No existing paper/main.tex or
  paper/main.pdf is present, so there is no paper artifact to regenerate.

### Preserve the later official account-debit revision

- A later official quota snapshot reports 110363.015345 seconds used
  (30.65639315138889 hours), total allowance 108000 seconds, zero reserved.
  The earlier closure observed 107918.349345 seconds. Retain both and the
  2444.666-second increase in `results/training/kaggle_quota_revision.json`.
- The account API does not attribute that increase to a kernel or allocation
  interval. Do not invent an explanation or substitute the favorable earlier
  total. The source-only version was explicitly saved with `--no-run`.
- README now distinguishes the two observations. No current-window GPU run
  is submitted; actual refreshed availability must admit the next batch.

### GPT-OSS original baseline acquisition complete

- The final missing-only cards resume finishes all 200 episodes in
  7062.755683898926 seconds, adding 75 to the exact saved 125-card prefix.
  Dice's already complete cohort is reused. Both games match the canonical
  seeds 200–399, six rounds, hidden values and public reveals.
- Verify every row has the same model, endpoint, temperature, token cap,
  timeout and reasoning settings. Prefix byte hash matches exactly. Commands,
  source, raw hashes and complete protocol: `results/expansion/gpt_complete_cohorts.json`
  and `gpt_complete_canonical.json`. Earlier failed attempts remain retained.
- No aggregate test score is opened. The priority supervisor now waits for
  the Bolt/Llama-11B and Mistral follow-ups before resuming the slower Llama-90B.

### T25 — Mistral availability: reasoning budget and repair state

- Hypothesis: the omitted-field control is reachable but its 4,096-token cap
  is exhausted by reasoning (`finish_reason=length`, no final content). This
  differs from the explicit `none` request's HTTP 400. Preserve both failures.
- Change: one 16,384-token probe on the same validation state, temperature,
  timeout and retries; distinct cache/label. Preserve `reasoning_content`
  during Mistral's repair turn, as its API requires. Two regression cases
  verify Mistral repair state and unchanged handling of other providers.
- Command and registered controls: `results/expansion/mistral_budget_plan.json`.
  Wait for the original baseline resume first. No held-out score chooses this
  budget, and successful availability is not evidence of trading quality.
- Existing controls and Ultra thinking's 53 complete dice episodes followed
  by three HTTP 504s: `results/expansion/round3_api_failures.json`. No partial
  cohort score is computed; retain the negative acquisition result.

### Seal the selected local test while the newest-family trial finishes

- Diagnosis: `train/final_candidate.py` freezes the correct validation winner,
  but its evaluator would immediately print test aggregate scores. Round 3
  requires both local and 27B selections to be frozen first.
- Change: selected local test acquisition now uses `--raw-only`, retaining
  episode files, runtime, memory and provenance without calling the scorer.
  Validation still produces its registered selection metrics.
- Verification: `pytest -q tests/test_mlx_evaluate.py` passes. The regression
  makes aggregation raise if called and verifies acquisition metadata exists
  without any leaderboard. No trained policy or selection rule changes.
- The final scoring entry point also requires both frozen selections and
  checks their adapter hashes before reading test cohorts. The 27B selector
  verifies both complete serial validation cohorts against the actual saved
  adapters, uses the registered max-min rule, and refuses to overwrite its
  frozen decision. `pytest -q tests/test_qwen38_selection.py` exercises a
  mismatched adapter, the SFT tie rule, and immutable selection.

### T23 training complete; validation still pending

- The fixed RLOO192 continuation adds 96 groups in 5,736.486491417003 seconds,
  preserving the complete 96-group prefix and its Adam state. Total training
  chain time is 10,864.692398625994 seconds. MLX allocated peak is
  4,415,212,600 bytes; sampled process footprint is 5,075,107,840 bytes.
- Command, revision, seed, history, memory guard and adapter/checkpoint hashes:
  `results/training/qwen35_rloo192_completion.json`. Only the latest resumable
  optimizer plus the final adapter are retained (129,985,219 bytes).
- Both 100-episode validation cohorts are running. Completion and training
  reward do not establish a policy improvement; T23's registered criterion
  will be assessed after both cohorts and corrected contrasts finish.
- The reset-gated saved-batch launcher verifies the exact registered source,
  refreshes existing OAuth scopes, and requires official refreshed quota
  before submission. Its receipt is `qwen38_round3_batch.json`; current
  `waiting_for_reset` status is not an executed GPU run.

### T23 outcome — a smaller, uncertain gain

- Both games complete all 100 validation episodes. Dice is
  70.41372699169959% of Bayes [55.808478838433615, 83.15306147971803]; cards
  is 16.120801203144943% [-27.79451657152036, 48.847228568152445]. There are
  no format failures. Source: `results/validation/qwen35-rloo192/cohorts.json`.
- The max-min selection score gains 4.163077501242611 points over RLOO96,
  below the registered five-point threshold. Both corrected paired intervals
  include zero. This is the first consecutive real nonimprovement after T22,
  not a claim that the policy got worse. Expected cards PnL remains uncertain;
  its positive realized-PnL interval does not satisfy the expected-PnL target.
- Full assessment and exact inputs: `qwen35_rloo192_iteration.json`. Earlier
  fourteen-contrast receipts remain alongside the expanded comparisons.

### T26 — one final RLOO budget doubling, 192 to 384 groups (registered)

- Hypothesis: one more fixed-budget doubling can distinguish insufficient
  updates from diminishing returns on the unchanged dice-only reward.
- Continue the exact RLOO192 adapter, Adam state and 192-group history; add
  192 groups. Keep seed 5, lr 1e-6, four generations and the same training
  prompts. Commands, checkpoint hashes and full controls are registered in
  `results/training/qwen35_rloo384_plan.json` before new validation.
- Validate 100 episodes/game on seeds 80000–80099. Count six follow-ups versus
  SFT and all three successive RLOO budget contrasts on both games: eighteen
  contrasts, Bonferroni 1-0.05/18, 72000 draws, seed 0. Preserve earlier six-
  and fourteen-contrast receipts. T22's positive dice increment also survives
  this wider family; T23 remains uncertain.
- Require a max-min score gain of at least five points over RLOO192, one
  corrected paired lower bound above zero, and no game with a corrected upper
  bound below zero. If this also fails, T23/T26 are two consecutive genuine
  nonimprovements. Diagnose the gap, freeze the max-min winner across all
  seven local trials, and acquire its single raw-only 200/game test. The
  final nine-candidate family still contains only one selected local policy.
- The first pre-push run correctly refused an incomplete seventh-candidate
  test fixture: one failure, 135 passes. Add RLOO384 to that fixture, keeping
  the same immutable-selection assertion. Exact failed check:
  `results/ci_local_round3_rloo384_initial_failure.json`; this is a test-fixture
  correction, not a training or trading outcome.

### Bolt dice acquisition complete; Llama-11B bounded retries fail

- Missing-only Bolt recovery adds 62 dice episodes in 6,038.942450761795
  seconds and completes seeds 200–399. The exact saved 138-episode prefix
  and all critical decoder settings are unchanged. Canonical hidden values,
  reveals and six-round episode structure pass; no aggregate score is opened.
- Command, raw/prefix hashes, canonical checks and exact Llama-11B retry
  failures: `results/expansion/bolt_dice_complete_cohort.json`. Llama-11B
  retries stop after three HTTP 404 attempts in each game; its partial cohort
  remains unscored. Bolt cards recovery is still running.
- The corrected local CI fixture passes all 136 tests, native Release,
  ASan/UBSan and the Kaggle CPU dry-run. Proof:
  `results/ci_local_round3_rloo384.json`; the initial failed run is retained.
