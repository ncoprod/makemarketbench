# Training and evaluation history

Receipt snapshot retained on 2026-10-31 at commit `6c33924`. This page records
training, validation, failures and acquisition status at that point. The
[current README](../README.md) carries the latest consolidated results.

## Experiments

Training uses exact Bayes demonstrations, disjoint teacher/policy validation,
and a fresh 200-episode-per-game test. The registered recipes, seeds, selection
rule and nine-candidate correction are in [protocol.json](../results/protocol.json).
Test scores do not select checkpoints or prompts. The earlier Qwen3-1.7B
baseline now completes 800 total iterations and both fresh test cohorts
([training](../results/training/qwen17_sft800.json),
[cohorts](../results/final/qwen17-sft800/cohorts.json)); its corrected comparison
waits for the remaining candidates and API cohorts.

**Local Qwen3.5-4B:** MLX 4-bit LoRA, followed by a larger DAgger round and a
small RLOO-style trial. The [token audit](../results/training/qwen35_token_audit.json)
checks non-thinking prefixes and sequence limits. Earlier attempts stopped at
reboot or memory guards; all are retained in [ITERATIONS.md](../ITERATIONS.md).
Projecting only answer positions preserves the full causal loss/gradients;
starting from the interrupted adapter, fixed training shapes complete
**800 microbatches / 400 optimizer updates** in
**14,021.710 seconds**. Full validation loss falls **2.113 → 0.280** on all
**480 examples**, with sampled physical footprint **6.713 GB**
([completed run, memory guard and adapter hash](../results/training/qwen35_sft_completion.json)).
The completed 100-episode-per-game validation gives **23.1% [−4.2, 48.0]** of
Bayes on dice and **8.9% [−35.1, 42.7]** on cards
([complete cohorts and scores](../results/validation/qwen35-sft/cohorts.json),
[numerical-rationale diagnosis](../results/validation/qwen35-sft/dice_rationale_diagnosis.json)).
Teacher loss does not establish policy quality. Local heavy work holds the
shared ML lock. The larger DAgger continuation stops after **599 microbatches /
299 updates** without improving validation loss; its selected adapter is
byte-identical to SFT ([complete training outcome](../results/training/qwen35_dagger_completion.json)).
Its completed 100-episode-per-game policy validation reproduces SFT exactly:
paired expected-PnL changes are zero in both games
([expanded contrast-family intervals](../results/validation/qwen35-dagger/vs_sft.json),
[cohorts and identities](../results/validation/qwen35-dagger/cohorts.json)).
The final local selection also checks the distinct last saved DAgger checkpoint
before opening the test ([registered validation amendment](../results/training/qwen35_dagger_last_plan.json)).
That checkpoint reaches **37.1% [7.4, 60.6]** on dice and
**−39.7% [−92.0, 1.5]** on cards, with no format failures
([complete validation](../results/validation/qwen35-dagger-last/cohorts.json),
[paired contrasts](../results/validation/qwen35-dagger-last/vs_sft.json)).
The dice-only RLOO-style trial completes all 24 groups in 1,175.507 seconds
([runtime, memory and adapter identity](../results/training/qwen35_rloo_completion.json)).
Validation rises to **48.7% of Bayes on dice** and **19.6% on cards**
([full cohorts and CIs](../results/validation/qwen35-rloo/cohorts.json)). The paired
dice expected-PnL gain survives the expanded correction; the cards gain does
not ([all contrasts](../results/validation/qwen35-rloo/vs_sft.json)). Fixed
96/192-group continuations are
[registered before selection or test](../results/training/qwen35_rloo_extensions_plan.json).
The 96-group run finishes its 72 new groups in 3,952.699 seconds, with exact
history-prefix preservation and bounded checkpoint storage
([completion receipt](../results/training/qwen35_rloo96_completion.json)). Its
complete validation reaches **66.3% [49.0, 80.6]** on dice and
**16.2% [−28.9, 50.0]** on cards
([cohorts](../results/validation/qwen35-rloo96/cohorts.json)). The paired dice
increment over 24 groups survives the registered correction; the cards change
does not ([both contrasts](../results/validation/qwen35-rloo96/vs_rloo.json)).
At 192 groups, validation reaches **70.4% [55.8, 83.2]** on dice and
**16.1% [−27.8, 48.8]** on cards
([complete cohorts](../results/validation/qwen35-rloo192/cohorts.json)). This
increment misses the registered material-improvement criterion
([assessment](../results/training/qwen35_rloo192_iteration.json)). The final
384-group continuation falls to **53.8% [34.3, 70.8]** on dice and
**19.7% [−20.7, 51.2]** on cards
([complete validation](../results/validation/qwen35-rloo384/cohorts.json)).
Two consecutive iterations miss the registered improvement criterion, so this
local recipe stops ([assessment and diagnosis](../results/training/qwen35_rloo384_iteration.json)).
The max-min rule freezes RLOO192 across all seven trials
([selection](../results/training/selection.json)); all eighteen validation
contrasts count. Its fresh **200-episode-per-game test is complete**, with
canonical states and file identities verified
([acquisition receipt](../results/final/selected/cohorts.json)). Scores remain
sealed while the newest-model selection is pending.

**CUDA Qwen3.5-4B:** Unsloth QLoRA completes **150 SFT and 300 GRPO updates in
17,983.032 seconds**, peaking at **6.327 GB allocated VRAM** on its training T4
([runtime](../results/kaggle/staged-v1/runtime.json),
[configs and checkpoint histories](../results/training/kaggle_staged_completion.json)).
It saves both adapters before evaluation inside Kaggle; no local conversion
is required. Incompatible model-size adapters are rejected. The initial HTTP
401 and subsequent OAuth recovery are retained in the
[submission receipt](../results/training/kaggle_retry.json).

The paired 100-episode validation ablation gives **+4.96 [2.05, 8.04]** expected
PnL on cards and **−1.37 [−2.29, −0.53]** on dice after GRPO. These descriptive
95% CIs are conditional on one training pipeline and the validation used for
selection ([ablation and source](../results/validation/staged_ablation.json)).
A [registered game router](../results/training/staged_routing_plan.json) therefore
uses the SFT adapter on dice and GRPO on cards: one base model, two adapters.
The staged GRPO test completes 200 episodes in each game. Serial SFT dice
also completes all 200, preserving the earlier 187 episode bytes. The registered
router now has both full cohorts; scores remain sealed. The secondary SFT cards
ablation has only 100 episodes: one worker passes the eight-state batching gate,
while the other fails on one quote before testing
([downloaded outcomes](../results/training/kaggle_staged_sft_complete_completion.json),
[router provenance](../results/final/staged-routed/cohorts.json)). A passing small
training-state check is not proof of numerical equivalence on every state.

Batching improves fixed-state latency but can change generated quotes. Across
two fresh development profiles, SFT batch4 changes two of sixteen quotes in
each repeat; GRPO batch4 changes none. Median latency falls **112.10 → 46.16
seconds** for SFT and **114.55 → 48.05 seconds** for GRPO, including prompt
caching ([all timings and quotes](../results/training/kaggle_inference_profile_completion.json)).
Further [compiler](../results/training/kaggle_compile_ablation_completion.json),
[batch-composition](../results/training/kaggle_composition_ablation_completion.json)
and [padding](../results/training/kaggle_padding_retry_completion.json) probes
expose additional discrepancies without identifying a unique cause. Frozen
evaluation settings stay unchanged; final inference retains both episode and
whole-batch bootstrap intervals. These probes measure systems behaviour,
not held-out policy quality.

A separate original CUDA pipeline also completes training and both 200-episode
SFT tests. Its frozen GRPO continuation now completes both 200-episode games,
preserving the original 120 dice episode bytes
([original outputs](../results/training/kaggle_original_completion.json),
[completed continuation](../results/training/kaggle_original_final_resume_completion.json)). Its trainer
configs specify seed 3407 despite outer metadata saying 0; the
[correction](../results/training/original_seed_correction.json) distinguishes
trainer, adapter-initialization and prompt-shuffle seeds. Backend and recipe
differences prevent interpreting these runs as a pure model-size comparison.

**Newest-family Qwen3.8-27B:** the smallest general-purpose Qwen3.8 release is
27B, Apache-2.0 ([pinned model details](../results/training/qwen_latest_details.json),
[official catalogue recheck](../results/training/qwen_latest_recheck_20261030.json),
[text and multimodal catalogues](../results/training/qwen_task_catalogues_20261030.json)).
Its Unsloth NF4 checkpoint occupies **22,344,833,801 bytes (20.81 GiB)** against
approximately **29.12 GiB** across two T4s, before activation, optimizer and
loader overhead ([weight metadata](../results/training/qwen38_quantized.json)).
Two loading probes fail before updates; head-aware multi-GPU placement then
completes both SFT/GRPO fit updates in **840.319 seconds**, with **11.994 GB /
12.018 GB** peak allocations ([runtime](../results/kaggle/qwen38-fit-v3/runtime.json),
[all attempts](../ITERATIONS.md)). The planner places the untied head on GPU 1;
some layers retain bfloat16 weights. This measured fit supports choosing the
newest family rather than assuming a single-T4 memory limit.

The [full registered pipeline](../results/training/qwen38_pipeline_plan.json)
reaches **500 downloaded SFT updates**
([progress](../results/kaggle/qwen38-training-v4/progress/step500-runtime.json)).
Its later phase and trained outputs remain unverified: computer use fails,
and official CLI/SDK routes cannot retrieve the unsaved live draft
([errors and recovery receipts](../results/training/kaggle_ui_outage.json)).
No policy score is inferred from training loss or the fit probe. Account billing,
script runtimes and unknown allocation boundaries remain distinct in the
[GPU budget ledger](../results/training/kaggle_budget.json).
The earlier weekly closure observed **29.9773 / 30 account-quota hours**, with
**81.65 seconds** left ([stopping evidence](../results/training/kaggle_budget_closure.json)).
A later official snapshot reports **30.6564 hours used**, zero reserved, and no
per-kernel attribution for the increase
([both observations](../results/training/kaggle_quota_revision.json)). The refreshed
allocation provides **30 account-quota hours**; private saved batch version 5
is submitted after that official confirmation
([quota and submission](../results/training/qwen38_round3_batch.json)). Actual
runtime and VRAM will come from downloaded outputs, not the session cap.

**NIM coverage:** the [fresh endpoint catalogue](../results/expansion/models_20261030.json)
lists **85 models**, with no Qwen. Sequential calls hold a shared NIM slot and cache responses.
Long-timeout GLM, DeepSeek and Ultra-thinking probes fail after three HTTP 504
responses ([receipts](../results/expansion/protocol.json)); reduced-reasoning retries
also fail ([receipts](../results/expansion/protocol-reasoning.json)). Unavailable
models receive no score. Lightning and Ultra with thinking disabled complete
fresh 200-episode cohorts in both games
([Lightning](../results/expansion/lightning_cohorts.json),
[Ultra](../results/expansion/ultra_cohorts.json)). GPT-OSS now completes
**200 episodes in both games**, preserving the earlier 125-card prefix and the
same request settings ([complete acquisition and canonical checks](../results/expansion/gpt_complete_cohorts.json)).
Its earlier three-504 failures remain in the
[partial-run receipt](../results/expansion/gpt_cards_partial_failure.json). Bolt stops with HTTP
404 after 95 dice episodes and before cards
([partial failure](../results/expansion/bolt_partial_failure.json)); a
[conditional missing-only retry](../results/expansion/bolt_resume_plan.json) ends
with 138 dice and 39 cards episodes after further 404s
([both preserved partials](../results/expansion/bolt_resume_partial_completion.json));
[bounded NIM backoff and missing-only recovery](../results/expansion/intermittent_resume_plan.json)
now completes **200 episodes in both games**, preserving both cached prefixes
and decoder settings ([verified cohorts](../results/expansion/bolt_complete_cohorts.json)).
All eight original API cohorts pass canonical seeded-game checks
([verification](../results/expansion/primary_api_canonical.json)); scores remain sealed.
Mistral-7B/Nemo fail with HTTP 404; Llama-3.2-11B saves
one dice episode before the same error
([all attempts](../results/expansion/protocol-supplement.json)). Mistral Large 4's
documented non-reasoning request returns HTTP 400; Llama-3.2-90B passes its
availability probe ([receipts](../results/expansion/protocol-additional.json)).
Granite-3.0-8B also fails its bounded HTTP 404 probe
([additional receipts](../results/expansion/protocol-additional.json)).
A [Mistral compatibility control](../results/expansion/mistral_default_plan.json)
omits the rejected reasoning field, but exhausts its 4,096-token budget on
reasoning without a final answer. A [registered larger-budget probe](../results/expansion/mistral_budget_plan.json)
uses the same validation state before any test-based choice. Ultra's
medium-effort thinking run saves 53 dice episodes before three HTTP 504s;
the incomplete cohort remains unscored
([failure receipts](../results/expansion/round3_api_failures.json)).
[Baseline-first scheduling](../results/expansion/baseline_priority_plan.json) preserves
the slow Llama-90B cohort, finishes missing GPT/Bolt baseline episodes first,
then resumes it with the same settings and cache.
Partial cohorts remain unscored; the corrected final comparison is pending.


## Round 3 saved 27B replication and inference diagnosis

The primary saved Qwen3.8-27B batch completes 800 SFT and 96 GRPO updates in
17,901.34559750557 seconds. The second seed restores SFT step 150 after stopping
at 250, then completes 96 GRPO updates in 10,116.505320549011 seconds
([primary](../results/training/qwen38_seed0_completion.json),
[replication](../results/training/qwen38_seed1_completion.json)).

Serial SFT dice validation, 100 common episodes, reaches 94.575920754075% of
Bayes [87.69126590056176, 100.88946539023392] for the primary and
70.24443757136461% [53.04110568983144, 83.92028500027654] for the replication.
The registered paired expected-PnL difference is 3.5530164985717927; its
ten-contrast-corrected interval is [1.1251129936209707, 6.329603282166639]
([source and identities](../results/validation/qwen38_sft_seed_dice.json)).
These intervals condition on the two adapters. Two training seeds cannot give a
precise population-level training-seed interval. Both cards cohorts stop at four
episodes and remain unscored. Evaluation config.seed is the inference RNG seed;
actual training seeds are verified separately through adapter hashes and saved
training runtime. The replication cannot replace the primary test candidate.

A generic BF16 warning did not describe the active inference weights. The
[retained dtype probe](../results/training/qwen38_dtype_probe_completion.json)
finds no effective dtype change and a warm ratio of 1.010168640792298.
[Force-eager execution](../results/training/qwen38_compile_probe_completion.json)
also brings no material improvement (ratio 1.0069778650347732). The native CUDA
profile attributes 57.19% of instrumented self device time to bitsandbytes GEMV
and 25.09% to matrix multiplication; those shares include profiling overhead.
The [registered batch-four probe](../results/training/qwen38_batch_probe_plan.json)
tests throughput and valid quotes on training states only. Any promoted batching
is a separate inference policy, even if its small probe agrees with serial.
