---
title: "MakeMarketBench: synthetic market making under informed flow"
author: Nicolas Coquelet
date: 1 November 2026
---


## Research note

Qwen3.8-27B SFT reaches **90.4 [85.2, 94.8]%** of the Bayes dealer expected PnL on dice and **incomplete (192/200 episodes), unscored** on cards. Complete test games use 200 common episodes; intervals are marginal 95% paired bootstrap intervals. This is a synthetic benchmark, not evidence of profitability on an exchange. All four complete NIM API baselines have negative mean expected PnL in both games.

## Results

TEST uses seeds 200-399. VAL uses 80000-80099 and selects the frozen primary 27B adapter. The table reports percentage of Bayes expected PnL [95% CI].

| Dealer | Split, n/game | Dice | Cards |
|:---|:---|---:|---:|
| Bayes reference | TEST, 200 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Qwen3-1.7B SFT800 | TEST, 200 | 55.1 [36.1, 71.5] | 16.0 [-7.9, 36.2] |
| Qwen3.5-4B local RLOO192 | TEST, 200 | 48.4 [30.9, 63.6] | 38.5 [17.8, 56.4] |
| Qwen3.5-4B original SFT | TEST, 200 | 80.2 [72.3, 87.3] | 32.6 [14.7, 48.5] |
| Qwen3.5-4B original GRPO | TEST, 200 | 64.5 [50.1, 76.6] | 48.4 [32.0, 62.8] |
| Qwen3.5-4B staged GRPO | TEST, 200 | 72.0 [59.8, 82.6] | 51.6 [35.3, 65.5] |
| Qwen3.5-4B router (two adapters) | TEST, 200 | 80.2 [72.5, 87.2] | 51.6 [35.3, 65.5] |
| Qwen3.8-27B SFT, batch four | TEST, 200 dice only | 90.4 [85.2, 94.8] | Incomplete; unscored |
| Nemotron-3-Ultra, thinking off | TEST, 200 | -127.4 [-167.1, -95.3] | -197.4 [-246.4, -154.7] |
| Nemotron-3.5-Bolt | TEST, 200 | -430.6 [-485.7, -386.9] | -717.0 [-800.6, -649.8] |
| Nemotron-3.5-Lightning, thinking off | TEST, 200 | -3108.5 [-3439.6, -2843.2] | -3047.8 [-3320.4, -2824.5] |
| GPT-OSS-20B, low reasoning | TEST, 200 | -183.7 [-222.1, -152.4] | -128.5 [-160.6, -101.9] |
| Qwen3.8-27B SFT, batch four | VAL, 100 | 94.5 [87.3, 101.0] | 77.5 [61.0, 91.1] |

The staged 4B SFT candidate additionally has 200 dice episodes (identical to the router dice row), but only 100 cards episodes, unscored. The router uses SFT on dice and GRPO on cards, frozen using validation. The raw logs retain every failed or incomplete run. Full tables, sources and limitations: [repository README](https://github.com/ncoprod/makemarketbench).

\newpage

## Protocol and limitations

A dealer posts prices and sizes for a hidden terminal value. Public reveals and its own fills update its information. An informed trader knows the value; noise flow supplies spread revenue. The Bayes reference updates its exact posterior and optimizes one-round expected PnL. It is myopic, not a proven upper bound on the sequential game.

Training, validation and test seeds are disjoint. Both local and newest-family choices freeze before test aggregates are opened. The conservative final family counts eleven candidates, including unavailable trials. Registered claims require corrected paired API contrasts and positive cards lower bounds under both episode and whole-inference-group bootstraps. A uniform tenfold bootstrap draw audit retains the primary verdicts; it cannot upgrade failures.

The 27B completion gate has an explicit disclosed deviation: complete SFT validation and the GRPO dice score mathematically fix the unchanged max-min winner, whatever missing GRPO cards would score. This certifies the selection outcome, not statistical superiority. Partial games are never extrapolated.

Intervals condition on frozen weights and recorded responses; training-seed variation and new API samples need replications. Quantizers, adapters and training recipes differ across backends. GRPO/RLOO rewards use teacher-visited states; learned-policy fills can induce a different state distribution. The earlier exploratory 50-episode results influenced development and are retained separately rather than treated as a controlled before/after experiment.

## Compute and engine engineering

A private saved batch completes 800 SFT and 96 GRPO updates of Apache-2.0 Qwen3.8-27B on two T4 GPUs in 17,901.346 seconds. Peak allocations are 12.238 / 12.261 GB. Periodic checkpoints retain optimizer and RNG state. A second training seed completes but remains validation-only. The three deadline-stopped evaluation versions preserve downloaded raw episodes and logs. The single bounded continuation fails before inference because Kaggle omits the ERROR source kernel from the saved input mounts. No new episodes are acquired; cards remains at 192/200, unscored. No second rerun is submitted.

The C++20 book uses flat price levels, pooled orders, intrusive FIFO links and branch-light inlined matching. In 200 alternating pairs, marginal median throughput improves from 20.14M to 20.81M events/s. Median paired speedup is 1.0511 [1.0318, 1.0694] after trial correction. Checksums match (6638620577960879557). The marginal-median ratio and median paired ratio are different estimands. Stateful Python/C++ parity, Release and ASan/UBSan checks cover the engine; the stream measures in-memory throughput, not network latency.

## Reproduction and evidence

The README reproduction commands rebuild the scores, bootstrap audit and figure from committed raw episodes without model weights or new API calls. Run `bash scripts/ci_local.sh` for the local fallback and `python scripts/build_paper.py` to rebuild this note.

The [final score manifest](https://github.com/ncoprod/makemarketbench/blob/main/results/final/summary.json) lists every score path and raw hash. The [selection receipt](https://github.com/ncoprod/makemarketbench/blob/main/results/training/qwen38_selection.json), [error receipts](https://github.com/ncoprod/makemarketbench/blob/main/results/training/qwen38_terminal_errors.json), [continuation receipt](https://github.com/ncoprod/makemarketbench/blob/main/results/training/qwen38_test_finish.json), [training receipt](https://github.com/ncoprod/makemarketbench/blob/main/results/training/qwen38_seed0_completion.json) and [engine samples](https://github.com/ncoprod/makemarketbench/blob/main/results/engine_trials/inline_matching_confirmation.json) retain commands, configs, revisions, source commits and runtimes. The [paper receipt](https://github.com/ncoprod/makemarketbench/blob/main/results/final/paper.json) records the exact build and input hashes.
