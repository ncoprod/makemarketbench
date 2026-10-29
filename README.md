# MakeMarketBench

**Can LLMs make markets under adverse selection?** MakeMarketBench turns the
classic trading-interview game "make me a market" into a scored benchmark. A
model quotes a bid and an ask (price and size) on a hidden value while
information is revealed round by round. An informed counterparty picks off
mispriced quotes and noise traders pay the spread. Every decision is compared
with an exact Bayesian reference dealer.

The repository contains:

- a **C++20 limit order book** (price-time priority) exposed to Python with
  pybind11, plus a pure-Python reference implementation. Parity is tested on
  randomised event streams, and both are benchmarked;
- **two games** with exact posteriors: a dice sum, and a card sum with hints;
- a **Bayes reference quoter** and naive baselines, with **scoring** that
  separates the edge earned from noise traders from the cost of adverse
  selection;
- an **LLM harness** for any OpenAI-compatible endpoint, with strict JSON
  quotes, one repair retry, backoff and on-disk caching;
- **training code**: local MLX LoRA SFT and a DAgger round on Apple silicon
  (run, with real before/after numbers), plus an Unsloth + TRL GRPO Kaggle
  kernel whose reward is the edge relative to the Bayes player (GPU run
  pending).

Repository name note: `quotebench` is already used by unrelated LLM benchmarks
(e.g. LeonardNJU/quoteBench, about shell quoting), so this project is called
MakeMarketBench.

## Why market making is a good probe

Quoting a two-sided market is a compact test of reasoning under uncertainty,
and a single number scores it: expected profit. A good dealer has to do
four things:

1. **Inference.** Compute a posterior for V from the public information
   (convolutions of dice, sampling without replacement, conditioning on
   hints).
2. **Adverse selection.** A fill is information. If you get lifted, V is
   more likely above your ask. The spread must cover the expected loss to
   informed traders (Glosten and Milgrom, 1985).
3. **Price/size trade-off.** Wider quotes lose noise flow, and larger sizes
   lose more to the informed trader. The optimum is interior and depends on
   the posterior's shape, not only its mean.
4. **Learning from your own fills.** The reference updates its posterior
   from the fills it observes. A dealer that ignores this leaves a measurable
   amount of edge on the table.

LLMs often answer "what is E[V]?" correctly but have no instinct for points
2 to 4. The benchmark measures this gap directly.

## Games

All randomness (V, reveals, counterparties) is a deterministic function of the
episode seed. Every dealer therefore faces the same counterparties (common
random numbers), and comparisons are paired by seed.

**Market mechanics (both games).** There are 6 rounds. In each round the
dealer posts `{"bid", "bid_size", "ask", "ask_size"}`: prices on a 0.1 tick,
sizes 0-10, and the bid strictly below the ask. The quote rests in the order
book for that round only. Exactly one counterparty then arrives and sends an
IOC order:

- with probability 30% an **informed trader** who knows V: they buy the whole
  ask size if V > ask, or sell the whole bid size if V < bid;
- otherwise a **noise trader** with valuation u = M + e, where M is
  E[V | public info] and e ~ N(0, s^2). If u > M they buy D ~ U{1..5} units
  at any price up to u; otherwise they sell D units at any price down to u.

Inventory carries over and is settled at V. The model is told all of these
rules, including the probabilities.

**Game 1: dice.** V is the sum of 8 hidden d6. One die is revealed before
each of rounds 2-6, and 3 dice stay hidden. Noise s = 4.

**Game 2: cards.** V is the sum of 5 cards drawn without replacement from a
52-card deck (A=1 ... K=13). Before rounds 2-6 the public sees, in order: a
card, a hint, a card, a hint, a card. Hints are true statements about the
whole hand, drawn from these templates:

- exactly k face cards;
- the highest card;
- the lowest card;
- the parity of V;
- the number of cards of 5 or lower;
- whether there is a pair.

The exact posterior comes from enumerating the rank multisets of the unseen
cards, and is checked against brute force in the tests. Noise s = 6.

**Game 3: Figgie-lite (multi-agent).** This is a simplified Figgie with four
players and four suit order books in the shared engine.

- The deck has 40 cards: one suit has 12, two have 10 and one has 8. The goal
  suit is the same colour as the 12-card suit.
- Everyone antes 50 and is dealt 10 cards.
- There are 4 rounds of one-card bids and asks. Quotes rest for the round and
  owners cancel their own orders at the end of it.
- At settlement, goal-suit cards pay 10 each, and the rest of the 200-chip pot
  goes to whoever holds the most goal cards.

The evaluated player sits in seat 0 against three *hand-Bayes* bots. Each bot
computes the exact layout posterior from its hand and values cards by Monte
Carlo. The metric is the **trading gain**: final score minus the score with no
trades, which removes most of the luck of the deal. This game is
game-theoretic, so the bots are a strong heuristic rather than an optimum, and
the outcome is very noisy: a single card can swing a 100-chip bonus.

## Reference dealer and baselines

- **bayes**: the exact posterior over V from the public information *and* from
  the dealer's own fills. The fill likelihood mixes the informed and noise
  components in closed form. The dealer then quotes the risk-neutral
  one-round optimum, found by grid search over every price tick and size for
  each side. Because exactly one counterparty arrives per round, the two sides
  decouple. This is *myopically* optimal: it ignores the information value of
  a quote, so it is a strong reference, not a proven upper bound.
- **bayes-public**: the same, but it ignores its fills (an ablation).
- **mean-sd**: quotes the exact public posterior mean ± 1 sd, with size 3.
- **fixed-prior**: quotes the prior mean ± 1 prior sd, with size 3, and never
  updates.
- **random**: uniform prices within 2 prior sd and uniform sizes.

## Scoring

The headline number is **% of Bayes edge**: 100 × mean E[PnL] of the
dealer / mean E[PnL] of the reference, over the same seeds. The 95% CI comes
from a paired bootstrap over episodes (10,000 resamples).

E[PnL] is the per-round expected profit given the true V and the posted
quote, with the counterparty draw integrated out in closed form, summed over
rounds. This is a Rao-Blackwellised estimator of realised PnL: it has the same
expectation and much lower variance. The test suite checks the closed form
against Monte Carlo runs through the matching engine. Realised PnL from the
engine is reported as well.

| Metric | Meaning |
|---|---|
| Noise edge / Adverse cost | Expected PnL earned from noise traders / lost to informed traders, per episode |
| Regret / round | Expected PnL of the reference quote minus the dealer's quote, under the exact posterior at the dealer's own state. CI is clustered by episode. |
| Spread / optimal | Median ratio of the quoted spread to the reference spread at the same state |
| Mid error | Mean absolute distance between the quote midpoint and the posterior mean |
| V in quote, size, abs(final position), realised PnL sd | Calibration and inventory-risk descriptors (in `results/leaderboard_*.md`) |
| Invalid answers | Share of rounds whose first answer failed parsing or validation. After one failed repair retry the round is played with no quote. |

## Leaderboard

Seeds 0-49 (50 episodes × 6 rounds per dealer and game). All models were
called on the free NVIDIA NIM endpoint at temperature 0.3, with at most 4096
output tokens and at most 2 requests in flight. `gpt-oss-20b` used
`reasoning_effort=low`: at the default effort it often spent all 4096 tokens
reasoning without answering. Raw episodes, including every model answer, are
in `results/raw/<game>/*.jsonl`.

### Game 1: dice

<!-- TABLE:dice -->
| Dealer | N | % of Bayes edge [95% CI] | E[PnL] / episode [95% CI] | Regret / round | Spread / optimal | Mid error | Adverse cost | Invalid answers |
|---|---|---|---|---|---|---|---|---|
| bayes | 50 | 100 [100, 100] | 14.0 [11.4, 16.2] | 0.00 | 1.00 | 0.16 | 4.2 | 0% |
| qwen3-1.7b-sft | 50 | 68 [38, 89] | 9.5 [4.8, 13.3] | 0.53 | 1.00 | 0.87 | 6.1 | 0% |
| bayes-public | 50 | 64 [44, 83] | 8.9 [6.1, 11.3] | 0.48 | 1.03 | 0.91 | 6.0 | 0% |
| mean-sd | 50 | 57 [42, 72] | 8.0 [5.9, 9.7] | 0.67 | 0.98 | 0.92 | 4.7 | 0% |
| fixed-prior | 50 | 27 [-9, 57] | 3.8 [-1.2, 8.1] | 1.37 | 1.19 | 1.88 | 4.7 | 0% |
| nvidia/nemotron-3-ultra-550b-a55b (no-think) | 50 | -41 [-107, 9] | -5.7 [-13.6, 1.4] | 2.88 | 0.67 | 1.90 | 15.7 | 0% |
| random | 50 | -41 [-97, -1] | -5.7 [-12.5, -0.1] | 2.76 | 1.22 | 2.59 | 11.2 | 0% |
| openai/gpt-oss-20b +hint | 50 | -109 [-169, -66] | -15.2 [-21.5, -9.9] | 4.40 | 0.22 | 2.81 | 22.0 | 2% |
| openai/gpt-oss-20b | 50 | -146 [-221, -91] | -20.4 [-27.5, -13.9] | 5.28 | 0.22 | 3.26 | 23.4 | 8% |
| nvidia/nemotron-3.5-bolt-e4b | 50 | -459 [-612, -368] | -64.1 [-72.9, -56.2] | 11.93 | 0.12 | 3.96 | 50.1 | 0% |
| qwen3-1.7b-base | 50 | -1728 [-2151, -1455] | -241.3 [-255.7, -226.1] | 41.27 | 0.12 | 24.00 | 124.3 | 0% |
<!-- /TABLE:dice -->

![dice leaderboard](docs/figures/leaderboard_dice.png)

### Game 2: cards

<!-- TABLE:cards -->
| Dealer | N | % of Bayes edge [95% CI] | E[PnL] / episode [95% CI] | Regret / round | Spread / optimal | Mid error | Adverse cost | Invalid answers |
|---|---|---|---|---|---|---|---|---|
| bayes | 50 | 100 [100, 100] | 20.0 [17.2, 22.5] | 0.00 | 1.00 | 0.28 | 4.8 | 0% |
| bayes-public | 50 | 80 [62, 96] | 16.1 [12.1, 19.3] | 0.52 | 1.02 | 1.25 | 6.0 | 0% |
| mean-sd | 50 | 63 [49, 75] | 12.6 [9.5, 15.1] | 1.08 | 0.90 | 1.17 | 5.2 | 0% |
| fixed-prior | 50 | 10 [-26, 38] | 2.0 [-4.7, 8.1] | 2.67 | 1.32 | 3.86 | 7.0 | 0% |
| random | 50 | -45 [-93, -11] | -9.1 [-16.6, -2.3] | 5.18 | 1.37 | 4.88 | 14.9 | 0% |
| qwen3-1.7b-sft | 50 | -80 [-161, -15] | -16.1 [-30.0, -3.2] | 5.01 | 1.00 | 4.09 | 21.6 | 0% |
| qwen3-1.7b-base | 50 | -648 [-781, -558] | -129.8 [-139.0, -121.0] | 24.43 | 0.08 | 33.19 | 60.2 | 22% |
<!-- /TABLE:cards -->

![cards leaderboard](docs/figures/leaderboard_cards.png)

### Game 3: Figgie-lite

Seat 0 plays against three hand-Bayes bots. The bot baselines were run on
seeds 0-199. The LLM seat, `gpt-oss-20b` with `reasoning_effort=low`, was
stopped after seeds 0-29 to save API budget for game 2, so the 30-seed rows
are the paired comparison. Gains are in chips per table, out of a 200-chip
pot.

<!-- TABLE:figgie -->
| Seat 0 | Seeds | N | Trading gain / table [95% CI] | Seat-0 trades / table | Invalid answers |
|---|---|---|---|---|---|
| hand-bayes | 0-199 | 200 | 4.97 [-2.34, 12.13] | 7.0 | 0% |
| naive | 0-199 | 200 | 0.82 [-4.99, 6.33] | 5.7 | 0% |
| noise | 0-199 | 200 | -6.48 [-12.14, -0.75] | 4.2 | 0% |
| naive | 0-29 | 30 | 10.52 [-1.80, 20.67] | 5.3 | 0% |
| hand-bayes | 0-29 | 30 | -5.08 [-21.73, 12.43] | 6.8 | 0% |
| openai/gpt-oss-20b | 0-29 | 30 | -6.04 [-20.49, 8.97] | 4.8 | 0% |
| noise | 0-29 | 30 | -8.18 [-14.82, -1.76] | 3.9 | 0% |
<!-- /TABLE:figgie -->

Only the noise bot's interval excludes zero (it loses chips). The ranking
of the two bot baselines even flips between the 30- and 200-seed samples. At this sample size the table cannot separate the LLM from the
simple bots. It is reported as a working multi-agent harness, not as a
result.

## Findings

These are from seeds 0-49. Every number comes from `results/leaderboard_*.json`.

1. **None of the untuned LLMs earned positive expected PnL on the dice
   game.** The Bayes dealer and the simple `mean-sd` heuristic were both
   profitable. In dice, every API model lost money in expectation, and
   `gpt-oss-20b` and `nemotron-3.5-bolt` did worse than the random quoter.
2. **The failure is adverse selection, not arithmetic.** The models quote
   far too tight: their median spread is 12-67% of the optimal spread at the
   same state (`gpt-oss-20b`: about 1.7 points against 8.2 for the
   reference). The informed trader then picks them off. Their expected loss
   to informed flow is 3.7-12x the reference's (15.7-50.1 against 4.2 per
   episode), while they earn less than the reference from noise traders.
   This is the textbook Glosten-Milgrom mistake.
3. **Handing the model the exact posterior does not fix it.** With the
   public posterior mean and sd written into the prompt (`+hint`),
   `gpt-oss-20b` still quotes a ~1.7-point spread. Regret per round improves
   only from 5.28 to 4.40, and the dealer stays deep in negative territory
   (-109% of the reference). The model also ignores what its own fills
   reveal: in a typical episode it keeps buying at the public mean after
   repeatedly being hit by the informed seller.
4. **Thinking budgets matter in practice.** With thinking enabled,
   `nemotron-3-ultra-550b` started deriving the optimal quote analytically,
   with normal tail integrals and size trade-offs. On its probe and on 9 of
   its first 10 benchmark requests, it hit the 4096-token cap before
   answering, after 60-180 s each
   (`results/aborted/nemotron-3-ultra-thinking.json`). `minimax-m3.1-flash` hit the cap in both
   modes. Under a fixed token budget, reasoning that never ends in a quote is
   worth zero.
5. **A 1.7B model distilled from the Bayes dealer beats every untuned LLM.**
   See the next section. It reaches the reference's spread exactly
   (spread/optimal = 1.00) in both games. On dice it captures 68% of the
   reference's edge. On cards it still loses money, because its posterior
   mean drifts after hints: its regret per round rises from 0.0 in round 1
   to 10.1 in round 6. Conditioning on hints and on cards drawn without
   replacement is the hard part for a small model.

## Training a small model

**Target.** Qwen3-1.7B, 4-bit (`mlx-community/Qwen3-1.7B-4bit`), LoRA, in
non-thinking mode, with the same prompts and parser as the API harness.
Training seeds are 10,000 and up (DAgger: 30,000 and up). Evaluation uses the
held-out leaderboard seeds 0-49.

**Local run (MLX, M1 Pro 16 GB).** `train/mlx_pipeline.sh` runs the stages
in order:

1. **Base:** evaluate the untuned model.
2. **SFT:** behaviour cloning of the Bayes reference. Each target is a short
   rationale ("Public E[V] = ..., after my fills E[V] = ... (sd ...)")
   followed by the reference quote as JSON. The data has 9,600 decisions from
   800 episodes per game (`train/make_sft_data.py`), and the loss is on the
   answer tokens only.
3. **DAgger:** the SFT model plays 50 new training episodes per game. The
   reference labels every state the model actually visits, including the
   model's own fills. Training resumes on these on-policy labels mixed with
   4,000 original examples.

**GPU path (Kaggle, GPU run pending).** `train/grpo_unsloth_kaggle.py`
(Unsloth + TRL `GRPOTrainer`, Qwen3-1.7B 4-bit QLoRA) does a short SFT warm
start and then GRPO. The reward is

    format reward + clip((EV(sampled quote) - EV(Bayes quote)) / max(EV(Bayes quote), 1), -2, 0.5)

where EV is the expected one-round PnL under the exact posterior at that
state. The dealer is risk-neutral and the episode score is a sum over rounds,
so this per-decision reward decomposes the episode score relative to the
Bayes player, with no Monte Carlo noise.

`python train/build_kaggle_kernel.py` writes a self-contained kernel in
`train/kaggle/`: the pure-Python engine is embedded, so no compiler is needed,
and `kernel-metadata.json` sets GPU and internet. Push it with
`kaggle kernels push -p train/kaggle`. It saves the LoRA adapter, evaluates
seeds 0-49 with the same prompts and parser, and writes raw episodes that
`mmb score` reads. The kernel is dry-run tested on CPU only. **No GPU numbers
are reported here.** The planning estimate in the script's docstring is not
a measurement: about 4-5 h on a T4 for 150 SFT steps, 300 GRPO steps with 8
generations and the evaluation, at roughly 9-11 GB peak VRAM. `train/serve_vllm.sh` serves the resulting adapter with
vLLM for evaluation through the standard harness.

<!-- TABLE:training -->
| Model | Game | N | % of Bayes edge [95% CI] | E[PnL] / episode [95% CI] | Regret / round | Invalid answers |
|---|---|---|---|---|---|---|
| qwen3-1.7b-base | dice | 50 | -1728 [-2151, -1455] | -241.3 [-255.7, -226.1] | 41.27 | 0% |
| qwen3-1.7b-sft | dice | 50 | 68 [38, 89] | 9.5 [4.8, 13.3] | 0.53 | 0% |
| qwen3-1.7b-base | cards | 50 | -648 [-781, -558] | -129.8 [-139.0, -121.0] | 24.43 | 22% |
| qwen3-1.7b-sft | cards | 50 | -80 [-161, -15] | -16.1 [-30.0, -3.2] | 5.01 | 0% |
<!-- /TABLE:training -->

**What the SFT run actually was.** It used LoRA rank 8 on 16 layers, batch 2,
learning rate 1e-4 and max sequence 1024, with gradient checkpointing. The
config is in `results/training/sft_adapter_config.json`. The run was planned
for 800 iterations but was stopped at **300 iterations** to fit the time
budget, so it saw 600 of the 9,600 examples, about 0.06 epochs. Validation
loss on the 480 held-out examples fell from 4.258 before training to 0.285
at iteration 100, 0.233 at 200 and 0.242 at 300
(`results/training/sft_qwen3-1.7b.log`). MLX reported a median of 14 loss tokens/s, which
counts answer tokens only because the prompt is masked. That came to 43k
answer tokens over the run, on the M1 Pro while other jobs shared the
machine.

## Engine

<!-- TABLE:engine -->
| Implementation | Events / s | Time for 1M events |
|---|---|---|
| C++ engine, native loop (flat pool-based book) | 20.21 M | 49 ms |
| C++ v1 layout, native loop (std::map + std::list) | 8.79 M | 114 ms |
| C++ via pybind11, one call per stream (`replay`) | 7.75 M | 129 ms |
| C++ via pybind11, one Python call per event | 2.88 M | 347 ms |
| Pure-Python reference, one call per event | 1.54 M | 650 ms |
| Pure-Python reference, `replay` loop with checksum | 1.21 M | 824 ms |
<!-- /TABLE:engine -->

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm,plot]"          # builds the C++ extension (CMake >= 3.20, C++20 compiler)
pytest -q                                  # Python tests, incl. C++/Python parity
cmake -S . -B build/native -DMMB_BUILD_NATIVE=ON && cmake --build build/native && ./build/native/test_book

mmb eval --game dice --player bayes bayes-public mean-sd fixed-prior random
export NVIDIA_API_KEY=...                  # free key from build.nvidia.com
mmb eval --game dice --llm openai/gpt-oss-20b --extra-body '{"reasoning_effort": "low"}'
mmb score --game dice                      # writes results/leaderboard_dice.{json,md}
python scripts/make_figures.py && python scripts/update_readme_tables.py
python scripts/bench_engine.py             # engine throughput -> results/engine_bench.json
```

Any OpenAI-compatible server works, for example Ollama
(`--base-url http://localhost:11434/v1 --api-key-env NONE`), `mlx_lm.server` or vLLM.

## Limitations

- **Small samples and wide intervals.** 50 episodes per dealer and game is
  what the shared free API budget allowed. The percentage CIs are wide
  because episode PnL depends on V. Regret per round is a lower-variance
  comparison of the same decisions.
- **Myopic reference.** The Bayes dealer is optimal one round at a time. It
  is not a full POMDP solution and ignores the information value of a quote.
  A dealer can, in principle, beat it slightly.
- **Stylised counterparties.** There is one arrival per round, the informed
  trader knows V exactly, and noise traders are Gaussian around the public
  mean. The rules are given to the model in full, so the benchmark tests
  reasoning within a known model, not model discovery.
- **Risk neutrality.** The score is expected PnL. Inventory is reported but
  not penalised.
- **Endpoint drift.** NIM model versions, quantisation and rate limits can
  change. Answers are cached by request hash, and raw transcripts are
  committed so the numbers can be audited.
- **Game 3 is a sandbox.** Figgie-lite has no exact reference: the bots
  ignore the information in other players' trades. The trading-gain metric is
  very noisy, and only one LLM was run on it.

## Related work

- Glosten and Milgrom (1985), *Bid, ask and transaction prices in a
  specialist market with heterogeneously informed traders*, and Kyle (1985),
  *Continuous auctions and insider trading*: the adverse-selection model
  behind the games.
- Avellaneda and Stoikov (2008), *High-frequency trading in a limit order
  book*: inventory-aware market making, which is out of scope here.
- MARKET-BENCH ([arXiv:2512.12264](https://arxiv.org/abs/2512.12264)):
  LLMs writing backtesters from strategy descriptions. It scores code and
  PnL, not quoting.
- Cattle Trade ([arXiv:2605.14537](https://arxiv.org/abs/2605.14537)): an
  LLM bluffing and trading card game.
- LiveTradeBench ([arXiv:2511.03628](https://arxiv.org/abs/2511.03628)) and
  *LLM Trading: Analysis of LLM Agent Behavior in Experimental Asset Markets*
  ([arXiv:2502.15800](https://arxiv.org/abs/2502.15800)): LLM agents in live
  and experimental markets.
- *When AI Trading Agents Compete: Adverse Selection of Meta-Orders by
  RL-Based Market Making* ([arXiv:2510.27334](https://arxiv.org/abs/2510.27334)):
  an RL market maker under adverse selection, without LLMs.
- Figgie sandboxes for bots:
  [0xDub/figgie-auto](https://github.com/0xDub/figgie-auto) and
  [CornellDataScience/FiggieBot](https://github.com/CornellDataScience/FiggieBot).
  OMakeMeAMarket is a market-making practice game for humans.
- DeepSeekMath / GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))
  and DAgger (Ross et al., 2011) for the training recipes.

When this was written (October 2026) we found no benchmark that asks LLMs to
quote two-sided markets against informed flow and scores them against a
Bayes-optimal dealer.

## Citation

```bibtex
@software{coquelet2026makemarketbench,
  author = {Coquelet, Nico},
  title  = {MakeMarketBench: Can LLMs Make Markets under Adverse Selection?},
  year   = {2026},
  url    = {https://github.com/ncoprod/makemarketbench}
}
```

## License

MIT
