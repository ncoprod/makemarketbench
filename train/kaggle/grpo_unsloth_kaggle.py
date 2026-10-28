"""GRPO fine-tuning of Qwen3 (4-bit QLoRA, Unsloth + TRL) on MakeMarketBench.

Designed for a free Kaggle T4 (16 GB) or P100 session. Pipeline:

  1. Unpack the pure-Python ``makemarket`` package (embedded by
     ``train/build_kaggle_kernel.py``, or found on ``sys.path``) - no C++ build.
  2. Build a prompt set of quoting decisions on *training* seeds (>= 10_000)
     visited by the Bayes reference, storing the exact posterior at each state.
  3. Optional SFT warm start on reference demonstrations (format + rough scale).
  4. GRPO. Reward per completion = format reward + edge reward, where the edge
     reward is the expected one-round PnL of the sampled quote under the exact
     posterior, minus that of the Bayes reference quote, divided by
     max(reference EV, 1). Because the dealer is risk-neutral and the score is a
     sum over rounds, this per-decision reward decomposes the episode score
     relative to the Bayes player.
  5. Save LoRA adapters and evaluate on the held-out leaderboard seeds 0-49 with
     the same prompts, parser and scorer as the API harness. Raw episodes are
     written in the same JSONL format as results/raw/<game>/ for ``mmb score``.

Status: written and dry-run tested locally (dataset + rewards, ``--dry-run``);
the GPU run itself is pending. Rough budget on a T4: SFT 150 steps ~25 min,
GRPO 300 steps x 8 generations ~3-4 h, evaluation (2 games x 50 episodes x 6
rounds) ~30-40 min; peak VRAM ~9-11 GB for Qwen3-1.7B, ~14 GB for Qwen3-4B.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PAYLOAD = "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"  # base64 tar.gz of src/makemarket, filled in by build_kaggle_kernel.py

WORK = Path(os.environ.get("MMB_WORK", "/kaggle/working" if Path("/kaggle").exists() else "train/runs/grpo"))


def ensure_package() -> None:
    try:
        import makemarket  # noqa: F401
        return
    except ImportError:
        pass
    if not PAYLOAD:
        raise SystemExit("makemarket not importable and no embedded payload; run build_kaggle_kernel.py")
    dest = WORK / "mmb_src"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tf:
        tf.extractall(dest)
    sys.path.insert(0, str(dest))
    os.environ["MMB_BACKEND"] = "py"


# --------------------------------------------------------------------------- data
def build_prompt_set(n_episodes: int, games: list[str], seed_offset: int = 20_000) -> list[dict]:
    """Quoting states visited by the Bayes reference, with exact posteriors."""
    from makemarket.games import make_game
    from makemarket.market import play_episode
    from makemarket.players import BayesQuoter, best_quote
    from makemarket.prompts import messages_for

    rows = []

    class Recorder(BayesQuoter):
        def quote(self, obs):
            b = self.belief(obs)
            q, info = super().quote(obs)
            ref_q, _ = best_quote(b, obs.m_pub, obs.game.params)
            rows.append({
                "messages": messages_for(obs),
                "game": obs.game.name,
                "values": b.values.tolist(),
                "probs": b.probs.tolist(),
                "m_pub": obs.m_pub,
                "ref_quote": ref_q.to_dict(),
                "ref_mean": b.mean, "ref_sd": b.sd,
            })
            return q, info

    for g in games:
        game = make_game(g)
        for i in range(n_episodes):
            play_episode(game, seed_offset + i, Recorder(), backend="py")
    random.Random(0).shuffle(rows)
    return rows


def sft_target(row: dict) -> str:
    return (f"Public E[V] = {row['m_pub']:.2f}; after my fills E[V] = {row['ref_mean']:.2f} "
            f"(sd {row['ref_sd']:.2f}).\n" + json.dumps(row["ref_quote"]))


# ------------------------------------------------------------------------ rewards
def quote_ev(text: str, row: dict):
    """(expected one-round PnL, belief, params, quote) for the quote in ``text``
    under the row's posterior, or None if the text is not a valid quote."""
    import numpy as np

    from makemarket.games import make_game
    from makemarket.games.base import Belief, round_expectations
    from makemarket.llm import parse_quote

    params = make_game(row["game"]).params
    try:
        q = parse_quote(text, params.max_size)
    except ValueError:
        return None
    b = Belief(np.asarray(row["values"]), np.asarray(row["probs"]))
    return round_expectations(q, b, row["m_pub"], params)["expected_pnl"], b, params, q


def make_reward_fns():
    import numpy as np

    from makemarket.games.base import Quote, round_expectations

    def _row(kwargs, i):
        return {k: kwargs[k][i] for k in ("game", "values", "probs", "m_pub", "ref_quote")}

    def format_reward(prompts, completions, **kwargs):
        out = []
        for i, c in enumerate(completions):
            text = c if isinstance(c, str) else c[0]["content"]
            out.append(0.5 if quote_ev(text, _row(kwargs, i)) is not None else -1.0)
        return out

    def edge_reward(prompts, completions, **kwargs):
        out = []
        for i, c in enumerate(completions):
            text = c if isinstance(c, str) else c[0]["content"]
            row = _row(kwargs, i)
            res = quote_ev(text, row)
            if res is None:
                out.append(-2.0)
                continue
            ev, b, params, _ = res
            ref = Quote(**row["ref_quote"])
            ref_ev = round_expectations(ref, b, row["m_pub"], params)["expected_pnl"]
            out.append(float(np.clip((ev - ref_ev) / max(ref_ev, 1.0), -2.0, 0.5)))
        return out

    return [format_reward, edge_reward]


# ----------------------------------------------------------------------- training
def render(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def train(args) -> None:
    from unsloth import FastLanguageModel  # noqa: I001  (must be imported before trl)
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_len, load_in_4bit=True,
        fast_inference=False)
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0)

    rows = build_prompt_set(args.episodes, args.games)
    print(f"prompt set: {len(rows)} states")

    if args.sft_steps > 0:
        sft_rows = [{"text": render(tokenizer, r["messages"]) + sft_target(r) + tokenizer.eos_token}
                    for r in rows[: args.sft_steps * args.sft_batch]]
        sft = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=Dataset.from_list(sft_rows),
                         args=SFTConfig(output_dir=str(WORK / "sft"), max_steps=args.sft_steps,
                                        per_device_train_batch_size=args.sft_batch, learning_rate=1e-4,
                                        logging_steps=10, save_strategy="no", report_to="none",
                                        dataset_text_field="text", max_seq_length=args.max_seq_len))
        sft.train()

    grpo_rows = [{"prompt": render(tokenizer, r["messages"]), "game": r["game"], "values": r["values"],
                  "probs": r["probs"], "m_pub": r["m_pub"], "ref_quote": r["ref_quote"]}
                 for r in rows[args.sft_steps * args.sft_batch:]]
    cfg = GRPOConfig(
        output_dir=str(WORK / "grpo"), learning_rate=args.lr, max_steps=args.grpo_steps,
        per_device_train_batch_size=args.num_generations, gradient_accumulation_steps=1,
        num_generations=args.num_generations, max_prompt_length=args.max_seq_len - args.max_completion,
        max_completion_length=args.max_completion, temperature=1.0, logging_steps=5,
        save_steps=100, report_to="none", lr_scheduler_type="cosine", warmup_ratio=0.05,
        optim="adamw_8bit")
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=make_reward_fns(),
                          args=cfg, train_dataset=Dataset.from_list(grpo_rows))
    trainer.train()
    model.save_pretrained(str(WORK / "lora_adapter"))
    tokenizer.save_pretrained(str(WORK / "lora_adapter"))
    FastLanguageModel.for_inference(model)
    evaluate(model, tokenizer, args)


# --------------------------------------------------------------------- evaluation
class HFQuoter:
    """Same prompts, parser and one repair retry as makemarket.llm.LLMQuoter."""

    def __init__(self, model, tokenizer, name: str, max_new_tokens: int = 128) -> None:
        self.model, self.tok, self.name, self.max_new = model, tokenizer, name, max_new_tokens

    def reset(self, episode_seed: int) -> None:
        pass

    def _gen(self, messages) -> str:
        import torch

        ids = self.tok(render(self.tok, messages), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def quote(self, obs):
        from makemarket.llm import parse_quote
        from makemarket.prompts import OUTPUT_SPEC, messages_for

        msgs = messages_for(obs)
        attempts = []
        for attempt in range(2):
            text = self._gen(msgs)
            attempts.append({"content": text})
            try:
                return parse_quote(text, obs.game.params.max_size), {
                    "attempts": attempts, "format_error": attempt > 0, "failed": False}
            except ValueError as e:
                attempts[-1]["error"] = str(e)
                msgs = msgs + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"Your answer was invalid: {e}. {OUTPUT_SPEC}"}]
        return None, {"attempts": attempts, "format_error": True, "failed": True}


def evaluate(model, tokenizer, args) -> None:
    from makemarket.evaluate import run_player
    from makemarket.games import make_game

    for g in args.games:
        out = WORK / "results" / "raw" / g / f"{args.label}.jsonl"
        t0 = time.time()
        eps = run_player(make_game(g), lambda: HFQuoter(model, tokenizer, args.label),
                         list(range(args.eval_episodes)), out, workers=1, backend="py")
        tot = sum(sum(r["exp_expected_pnl"] for r in e["rounds"]) for e in eps) / len(eps)
        print(f"[eval] {g}: {len(eps)} episodes, mean E[PnL]={tot:.2f} ({time.time() - t0:.0f}s) -> {out}")


# --------------------------------------------------------------------------- main
def dry_run(args) -> None:
    """CPU check of everything except the model: prompt set and rewards."""
    rows = build_prompt_set(3, args.games)
    fmt, edge = make_reward_fns()
    kw = {k: [r[k] for r in rows[:4]] for k in ("game", "values", "probs", "m_pub", "ref_quote")}
    comps = [sft_target(rows[0]), "garbage", json.dumps({"bid": 1, "bid_size": 1, "ask": 99, "ask_size": 1}),
             sft_target(rows[3])]
    print("states:", len(rows))
    print("format:", fmt(None, comps, **kw))
    print("edge:  ", [round(x, 3) for x in edge(None, comps, **kw)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
    ap.add_argument("--label", default="qwen3-1.7b-grpo")
    ap.add_argument("--games", nargs="+", default=["dice", "cards"])
    ap.add_argument("--episodes", type=int, default=400, help="training episodes per game")
    ap.add_argument("--sft-steps", type=int, default=150)
    ap.add_argument("--sft-batch", type=int, default=8)
    ap.add_argument("--grpo-steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=96)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ensure_package()
    if args.dry_run:
        dry_run(args)
        return
    if not Path("/kaggle").exists() and "MMB_FORCE" not in os.environ:
        raise SystemExit("GPU training is meant for Kaggle/CUDA; use --dry-run locally")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)
    train(args)


if __name__ == "__main__":
    main()
