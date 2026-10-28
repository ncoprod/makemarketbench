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

PAYLOAD = ""  # base64 tar.gz of src/makemarket, filled in by build_kaggle_kernel.py

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
