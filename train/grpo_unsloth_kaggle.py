"""GRPO fine-tuning of Qwen3.5-4B (4-bit QLoRA, Unsloth + TRL).

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
  5. Save LoRA adapters and evaluate on held-out seeds 200-399 with
     the same prompts, parser and scorer as the API harness. Raw episodes are
     written in the same JSONL format as results/raw/<game>/ for ``mmb score``.

CPU checks cover rewards and result export. Actual runtime and peak CUDA memory
are written to runtime.json, including failed runs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
import random
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

PAYLOAD = ""  # base64 tar.gz of src/makemarket, filled in by build_kaggle_kernel.py
SOURCE_COMMIT = ""

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
def _config(cls, **kw):
    """Build a TRL config, tolerating renamed/removed fields across TRL versions."""
    import inspect

    params = inspect.signature(cls.__init__).parameters
    if "max_seq_length" in kw and "max_seq_length" not in params and "max_length" in params:
        kw["max_length"] = kw.pop("max_seq_length")
    dropped = [k for k in kw if k not in params]
    for k in dropped:
        kw.pop(k)
    if dropped:
        print(f"[config] {cls.__name__}: ignoring unsupported fields {dropped}")
    return cls(**kw)


def _trainer(cls, tokenizer, **kw):
    """Pass the tokenizer as processing_class (new TRL) or tokenizer (old TRL)."""
    import inspect

    params = inspect.signature(cls.__init__).parameters
    key = "processing_class" if "processing_class" in params else "tokenizer"
    return cls(**{key: tokenizer}, **kw)


def render(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def execution(args) -> dict:
    return {"command": [sys.executable, *sys.argv], "config": vars(args),
            "seed": args.seed, "source_commit": SOURCE_COMMIT,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_revision": getattr(args, "revision", None),
            "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "embedded_package_sha256": hashlib.sha256(base64.b64decode(PAYLOAD)).hexdigest()}


def train(args, report_runtime) -> None:
    from unsloth import FastLanguageModel  # noqa: I001  (must be imported before trl)
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback, TrainerCallback
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    report_runtime("loading", 0)

    class RuntimeCallback(TrainerCallback):
        def __init__(self, phase):
            self.phase = phase

        def on_log(self, args, state, control, **kwargs):
            report_runtime(self.phase, state.global_step)

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in checkpoint.iterdir() if p.is_file()}
            with (WORK / "checkpoints.jsonl").open("a") as log:
                log.write(json.dumps({"phase": self.phase, "step": state.global_step,
                                      "checkpoint": str(checkpoint), "sha256": files,
                                      "recorded_at": datetime.now(timezone.utc).isoformat()}) + "\n")
            report_runtime(self.phase, state.global_step)

    load_options = {"offload_layers": args.offload_layers} if args.offload_layers else {}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, revision=args.revision, max_seq_length=args.max_seq_len, load_in_4bit=True,
        fast_inference=False, device_map=args.device_map, **load_options)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    adapters = [p for p in Path("/kaggle/input").rglob("adapter_model.safetensors")
                if json.loads(p.with_name("adapter_config.json").read_text())[
                    "base_model_name_or_path"] in (args.base_model, args.model)]
    if args.adapter:
        adapter = Path(args.adapter)
        base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
        if base not in (args.base_model, args.model):
            raise ValueError(f"Adapter base {base!r} does not match {args.model!r}")
        adapters = [adapter / "adapter_model.safetensors"]
        args.adapter_sha256 = hashlib.sha256(adapters[0].read_bytes()).hexdigest()
    elif len(adapters) > 1:
        raise ValueError("Multiple compatible adapters found; select one with --adapter")
    if adapters:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapters[0].parent), is_trainable=True)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print(f"[warm start] PEFT adapter: {adapters[0]}", flush=True)
    else:
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=args.seed)

    if args.eval_only:
        if not adapters:
            raise ValueError("--eval-only requires a compatible saved adapter")
        FastLanguageModel.for_inference(model)
        report_runtime("evaluation", 0)
        if args.check_batching:
            check_batching(model, tokenizer, args)
        evaluate(model, tokenizer, args)
        return

    rows = build_prompt_set(args.episodes, args.games)
    print(f"prompt set: {len(rows)} states")

    if args.sft_steps > 0 and not args.resume and (not adapters or args.sft_resume):
        def example(row):
            return {"prompt": render(tokenizer, row["messages"]),
                    "completion": sft_target(row) + tokenizer.eos_token}

        sft_rows = [example(r) for r in rows[: args.sft_steps * args.sft_batch]]
        validation = {}
        callbacks = [RuntimeCallback("sft")]
        if args.sft_validation_episodes:
            validation["eval_dataset"] = Dataset.from_list([
                example(r) for r in build_prompt_set(args.sft_validation_episodes, args.games, 60_000)])
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
        evaluation = (dict(eval_strategy="steps", eval_steps=min(50, args.save_steps),
                           load_best_model_at_end=True, metric_for_best_model="eval_loss",
                           greater_is_better=False, per_device_eval_batch_size=1,
                           restore_callback_states_from_checkpoint=True)
                      if validation else {})
        sft = _trainer(SFTTrainer, tokenizer, model=model, train_dataset=Dataset.from_list(sft_rows),
                       callbacks=callbacks, **validation,
                       args=_config(SFTConfig, output_dir=str(WORK / "sft"), max_steps=args.sft_steps,
                                    per_device_train_batch_size=args.sft_batch, learning_rate=1e-4,
                                    gradient_accumulation_steps=args.sft_accumulation,
                                    logging_steps=10, save_strategy="steps", save_steps=min(50, args.save_steps),
                                    save_total_limit=2, report_to="none",
                                    seed=args.seed, data_seed=args.seed,
                                    completion_only_loss=True, max_seq_length=args.max_seq_len,
                                    **evaluation))
        (WORK / "sft_config.json").write_text(json.dumps(
            {**execution(args), "trainer_config": sft.args.to_dict()}, indent=2) + "\n")
        sft.train(resume_from_checkpoint=args.sft_resume or None)
        del sft

    if not args.resume:
        model.save_pretrained(str(WORK / "warmstart_adapter"))
        tokenizer.save_pretrained(str(WORK / "warmstart_adapter"))
        if not args.train_only:
            FastLanguageModel.for_inference(model)
            report_runtime("warmstart_evaluation", 0)
            evaluate(model, tokenizer, args, label=args.label + "-warmstart")
            FastLanguageModel.for_training(model)

    if args.grpo_steps == 0:
        return

    grpo_rows = [{"prompt": render(tokenizer, r["messages"]), "game": r["game"], "values": r["values"],
                  "probs": r["probs"], "m_pub": r["m_pub"], "ref_quote": r["ref_quote"]}
                 for r in rows[args.sft_steps * args.sft_batch:]]
    cfg = _config(
        GRPOConfig, output_dir=str(WORK / "grpo"), learning_rate=args.lr, max_steps=args.grpo_steps,
        per_device_train_batch_size=args.num_generations, gradient_accumulation_steps=1,
        num_generations=args.num_generations, max_prompt_length=args.max_seq_len - args.max_completion,
        max_completion_length=args.max_completion, temperature=1.0, logging_steps=5,
        save_steps=args.grpo_save_steps or args.save_steps, save_total_limit=2, report_to="none", lr_scheduler_type="cosine", warmup_ratio=0.05,
        optim="adamw_8bit", seed=args.seed, data_seed=args.seed)
    trainer = _trainer(GRPOTrainer, tokenizer, model=model, reward_funcs=make_reward_fns(),
                       callbacks=[RuntimeCallback("grpo")],
                       args=cfg, train_dataset=Dataset.from_list(grpo_rows))
    (WORK / "grpo_config.json").write_text(json.dumps(
        {**execution(args), "trainer_config": trainer.args.to_dict()}, indent=2) + "\n")
    trainer.train(resume_from_checkpoint=args.resume or None)
    model.save_pretrained(str(WORK / "lora_adapter"))
    tokenizer.save_pretrained(str(WORK / "lora_adapter"))
    if not args.train_only:
        FastLanguageModel.for_inference(model)
        report_runtime("evaluation", 0)
        evaluate(model, tokenizer, args)


# --------------------------------------------------------------------- evaluation
class HFQuoter:
    """Same prompts, parser and one repair retry as makemarket.llm.LLMQuoter."""

    def __init__(self, model, tokenizer, name: str, max_new_tokens: int = 128) -> None:
        self.model, self.tok, self.name, self.max_new = model, tokenizer, name, max_new_tokens
        self.cache = {}

    def reset(self, episode_seed: int) -> None:
        pass

    def _gen(self, messages) -> str:
        import torch

        ids = self.tok(render(self.tok, messages), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def quote(self, obs):
        return self.quote_many([obs])[0]

    def _gen_many(self, messages):
        if len(messages) == 1:
            return [self._gen(messages[0])]
        import torch

        self.tok.padding_side = "left"
        ids = self.tok([render(self.tok, msgs) for msgs in messages],
                       padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**ids, max_new_tokens=self.max_new, do_sample=False)
        return self.tok.batch_decode(output[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def quote_many(self, observations):
        from makemarket.llm import parse_quote
        from makemarket.prompts import OUTPUT_SPEC, messages_for

        prompts = [json.dumps(messages_for(obs)) for obs in observations]
        cached = set(self.cache)
        pending = {prompt: {"messages": json.loads(prompt), "attempts": [],
                            "max_size": obs.game.params.max_size}
                   for prompt, obs in zip(prompts, observations) if prompt not in self.cache}
        for _ in range(2):
            if not pending:
                break
            keys = list(pending)
            texts = self._gen_many([pending[key]["messages"] for key in keys])
            assert len(texts) == len(keys)
            for key, text in zip(keys, texts):
                row = pending[key]
                row["attempts"].append({"content": text})
                try:
                    quote = parse_quote(text, row["max_size"])
                except ValueError as exc:
                    quote = None
                    row["attempts"][-1]["error"] = str(exc)
                    row["messages"] += [{"role": "assistant", "content": text},
                                        {"role": "user", "content": f"Your answer was invalid: {exc}. {OUTPUT_SPEC}"}]
                self.cache[key] = quote, {"attempts": row["attempts"],
                                          "format_error": len(row["attempts"]) > 1,
                                          "failed": quote is None}
                if quote is not None:
                    del pending[key]
        return [(self.cache[key][0], {**self.cache[key][1], "cached": key in cached}) for key in prompts]


def check_batching(model, tokenizer, args):
    """Check padding on training states before committing held-out GPU compute."""
    from makemarket.games import make_game
    from makemarket.market import episode_steps
    from makemarket.players import BayesQuoter

    observations = []
    for name in args.games:
        for index, round_index in enumerate((0, 1, 3, 5)):
            teacher = BayesQuoter()
            teacher.reset(20000 + index)
            steps = episode_steps(make_game(name), 20000 + index, "teacher", "py")
            obs = next(steps)
            for _ in range(round_index):
                obs = steps.send(teacher.quote(obs))
            observations.append(obs)
    serial = HFQuoter(model, tokenizer, args.label)
    batched = HFQuoter(model, tokenizer, args.label)
    started = time.monotonic()
    expected = [serial.quote(obs) for obs in observations]
    serial_seconds = time.monotonic() - started
    started = time.monotonic()
    actual = []
    for offset in range(0, len(observations), args.eval_batch_size):
        actual.extend(batched.quote_many(observations[offset:offset + args.eval_batch_size]))
    def encode(answers):
        return [quote.to_dict() if quote is not None else None for quote, _ in answers]

    record = {**execution(args), "training_seeds": [20000, 20003], "rounds": [0, 1, 3, 5],
              "serial_seconds": serial_seconds, "batch_seconds": time.monotonic() - started,
              "serial_quotes": encode(expected), "batch_quotes": encode(actual),
              "serial_attempts": [info["attempts"] for _, info in expected],
              "batch_attempts": [info["attempts"] for _, info in actual],
              "timing_note": "Serial runs first and includes warm-up; diagnostic, not a steady-state speedup estimate."}
    record["quotes_equal"] = record["serial_quotes"] == record["batch_quotes"]
    (WORK / "batch_parity.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["quotes_equal"]:
        raise ValueError("Serial/batched quotes differ on training states; inspect batch_parity.json")


def resume_evaluation(args, label) -> None:
    import shutil

    from makemarket.cli import slug
    from makemarket.evaluate import load_results
    from makemarket.games import make_game

    if not getattr(args, "evaluation_prefix", None):
        return
    args.evaluation_prefix_sha256 = {}
    for game in args.games:
        relative = Path("results/raw") / game / f"{slug(label)}.jsonl"
        source, destination = Path(args.evaluation_prefix) / relative, WORK / relative
        if not source.exists():
            continue
        rows = load_results(source)
        seeds = [row["seed"] for row in rows]
        assert len(set(seeds)) == len(seeds), "Duplicate resumed seeds"
        assert set(seeds) <= set(range(args.eval_seed, args.eval_seed + args.eval_episodes))
        for row in rows:
            assert row["game"] == game and row["player"] == label
            cfg = row["provenance"]["config"]
            for key in ("model", "revision", "adapter_sha256", "eval_batch_size", "seed"):
                assert cfg[key] == getattr(args, key), f"Resumed {key} differs"
            assert cfg.get("inference_precision", "native") == getattr(args, "inference_precision", "native")
            episode = make_game(game).episode(row["seed"])
            assert row["value"] == episode.value and len(row["rounds"]) == 6
            assert [r["reveal"] for r in row["rounds"]] == episode.reveals
        assert not destination.exists(), "Do not overwrite acquired episodes"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        args.evaluation_prefix_sha256[game] = hashlib.sha256(source.read_bytes()).hexdigest()


def evaluate(model, tokenizer, args, label=None) -> None:
    from makemarket.analysis import player_diagnostics
    from makemarket.cli import slug
    from makemarket.evaluate import load_results, run_batched_player, run_player
    from makemarket.games import make_game
    from makemarket.players import BayesQuoter
    from makemarket.scoring import markdown_table, summarize

    label = label or args.label
    resume_evaluation(args, label)
    provenance = {**execution(args), "label": label}
    # Greedy decoding permits reuse of identical prompts within this adapter.
    player = HFQuoter(model, tokenizer, label)

    for g in args.games:
        out = WORK / "results" / "raw" / g / f"{slug(label)}.jsonl"
        t0 = time.time()
        seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
        if getattr(args, "eval_batch_size", 1) > 1:
            eps = run_batched_player(make_game(g), player, seeds, out, args.eval_batch_size, provenance)
        else:
            eps = run_player(make_game(g), lambda: player, seeds, out,
                             workers=1, backend="py", provenance=provenance)
        tot = sum(sum(r["exp_expected_pnl"] for r in e["rounds"]) for e in eps) / len(eps)
        print(f"[eval] {g}: {len(eps)} episodes, mean E[PnL]={tot:.2f} ({time.time() - t0:.0f}s) -> {out}")
        seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
        run_player(make_game(g), BayesQuoter, seeds,
                   out.with_name("bayes.jsonl"), workers=1, backend="py", provenance=provenance)
        players = {eps[0]["player"]: eps for f in sorted(out.parent.glob("*.jsonl"))
                   if (eps := load_results(f))}
        rows = summarize(players)
        for row in rows:
            row.update(player_diagnostics(players[row["player"]]))
        dest = WORK / "results" / f"leaderboard_{g}"
        dest.with_suffix(".json").write_text(json.dumps(
            {"game": g, "seeds": seeds, "rows": rows, "provenance": provenance}, indent=2) + "\n")
        dest.with_suffix(".md").write_text(markdown_table(rows) + "\n")


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
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636")
    ap.add_argument("--label", default="qwen3.5-4b-grpo")
    ap.add_argument("--games", nargs="+", default=["dice", "cards"])
    ap.add_argument("--episodes", type=int, default=400, help="training episodes per game")
    ap.add_argument("--sft-steps", type=int, default=150)
    ap.add_argument("--sft-batch", type=int, default=8)
    ap.add_argument("--sft-accumulation", type=int, default=2)
    ap.add_argument("--sft-validation-episodes", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--grpo-save-steps", type=int, help="GRPO checkpoint cadence, independent of SFT validation")
    ap.add_argument("--budget-seconds", type=int, default=0, help="wall cap, preserving periodic checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-map", default="sequential", help="balanced splits a large model across GPUs")
    ap.add_argument("--offload-layers", choices=["auto"], help="stream layers from host RAM when GPU loading fails")
    ap.add_argument("--grpo-steps", type=int, default=300, help="zero saves the SFT adapter without GRPO")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=96)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--check-batching", action="store_true", help="compare serial/batched training-state quotes first")
    ap.add_argument("--eval-seed", type=int, default=200)
    ap.add_argument("--evaluation-prefix", help="verified prior policy results; acquire missing seeds only")
    ap.add_argument("--resume", help="GRPO checkpoint directory, including optimizer state")
    ap.add_argument("--sft-resume", help="SFT checkpoint directory, including optimizer state")
    ap.add_argument("--adapter", help="explicit compatible PEFT adapter directory")
    stage = ap.add_mutually_exclusive_group()
    stage.add_argument("--train-only", action="store_true", help="save both adapters; evaluate in a later session")
    stage.add_argument("--eval-only", action="store_true", help="evaluate a saved adapter without training")
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args()
    WORK.mkdir(parents=True, exist_ok=True)
    ensure_package()
    if args.dry_run:
        dry_run(args)
        return
    if not Path("/kaggle").exists() and "MMB_FORCE" not in os.environ:
        raise SystemExit("GPU training is meant for Kaggle/CUDA; use --dry-run locally")
    started = time.time()
    if args.budget_seconds:
        def expire(signum, frame):
            raise TimeoutError("Registered training wall budget exhausted")
        signal.signal(signal.SIGALRM, expire)
        signal.alarm(args.budget_seconds)
    record = {**execution(args), "status": "running",
              "execution_started_at": datetime.now(timezone.utc).isoformat()}

    def report_runtime(phase=None, step=None):
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        record["wall_seconds"] = time.time() - started
        if phase is not None:
            record.update(phase=phase, phase_step=step)
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            record["memory_by_gpu"] = [
                {"device": i, "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(i)}
                for i in range(torch.cuda.device_count())]
        temporary = WORK / "runtime.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(WORK / "runtime.json")

    report_runtime()
    try:
        packages = ["unsloth"]
        if args.model == "unsloth/Qwen3.8-27B-unsloth-bnb-4bit":
            # Reuse the stack that actually completed updates on two T4s.
            packages = ["unsloth==2026.10.10", "unsloth_zoo==2026.10.10", "trl==1.15.0",
                        "transformers==5.16.1", "peft==0.20.0", "bitsandbytes==0.50.2",
                        "datasets==4.8.5", "accelerate==1.14.0", "triton==3.6.0",
                        "torch==2.11.0+cu128", "--extra-index-url", "https://download.pytorch.org/whl/cu128"]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle did not allocate a CUDA GPU")
        record["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        record["gpu_capacity_bytes"] = [torch.cuda.get_device_properties(i).total_memory
                                        for i in range(torch.cuda.device_count())]
        record["cuda_version"] = torch.version.cuda
        record["versions"] = {p: importlib.metadata.version(p)
                              for p in ("torch", "unsloth", "unsloth_zoo", "trl", "transformers", "peft",
                                        "bitsandbytes", "datasets", "accelerate", "triton")}
        train(args, report_runtime)
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["execution_finished_at"] = datetime.now(timezone.utc).isoformat()
        report_runtime()


if __name__ == "__main__":
    main()
