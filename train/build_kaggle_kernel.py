"""Assemble a self-contained Kaggle script kernel in train/kaggle/.

The pure-Python ``makemarket`` package is embedded as a base64 tarball, so the
kernel needs neither the GitHub repo nor a C++ toolchain. Push with
``kaggle kernels push -p train/kaggle`` (after editing the ``id`` username).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "train" / "kaggle"


def payload() -> str:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tf:
        for f in sorted((ROOT / "src" / "makemarket").rglob("*.py")):
            info = tf.gettarinfo(str(f), arcname=str(f.relative_to(ROOT / "src")))
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            with f.open("rb") as content:
                tf.addfile(info, content)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    stage = ap.add_mutually_exclusive_group()
    stage.add_argument("--smoke", action="store_true", help="one GRPO update, validation seeds only")
    stage.add_argument("--train-only", action="store_true", help="save adapters before separate evaluation sessions")
    stage.add_argument("--evaluate-staged", action="store_true", help="validate staged SFT and GRPO on one GPU each")
    stage.add_argument("--qwen38-train", action="store_true", help="bounded newest-family SFT and GRPO")
    stage.add_argument("--qwen38-continue", action="store_true", help="resume retained newest-family optimizer states")
    stage.add_argument("--qwen38-evaluate", action="store_true", help="evaluate saved newest-family adapters")
    ap.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    ap.add_argument("--evaluation-budget-seconds", type=int, default=10800)
    ap.add_argument("--out", type=Path, help="output directory; useful for local CI without modifying tracked kernels")
    ap.add_argument("--checkpoint-kernel", default="nicolascoquelet/makemarketbench-qwen3-8-27b-training")
    ap.add_argument("--evaluate-adapter", choices=["sft", "grpo"], help="required for the selected newest-family test")
    args = ap.parse_args()
    if args.qwen38_evaluate and args.evaluation_split == "test" and not args.evaluate_adapter:
        ap.error("Freeze the validation winner first, then specify --evaluate-adapter for its test")
    assert 0 < args.evaluation_budget_seconds <= 43200
    out = (ROOT / "runs" / "kaggle_qwen38_continue" if args.qwen38_continue else
           ROOT / "runs" / "kaggle_qwen38_train" if args.qwen38_train else
           ROOT / "runs" / "kaggle_qwen38_evaluate" if args.qwen38_evaluate else
           ROOT / "runs" / "kaggle_evaluate_staged" if args.evaluate_staged else
           ROOT / "runs" / "kaggle_smoke" if args.smoke else
           ROOT / "runs" / "kaggle_train_only" if args.train_only else OUT)
    if args.out:
        out = args.out
    out.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "train" / "grpo_unsloth_kaggle.py").read_text()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    src = src.replace('SOURCE_COMMIT = ""', f'SOURCE_COMMIT = "{commit}"')
    flags = []
    if args.smoke:
        flags = ["--sft-steps", "0", "--grpo-steps", "1", "--num-generations", "2",
                 "--episodes", "4", "--games", "dice", "--eval-episodes", "1",
                 "--eval-seed", "80000", "--label", "qwen3.5-4b-grpo-smoke"]
    elif args.train_only:
        flags = ["--train-only", "--label", "qwen3.5-4b-grpo-staged"]
    elif args.qwen38_train:
        flags = ["--model", "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
                 "--base-model", "Qwen/Qwen3.8-27B", "--revision", "8aa5f05d26b7205477066e1449e0af13f762a299",
                 "--device-map", "unsloth_balanced", "--sft-steps", "800", "--sft-batch", "1",
                 "--sft-accumulation", "1", "--sft-validation-episodes", "4", "--save-steps", "50",
                 "--grpo-save-steps", "5",
                 "--grpo-steps", "96", "--num-generations", "2", "--max-seq-len", "1024",
                 "--lora-rank", "8", "--seed", "0", "--budget-seconds", "37800",
                 "--train-only", "--label", "qwen3.8-27b-sft-grpo"]
    if flags:
        src = src.replace("    main()\n", f"    sys.argv[1:1] = {flags!r}\n    main()\n")
    marker = 'PAYLOAD = ""'
    assert marker in src
    src = src.replace(marker, f'PAYLOAD = "{payload()}"', 1)
    if args.evaluate_staged or args.qwen38_evaluate:
        newest = args.qwen38_evaluate
        family = "qwen3.8-27b" if newest else "qwen3.5-4b-staged"
        seed, episodes = (200, 200) if args.evaluation_split == "test" else (80000, 100)
        batch_size = 1 if newest else 4
        adapters = [("warmstart_adapter", f"{family}-sft"), ("lora_adapter", f"{family}-grpo")]
        if newest and args.evaluate_adapter:
            adapters = [adapters[0 if args.evaluate_adapter == "sft" else 1]]
        model_flags = (["--model", "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
                        "--base-model", "Qwen/Qwen3.8-27B", "--revision",
                        "8aa5f05d26b7205477066e1449e0af13f762a299",
                        "--device-map", "unsloth_balanced", "--lora-rank", "8"] if newest else [])
        revision = "8aa5f05d26b7205477066e1449e0af13f762a299" if newest else "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"
        src = f'''import json, os, subprocess, sys, time
from pathlib import Path

started = time.monotonic()
core = Path("/kaggle/working/evaluate_core.py")
core.write_text({src!r})
workers = []
deadline = started + {args.evaluation_budget_seconds}
for gpu, (directory, label) in enumerate({adapters!r}):
    adapters = list(Path("/kaggle/input").rglob(directory + "/adapter_config.json"))
    assert len(adapters) == 1, (directory, adapters)
    work = Path("/kaggle/working") / label
    work.mkdir(exist_ok=True)
    env = {{**os.environ, "MMB_WORK": str(work)}}
    if not {newest!r}:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [sys.executable, str(core), "--eval-only", "--adapter", str(adapters[0].parent),
               "--label", label, "--eval-seed", "{seed}", "--eval-episodes", "{episodes}",
               "--eval-batch-size", "{batch_size}", *{([] if newest else ['--check-batching'])!r},
               "--budget-seconds", str(max(1, int(deadline - time.monotonic()))),
               *{model_flags!r}]
    workers.append(subprocess.Popen(command, env=env, cwd=work))
    if {newest!r}:
        try:
            workers[-1].wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            workers[-1].kill()
            workers[-1].wait()
            break
try:
    codes = [worker.wait(timeout=max(1, deadline - time.monotonic())) for worker in workers]
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.kill()
        worker.wait()
    Path("/kaggle/working/pair_runtime.json").write_text(json.dumps({{
        "wall_seconds": time.monotonic() - started, "budget_seconds": {args.evaluation_budget_seconds},
        "exit_codes": [worker.returncode for worker in workers],
        "command": [sys.executable, *sys.argv], "source_commit": {commit!r},
        "seed": 0, "model_revision": {revision!r}, "evaluation_seed": {seed},
        "episodes_per_game": {episodes}, "batch_size": {batch_size},
    }}, indent=2) + "\\n")
if any(codes):
    raise RuntimeError(f"Evaluation worker exit codes: {{codes}}")
'''
    if args.qwen38_continue:
        continuation = (ROOT / "train" / "qwen38_continue.py").read_text()
        src = f'''import shutil, subprocess, sys
from pathlib import Path

sources = list(Path("/kaggle/input").rglob("train_core.py"))
assert len(sources) == 1, sources
source = sources[0].parent
work = Path("/kaggle/working")
for name in ["train_core.py", "runtime.json", "checkpoints.jsonl", "warmstart_adapter", "lora_adapter", "sft", "grpo"]:
    item = source / name
    if item.is_dir():
        shutil.copytree(item, work / name)
    elif item.is_file():
        shutil.copyfile(item, work / name)
launcher = work / "continue.py"
launcher.write_text({continuation!r})
subprocess.run([sys.executable, str(launcher), "--source-commit", {commit!r}], check=True)
'''
    if args.qwen38_train:
        src = f'''import subprocess, sys
from pathlib import Path

core = Path("/kaggle/working/train_core.py")
core.write_text({src!r})
with Path("/kaggle/working/train.log").open("w") as log:
    try:
        subprocess.run([sys.executable, str(core)], check=True, timeout=39000,
                       stdout=log, stderr=subprocess.STDOUT)
    finally:
        log.flush()
        print(Path("/kaggle/working/train.log").read_text()[-8000:])
'''
    (out / "grpo_unsloth_kaggle.py").write_text(src)
    meta = {
        "id": "nicolascoquelet/makemarketbench-grpo-qwen3",
        "title": "MakeMarketBench GRPO Qwen3",
        "code_file": "grpo_unsloth_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if args.smoke:
        meta.update(id=meta["id"] + "-smoke", title=meta["title"] + " smoke")
    elif args.train_only:
        meta.update(id=meta["id"] + "-train-only", title=meta["title"] + " train only")
    elif args.evaluate_staged:
        suffix = "final" if args.evaluation_split == "test" else "validate"
        meta.update(id=meta["id"] + f"-{suffix}-staged", title=meta["title"] + f" {suffix} staged",
                    kernel_sources=["nicolascoquelet/makemarketbench-grpo-qwen3-train-only"])
    elif args.qwen38_continue:
        meta.update(id="nicolascoquelet/makemarketbench-qwen3-8-27b-continue",
                    title="MakeMarketBench Qwen3.8 27B continuation",
                    kernel_sources=[args.checkpoint_kernel])
    elif args.qwen38_evaluate:
        suffix = "final" if args.evaluation_split == "test" else "validate"
        meta.update(id=f"nicolascoquelet/makemarketbench-qwen3-8-27b-{suffix}",
                    title=f"MakeMarketBench Qwen3.8 27B {suffix}",
                    kernel_sources=[args.checkpoint_kernel])
    elif args.qwen38_train:
        meta.update(id="nicolascoquelet/makemarketbench-qwen3-8-27b-training",
                    title="MakeMarketBench Qwen3.8 27B training")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/grpo_unsloth_kaggle.py and kernel-metadata.json")


if __name__ == "__main__":
    main()
