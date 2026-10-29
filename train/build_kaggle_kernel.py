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
    args = ap.parse_args()
    out = (ROOT / "runs" / "kaggle_evaluate_staged" if args.evaluate_staged else
           ROOT / "runs" / "kaggle_smoke" if args.smoke else
           ROOT / "runs" / "kaggle_train_only" if args.train_only else OUT)
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
    if flags:
        src = src.replace("    main()\n", f"    sys.argv[1:1] = {flags!r}\n    main()\n")
    marker = 'PAYLOAD = ""'
    assert marker in src
    src = src.replace(marker, f'PAYLOAD = "{payload()}"', 1)
    if args.evaluate_staged:
        src = f'''import json, os, subprocess, sys, time
from pathlib import Path

started = time.monotonic()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)
core = Path("/kaggle/working/evaluate_core.py")
core.write_text({src!r})
workers = []
for gpu, (directory, label) in enumerate([
    ("warmstart_adapter", "qwen3.5-4b-staged-sft"),
    ("lora_adapter", "qwen3.5-4b-staged-grpo"),
]):
    adapters = list(Path("/kaggle/input").rglob(directory + "/adapter_config.json"))
    assert len(adapters) == 1, (directory, adapters)
    env = {{**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "MMB_WORK": "/kaggle/working/" + label}}
    command = [sys.executable, str(core), "--eval-only", "--adapter", str(adapters[0].parent),
               "--label", label, "--eval-seed", "80000", "--eval-episodes", "100"]
    workers.append(subprocess.Popen(command, env=env))
deadline = started + 10800
try:
    codes = [worker.wait(timeout=max(1, deadline - time.monotonic())) for worker in workers]
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.kill()
        worker.wait()
    Path("/kaggle/working/pair_runtime.json").write_text(json.dumps({{
        "wall_seconds": time.monotonic() - started, "budget_seconds": 10800,
        "exit_codes": [worker.returncode for worker in workers],
        "command": [sys.executable, *sys.argv], "source_commit": {commit!r},
        "seed": 0, "model_revision": "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636",
    }}, indent=2) + "\\n")
if any(codes):
    raise RuntimeError(f"Evaluation worker exit codes: {{codes}}")
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
        meta.update(id=meta["id"] + "-validate-staged", title=meta["title"] + " validate staged",
                    kernel_sources=["nicolascoquelet/makemarketbench-grpo-qwen3-train-only"])
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/grpo_unsloth_kaggle.py and kernel-metadata.json")


if __name__ == "__main__":
    main()
