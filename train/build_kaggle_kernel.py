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
    args = ap.parse_args()
    out = (ROOT / "runs" / "kaggle_smoke" if args.smoke else
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
    (out / "grpo_unsloth_kaggle.py").write_text(src.replace(marker, f'PAYLOAD = "{payload()}"', 1))
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
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/grpo_unsloth_kaggle.py and kernel-metadata.json")


if __name__ == "__main__":
    main()
