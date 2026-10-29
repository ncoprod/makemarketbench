"""Assemble a self-contained Kaggle script kernel in train/kaggle/.

The pure-Python ``makemarket`` package is embedded as a base64 tarball, so the
kernel needs neither the GitHub repo nor a C++ toolchain. Push with
``kaggle kernels push -p train/kaggle`` (after editing the ``id`` username).
"""

from __future__ import annotations

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
    OUT.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "train" / "grpo_unsloth_kaggle.py").read_text()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    src = src.replace('SOURCE_COMMIT = ""', f'SOURCE_COMMIT = "{commit}"')
    marker = 'PAYLOAD = ""'
    assert marker in src
    (OUT / "grpo_unsloth_kaggle.py").write_text(src.replace(marker, f'PAYLOAD = "{payload()}"', 1))
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
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT}/grpo_unsloth_kaggle.py and kernel-metadata.json")


if __name__ == "__main__":
    main()
