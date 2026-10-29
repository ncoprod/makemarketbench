"""Small execution receipts for reproducible experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def receipt(config: dict, model_revision: str | None = None) -> dict:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    versions = {}
    for name in ("numpy", "openai", "mlx", "mlx-lm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"command": shlex.join([sys.executable, *sys.argv]), "config": config,
            "model_revision": model_revision, "git_commit": commit,
            "uncommitted_diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": platform.python_version(), "versions": versions}
