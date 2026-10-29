"""Record official account quota without credentials or signed output URLs.

Run with the Python environment that provides the Kaggle CLI. Submission times
and script runtimes are retained separately from authoritative account billing.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    api = KaggleApi()
    api.authenticate()
    started = datetime.now(timezone.utc).isoformat()
    with api.build_kaggle_client() as client:
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    record = {
        "recorded_at": started,
        "quota_method": "official SDK GetAcceleratorQuotaStatistics",
        "provenance": {
            "command": [sys.executable, *sys.argv], "seed": None, "model_revision": None,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "quota": {
            "refresh_at": quota.quota_refresh_time.isoformat(),
            "gpu_seconds_used": quota.gpu_quota.time_used.total_seconds(),
            "gpu_seconds_reserved": quota.gpu_quota.time_reserved.total_seconds(),
            "gpu_seconds_total": quota.gpu_quota.total_time_allowed.total_seconds(),
        },
    }
    folder = Path("results/training")
    (folder / "kaggle_account_budget_snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
    path = folder / "kaggle_budget.json"
    ledger = json.loads(path.read_text())
    ledger["quota_snapshots"].append(record)
    ledger.update(recorded_at=started, provenance=record["provenance"])
    path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(record["quota"]))


if __name__ == "__main__":
    main()
