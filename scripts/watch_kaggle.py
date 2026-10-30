"""Poll a submitted kernel every 610 seconds and download terminal outputs."""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def terminal_status(output):
    match = re.search(r'KernelWorkerStatus\.(\w+)', output)
    return bool(match and match[1] in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "FAILED"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = Path(args.receipt)
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(Path("runs/kaggle_oauth_config").resolve())}
    for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_TOKEN_FILE"):
        env.pop(name, None)
    while True:
        record = json.loads(path.read_text())
        last = record["polls"][-1]["at"] if record["polls"] else record["submitted_at"]
        time.sleep(max(0, datetime.fromisoformat(last).timestamp() + 610 - time.time()))
        record = json.loads(path.read_text())
        command = ["kaggle", "kernels", "status", record["metadata"]["id"]]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        output = (result.stdout + result.stderr).strip()
        record["polls"].append({"at": datetime.now(timezone.utc).isoformat(), "command": command,
                                "exit_code": result.returncode, "output": output})
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(output, flush=True)
        if result.returncode or not terminal_status(output):
            continue
        command = ["kaggle", "kernels", "output", record["metadata"]["id"], "-p", args.out,
                   "--page-size", "200", "--file-pattern",
                   r"^(?!.*(?:mmb_src|unsloth_compiled_cache|\.virtual_documents)/).*"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        record["output_pull"] = {"command": command, "exit_code": result.returncode,
                                 "output": (result.stdout + result.stderr).strip()}
        record["status"] = "outputs_retrieved" if result.returncode == 0 else "output_pull_failed"
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(record["status"], flush=True)
        break


if __name__ == "__main__":
    main()
