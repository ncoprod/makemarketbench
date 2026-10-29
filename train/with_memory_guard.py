"""Stop our MLX child if macOS reports more than 7 GiB physical footprint.

macOS RSS excludes much of Metal memory; top's MEM column includes it. Samples
are rounded and periodic, so this is a guard, not an exact peak measurement.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from makemarket.provenance import file_sha256, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True)
    ap.add_argument("--max-gib", type=float, default=7)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    record = {"provenance": receipt(vars(args)), "source_sha256": file_sha256(__file__),
              "status": "running", "sampled_footprint_peak_bytes_approx": 0}
    started = time.monotonic()
    child = subprocess.Popen(args.command)
    record["child_pid"] = child.pid
    try:
        while child.poll() is None:
            output = subprocess.check_output(
                ["top", "-l", "1", "-pid", str(child.pid), "-stats", "pid,mem"], text=True)
            match = re.search(rf"^{child.pid}\s+([\d.]+)([BKMG])", output, re.MULTILINE)
            if match:
                size = float(match[1]) * 1024 ** "BKMG".index(match[2])
                record["sampled_footprint_peak_bytes_approx"] = max(
                    size, record["sampled_footprint_peak_bytes_approx"])
                record["wall_seconds"] = time.monotonic() - started
                Path(args.result).write_text(json.dumps(record, indent=2) + "\n")
                if size > args.max_gib * 1024**3:
                    record["status"] = "physical_memory_limit"
                    child.kill()
                    break
            time.sleep(5)
    finally:
        if child.poll() is None:
            child.kill()
        code = child.wait()
        if record["status"] == "running":
            record["status"] = "completed" if code == 0 else "failed"
        record.update(returncode=code, wall_seconds=time.monotonic() - started)
        Path(args.result).write_text(json.dumps(record, indent=2) + "\n")
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
