"""Export the cached responses of an abandoned request variant to results/.

Used for nemotron-3-ultra with thinking enabled: its first leaderboard requests
were cached before the run switched to the no-think variant. The responses are
copied verbatim (content, finish reason, tokens, latency) with a summary, so
the README statement about them can be audited.

    python scripts/export_aborted_run.py cache/llm/nvidia_nemotron-3-ultra-550b-a55b \\
        results/aborted/nemotron-3-ultra-thinking.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from makemarket.llm import extract_json


def main(src: str, dst: str) -> None:
    rows = []
    for f in sorted(Path(src).glob("*.json")):
        d = json.loads(f.read_text())
        try:
            extract_json(d["content"])
            valid = True
        except ValueError:
            valid = False
        rows.append({"request_hash": f.stem, "finish_reason": d["finish_reason"],
                     "completion_tokens": d["completion_tokens"], "latency_s": round(d["latency"], 1),
                     "has_valid_json": valid, "content": d["content"]})
    lat = sorted(r["latency_s"] for r in rows)
    summary = {"requests": len(rows),
               "hit_token_cap": sum(r["finish_reason"] == "length" for r in rows),
               "valid_answers": sum(r["has_valid_json"] for r in rows),
               "latency_s_min": lat[0], "latency_s_max": lat[-1]}
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": src, "summary": summary, "responses": rows}, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:3])
