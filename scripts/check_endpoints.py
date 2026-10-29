"""Record which candidate models the endpoint actually serves.

Sends one tiny chat request per model, sequentially (one request in flight),
and writes the HTTP outcome to results/endpoint_check.json, so that the
README's list of models that could not be evaluated is auditable.

    source ~/.zshrc  # provides NVIDIA_API_KEY
    python scripts/check_endpoints.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

CANDIDATES = [
    "moonshotai/kimi-k2.6",
    "nvidia/nemotron-3-super-120b-a12b",
    "mistralai/mistral-large-2-instruct",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "deepseek-ai/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
]
OUT = Path("results/endpoint_check.json")


def main() -> None:
    from openai import OpenAI

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                    api_key=os.environ.get("NVIDIA_API_KEY", "missing"), timeout=120, max_retries=0)
    rows = []
    for model in CANDIDATES:
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=model, max_tokens=8, temperature=0,
                                               messages=[{"role": "user", "content": "Reply with OK."}])
            status, note = 200, f"finish={r.choices[0].finish_reason}"
        except Exception as e:
            status = getattr(e, "status_code", None)
            note = type(e).__name__
        rows.append({"model": model, "status": status, "note": note, "latency_s": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        time.sleep(2)
    OUT.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "endpoint": "https://integrate.api.nvidia.com/v1", "models": rows},
                              indent=2) + "\n")


if __name__ == "__main__":
    main()
