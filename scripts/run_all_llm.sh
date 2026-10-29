#!/usr/bin/env bash
# Full API evaluation schedule (sequential, at most 2 requests in flight).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -u scripts/run_llm_queue.py --games dice
# ablation: same model, public posterior moments given in the prompt
.venv/bin/mmb eval --game dice --llm openai/gpt-oss-20b --hint --label "openai/gpt-oss-20b +hint" \
  --extra-body '{"reasoning_effort": "low"}' --workers 2
# Figgie-lite seat 0 (4 requests per episode)
.venv/bin/mmb figgie --seeds 0-49 --llm openai/gpt-oss-20b --extra-body '{"reasoning_effort": "low"}'
$PY -u scripts/run_llm_queue.py --games cards
echo "ALL DONE"
