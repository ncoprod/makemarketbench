#!/usr/bin/env bash
# Local training on Apple silicon with MLX (tested on an M1 Pro, 16 GB).
#
#   stage 0  evaluate the untuned base model (no-think mode) on held-out seeds 0-49
#   stage 1  LoRA SFT on Bayes-reference demonstrations (train/make_sft_data.py)
#   stage 2  evaluate the SFT adapter
#   stage 3  DAgger round: the SFT model plays training seeds, the reference labels
#            the visited states, LoRA training resumes on the aggregated data
#   stage 4  evaluate the DAgger adapter
#
# Usage: bash train/mlx_pipeline.sh [first_stage] [last_stage]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-mlx-community/Qwen3-1.7B-4bit}  # or a local directory with the same files
TAG=${TAG:-qwen3-1.7b}
PORT=${PORT:-8081}
ITERS=${ITERS:-800}
DAGGER_ITERS=${DAGGER_ITERS:-200}
DAGGER_EPISODES=${DAGGER_EPISODES:-50}
BATCH=${BATCH:-2}
LR=${LR:-1e-4}
LAYERS=${LAYERS:-16}
SEEDS=${SEEDS:-0-49}
RUNS=train/runs
PY=.venv/bin/python
mkdir -p "$RUNS" logs

serve() {  # serve [adapter_path]
  local extra=()
  if [ -n "${1:-}" ]; then extra=(--adapter-path "$1"); fi
  .venv/bin/mlx_lm.server --model "$MODEL" ${extra[@]+"${extra[@]}"} --port "$PORT" \
    --chat-template-args '{"enable_thinking":false}' > "logs/mlx_server_$PORT.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null; then return 0; fi
    sleep 2
  done
  echo "server did not start" >&2; exit 1
}
stop() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }

evaluate() {  # evaluate label [adapter_path]
  serve "${2:-}"
  for g in dice cards; do
    .venv/bin/mmb eval --game "$g" --llm default_model --label "$1" --seeds "$SEEDS" \
      --base-url "http://127.0.0.1:$PORT/v1" --api-key-env NONE --temperature 0 \
      --max-tokens 256 --workers 1 --cache-dir cache/local
  done
  stop
}

FIRST=${1:-0}
LAST=${2:-4}
for stage in $(seq "$FIRST" "$LAST"); do
  echo "=== stage $stage ($(date +%H:%M:%S))"
  case $stage in
    0) evaluate "$TAG-base" ;;
    1) [ -f train/data/train.jsonl ] || $PY train/make_sft_data.py --episodes 800
       .venv/bin/mlx_lm.lora --model "$MODEL" --train --data train/data --mask-prompt \
         --iters "$ITERS" --batch-size "$BATCH" --learning-rate "$LR" --num-layers "$LAYERS" \
         --max-seq-length 1024 --grad-checkpoint --steps-per-report 10 --steps-per-eval 100 \
         --val-batches 10 --save-every 100 --adapter-path "$RUNS/$TAG-sft" --seed 0 ;;
    2) evaluate "$TAG-sft" "$RUNS/$TAG-sft" ;;
    3) serve "$RUNS/$TAG-sft"
       $PY train/dagger_data.py --base-url "http://127.0.0.1:$PORT/v1" --episodes "$DAGGER_EPISODES" \
         --out train/data_dagger --merge train/data/train.jsonl
       stop
       mkdir -p "$RUNS/$TAG-dagger"
       .venv/bin/mlx_lm.lora --model "$MODEL" --train --data train/data_dagger --mask-prompt \
         --iters "$DAGGER_ITERS" --batch-size "$BATCH" --learning-rate "$LR" --num-layers "$LAYERS" \
         --max-seq-length 1024 --grad-checkpoint --steps-per-report 10 --steps-per-eval 100 \
         --val-batches 10 --save-every 100 --resume-adapter-file "$RUNS/$TAG-sft/adapters.safetensors" \
         --adapter-path "$RUNS/$TAG-dagger" --seed 1 ;;
    4) evaluate "$TAG-dagger" "$RUNS/$TAG-dagger" ;;
  esac
done
echo "=== done ($(date +%H:%M:%S))"
