#!/usr/bin/env bash
# Serve a base model plus the GRPO LoRA adapter with vLLM (CUDA GPU required),
# then evaluate it with the standard harness through the OpenAI-compatible API.
#
#   bash train/serve_vllm.sh /path/to/lora_adapter [base_model]
#
# The adapter is the `lora_adapter/` directory written by
# train/grpo_unsloth_kaggle.py. Qwen3 thinking is disabled through the chat
# template so the served model matches the training format.
set -euo pipefail
ADAPTER=${1:?usage: serve_vllm.sh ADAPTER_DIR [BASE_MODEL]}
BASE=${2:-Qwen/Qwen3-1.7B}
PORT=${PORT:-8000}

pip install -q "vllm>=0.8"
vllm serve "$BASE" --port "$PORT" --enable-lora --max-lora-rank 16 \
  --lora-modules mmb-grpo="$ADAPTER" --max-model-len 2048 \
  --gpu-memory-utilization 0.85 > vllm.log 2>&1 &
PID=$!
until curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null; do sleep 5; done

for g in dice cards; do
  mmb eval --game "$g" --llm mmb-grpo --label qwen3-1.7b-grpo --seeds 0-49 \
    --base-url "http://127.0.0.1:$PORT/v1" --api-key-env NONE --temperature 0 \
    --max-tokens 256 --workers 4 \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
done
mmb score --game dice && mmb score --game cards
kill "$PID"
