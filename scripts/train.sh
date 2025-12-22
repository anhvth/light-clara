#!/usr/bin/env bash
set -euo pipefail

# Minimal debug-focused launcher. Override any var via env or pass extra args.
STAGE="${STAGE:-stage1}"
DATASET_NAME="${DATASET_NAME:-apple/CLaRa_multi_stage}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
COMPRESS_RATE="${COMPRESS_RATE:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/debug}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-}"
DEBUG_MODE="${DEBUG_MODE:-False}"
DEBUG_EVERY_STEPS="${DEBUG_EVERY_STEPS:-5}"

args=(
  "--stage" "$STAGE"
  "--dataset_name" "$DATASET_NAME"
  "--dataset_split" "$DATASET_SPLIT"
  "--model_name" "$MODEL_NAME"
  "--compress_rate" "$COMPRESS_RATE"
  "--batch_size" "$BATCH_SIZE"
  "--max_steps" "$MAX_STEPS"
  "--save_steps" "$SAVE_STEPS"
  "--learning_rate" "$LEARNING_RATE"
  "--checkpoint_dir" "$CHECKPOINT_DIR"
  "--report_to" "$REPORT_TO"
  "--debug_mode" "$DEBUG_MODE"
  "--debug_every_steps" "$DEBUG_EVERY_STEPS"
)

if [[ -n "$RUN_NAME" ]]; then
  args+=("--run_name" "$RUN_NAME")
fi

# Allow callers to append any extra flags.
args+=("$@")

echo "[Debug Train] uv run lamb ${args[*]}"
uv run lamb "${args[@]}"
