#!/usr/bin/env bash
set -euo pipefail

# All training flags live here; editing these values controls the run.
STAGE="${STAGE:-stage1}"
DATASET_NAME="${DATASET_NAME:-apple/CLaRa_multi_stage}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_LIMIT="${DATASET_LIMIT:-1000000}"
DATASET_STREAMING="${DATASET_STREAMING:-true}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
COMPRESS_RATE="${COMPRESS_RATE:-8}"
DOC_MAX_LENGTH="${DOC_MAX_LENGTH:-256}"
GENERATION_TOP_K="${GENERATION_TOP_K:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GENERATOR_LEARNING_RATE="${GENERATOR_LEARNING_RATE:-1e-5}"
MSE_WEIGHT="${MSE_WEIGHT:-0.1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/clara_stage1}"
DEBUG_MODE="${DEBUG_MODE:-False}"
DEBUG_EVERY_STEPS="${DEBUG_EVERY_STEPS:-1}"
DEBUG_NUM_SAMPLES="${DEBUG_NUM_SAMPLES:-3}"
DEBUG_REPEAT_DATASET="${DEBUG_REPEAT_DATASET:-0}"
TOKENIZER_TEMPLATE="${TOKENIZER_TEMPLATE:-}"
BF16="${BF16:-False}"
FP16="${FP16:-False}"
TENSORBOARD="${TENSORBOARD:-False}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-logs/tensorboard}"

export STAGE DATASET_NAME DATASET_SPLIT DATASET_LIMIT DATASET_STREAMING MODEL_NAME COMPRESS_RATE
export DOC_MAX_LENGTH GENERATION_TOP_K BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_STEPS SAVE_STEPS
export LEARNING_RATE GENERATOR_LEARNING_RATE MSE_WEIGHT CHECKPOINT_DIR DEBUG_MODE DEBUG_EVERY_STEPS
export DEBUG_NUM_SAMPLES DEBUG_REPEAT_DATASET TOKENIZER_TEMPLATE BF16 FP16 TENSORBOARD TENSORBOARD_LOGDIR

python - <<'PY'
from __future__ import annotations
import os
from tabulate import tabulate

keys = [
    ("stage", "STAGE"),
    ("dataset_name", "DATASET_NAME"),
    ("dataset_split", "DATASET_SPLIT"),
    ("dataset_limit", "DATASET_LIMIT"),
    ("dataset_streaming", "DATASET_STREAMING"),
    ("model_name", "MODEL_NAME"),
    ("compress_rate", "COMPRESS_RATE"),
    ("doc_max_length", "DOC_MAX_LENGTH"),
    ("generation_top_k", "GENERATION_TOP_K"),
    ("batch_size", "BATCH_SIZE"),
    ("gradient_accumulation_steps", "GRADIENT_ACCUMULATION_STEPS"),
    ("max_steps", "MAX_STEPS"),
    ("save_steps", "SAVE_STEPS"),
    ("learning_rate", "LEARNING_RATE"),
    ("generator_learning_rate", "GENERATOR_LEARNING_RATE"),
    ("mse_weight", "MSE_WEIGHT"),
    ("checkpoint_dir", "CHECKPOINT_DIR"),
    ("debug_mode", "DEBUG_MODE"),
    ("debug_every_steps", "DEBUG_EVERY_STEPS"),
    ("debug_num_samples", "DEBUG_NUM_SAMPLES"),
    ("debug_repeat_dataset", "DEBUG_REPEAT_DATASET"),
    ("tokenizer_template", "TOKENIZER_TEMPLATE"),
    ("bf16", "BF16"),
    ("fp16", "FP16"),
    ("tensorboard", "TENSORBOARD"),
    ("tensorboard_logdir", "TENSORBOARD_LOGDIR"),
]
rows = [(friendly, os.environ.get(env, "")) for friendly, env in keys]
print("\n[Train Script Config]")
print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="github"))
PY

args=(
  "--stage" "$STAGE"
  "--dataset_name" "$DATASET_NAME"
  "--dataset_split" "$DATASET_SPLIT"
  "--dataset_limit" "$DATASET_LIMIT"
  "--dataset_streaming" "$DATASET_STREAMING"
  "--model_name" "$MODEL_NAME"
  "--compress_rate" "$COMPRESS_RATE"
  "--doc_max_length" "$DOC_MAX_LENGTH"
  "--generation_top_k" "$GENERATION_TOP_K"
  "--batch_size" "$BATCH_SIZE"
  "--gradient_accumulation_steps" "$GRADIENT_ACCUMULATION_STEPS"
  "--max_steps" "$MAX_STEPS"
  "--save_steps" "$SAVE_STEPS"
  "--learning_rate" "$LEARNING_RATE"
  "--generator_learning_rate" "$GENERATOR_LEARNING_RATE"
  "--mse_weight" "$MSE_WEIGHT"
  "--checkpoint_dir" "$CHECKPOINT_DIR"
  "--debug_mode" "$DEBUG_MODE"
  "--debug_every_steps" "$DEBUG_EVERY_STEPS"
  "--debug_num_samples" "$DEBUG_NUM_SAMPLES"
  "--debug_repeat_dataset" "$DEBUG_REPEAT_DATASET"
  "--tokenizer_template" "$TOKENIZER_TEMPLATE"
  "--bf16" "$BF16"
  "--fp16" "$FP16"
  "--tensorboard" "$TENSORBOARD"
  "--tensorboard_logdir" "$TENSORBOARD_LOGDIR"
)

args+=("$@")

echo "[Train] Running uv run lamb ${args[*]}"
uv run lamb "${args[@]}"
