#!/usr/bin/env bash
# This script is intended to run on H100 machines for debugging training issues.
set -euo pipefail

git checkout dev
git pull --ff-only

# Use pip + an isolated venv (Unsloth can break if we mix with system site-packages).
# Kaggle images sometimes ship Python with ensurepip disabled, so venv creation can fail.
if ! python -m venv .venv_h100 2>/dev/null; then
  echo "[train_debug] python -m venv failed (ensurepip). Falling back to --without-pip + get-pip.py" >&2
  python -m venv .venv_h100 --without-pip
  if command -v curl >/dev/null 2>&1; then
    curl -sSfL https://bootstrap.pypa.io/get-pip.py | .venv_h100/bin/python -
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://bootstrap.pypa.io/get-pip.py | .venv_h100/bin/python -
  else
    echo "[train_debug] Need curl or wget to bootstrap pip." >&2
    exit 1
  fi
fi

# shellcheck disable=SC1091
source .venv_h100/bin/activate

python -m pip install --upgrade pip setuptools wheel -q

# Install project deps + dev tools (ruff/pytest), then Unsloth.
python -m pip install -e ".[dev]" -q
python -m pip install --upgrade unsloth -q
export LAMB_NAN_GUARD=1
export LAMB_NAN_GUARD_EVERY=1
export LAMB_DISABLE_DYNAMO=1
# Also ask Torch to avoid compile/dynamo globally.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
# Optional: print python + numpy locations to confirm we're not using system site-packages.
# export LAMB_ENV_DIAG=1
# Set to 1 to bisect whether NaNs originate from the encoder path.
# export LAMB_DETACH_MEMORY=1
python -m lamb.cli \
  --stage stage1 \
  --dataset_limit 100000 \
  --dataset_split test \
  --model_name Qwen/Qwen3-0.6B \
  --compress_rate 8 \
  --batch_size 1 \
  --max_steps 1000 \
  --save_steps 999 \
  --learning_rate 0.0001 \
  --mse_weight 0.1 \
  --checkpoint_dir checkpoints/debug_demo \
  --generation_top_k 20 \
  --doc_max_length 128 \
  --debug_mode False \
  --debug_every_steps 1 \
  --debug_num_samples 1 \
  --debug_repeat_dataset 0 \
  --report_to tensorboard \
  --use_clara_original True \