#!/bin/bash
# Test CLaRa debug mode with color-coded generation output

echo "Testing CLaRa Debug Mode..."
echo "============================"
echo ""

uv run lamb \
  --stage stage1 \
  --dataset_limit 4 \
  --dataset_split test \
  --model_name Qwen/Qwen3-0.6B \
  --compress_rate 8 \
  --batch_size 1 \
  --max_steps 20 \
  --save_steps 999 \
  --learning_rate 0.0001 \
  --mse_weight 0.1 \
  --checkpoint_dir checkpoints/debug_demo \
  --generation_top_k 1 \
  --doc_max_length 128 \
  --debug_mode True \
  --debug_every_steps 5 \
  --debug_num_samples 1 \
  --debug_repeat_dataset 5
