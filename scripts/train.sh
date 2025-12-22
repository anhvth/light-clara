#!/bin/bash
# Real training script for CLaRa

uv run lamb \
  --stage stage1 \
  --dataset_name "apple/CLaRa_multi_stage" \
  --dataset_split train \
  --dataset_limit 1000000 \
  --model_name Qwen/Qwen3-0.6B \
  --compress_rate 8 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_steps 10000 \
  --save_steps 500 \
  --learning_rate 1e-4 \
  --generator_learning_rate 1e-5 \
  --mse_weight 0.1 \
  --checkpoint_dir checkpoints/clara_stage1 \
  --generation_top_k 1 \
  --doc_max_length 256 \
  --debug_mode False
