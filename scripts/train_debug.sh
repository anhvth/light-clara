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
  --debug_repeat_dataset 100 \
  --report_to tensorboard \
  --use_clara_original True
  
  # --chat_template Qwen/Qwen3-4B-Instruct-2507 \