export LAMB_DETECT_ANOMALY=1
export LAMB_NAN_GUARD=1
export LAMB_NAN_GUARD_EVERY=1
export LAMB_DISABLE_DYNAMO=1
# Set to 1 to bisect whether NaNs originate from the encoder path.
# export LAMB_DETACH_MEMORY=1
# uv run lamb \
python -m src.lamb.cli\
  --stage stage1 \
  --dataset_limit 100000 \
  --dataset_split test \
  --model_name unsloth/Qwen3-14B-bnb-4bit \
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
  --use_clara_original True \
  --qlora True