# CLaRa Debug Mode

The debug mode is now properly implemented following the pattern from `src/lamb/debug.py`.

## Features

### 1. **Dataset Repetition** (`--debug_repeat_dataset N`)
Repeats the small dataset N times to allow the model to overfit and show learning progress:
```bash
--debug_repeat_dataset 5  # 8 samples → 48 samples (8 * 6)
```

### 2. **Color-Coded Token Output** (`--debug_every_steps N`)
Every N steps, shows the gold answer tokens colored by the model's probability:
- **RED** = Low confidence (P ~ 0.0)
- **GREEN** = High confidence (P ~ 1.0)
- **YELLOW/ORANGE** = Medium confidence (P ~ 0.5)

The color gradient smoothly transitions: Red → Orange → Yellow → Light Green → Green

### 3. **What Gets Printed**

For each debug sample:
```
================================================================================
[Debug] Sample 0
================================================================================
[Debug] Answer token accuracy: 45.23% (19/42)
[Debug] Answer CE loss: 2.345678

[Debug] Answer colored by P(gold token) - red=low confidence, green=high confidence:

<|im_start|>system
Answer the question...
<|im_end|>
<|im_start|>user
Question: Who is the mother...?
<|im_end|>
<|im_start|>assistant
[COLORED TOKENS HERE - each token colored by model's probability for that exact token]
================================================================================
```

## Usage Example

```bash
uv run lamb \
  --stage stage1 \
  --dataset_limit 8 \
  --model_name Qwen/Qwen3-0.6B \
  --compress_rate 16 \
  --batch_size 2 \
  --max_steps 100 \
  --debug_mode True \
  --debug_every_steps 10 \
  --debug_num_samples 2 \
  --debug_repeat_dataset 10
```

## How It Works

1. **Repeats dataset**: If you have 8 samples and `--debug_repeat_dataset 10`, you get 88 total samples
2. **Every 10 steps**: Shows color-coded output for first 2 samples in current batch
3. **Colors show learning**: As training progresses, more tokens turn green (higher confidence)
4. **Overfit indicator**: When all tokens are green, model has memorized that example

## Why This is Useful

- **Visual feedback**: See exactly which tokens the model is confident/struggling with
- **Debug overfitting**: Repeat small dataset to verify model CAN learn
- **Monitor progress**: Watch colors shift from red→green as training progresses
- **Identify issues**: If tokens stay red after many steps, something is wrong

The color coding uses the exact same RGB gradient as the existing `debug.py` to maintain consistency!
