# CLaRa Compression Tutorial

This walkthrough explains how the document compressor turns encoder hidden states into fixed memory tokens, using the provided training script in scripts/train_debug.sh. It covers tensor shapes, masks, and the rationale for each step.

## What the training script runs

The debug script trains the compressor on Qwen/Qwen3-0.6B with 8 memory tokens and sequence lengths capped at 128 for documents and 1024 for the decoder:

```
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
  --report_to tensorboard
```

Key compression knobs:
- compress_rate=8 → num_memory_tokens = 8
- doc_max_length=128 → encoder sequence length (seq_len) is at most 128
- use_compressor_mlp=True by default → per-memory MLP refinement is on

## Inputs to the compressor

During the encoder forward pass (with the encoder LoRA adapter enabled), you obtain:
- encoder_hidden_states: shape [batch, seq_len, hidden_size]; here seq_len ≤ 128
- attention_mask: shape [batch, seq_len]; 1 for real tokens, 0 for padding

Masks let us ignore padding when routing tokens into memory slots.

## Step-by-step: token → memory routing

1) Router logits per token
- Each token vector h_i ∈ ℝ^{hidden_size} is projected by a shared linear layer to num_memory_tokens logits.
- Code path: router (Linear hidden_size → num_memory_tokens) + GELU.
- Output scores shape: [batch, seq_len, num_memory_tokens].

2) Masking padding
- If attention_mask is provided, positions with 0 are set to −∞ before softmax so they contribute zero weight.
- Masked scores shape stays [batch, seq_len, num_memory_tokens].

3) Softmax over sequence length
- For each memory token slot n, we softmax across seq_len to get weights w_{i,n}.
- weights shape: [batch, seq_len, num_memory_tokens]; each column sums to 1 along seq_len.
- In math: $w_{i,n} = \text{softmax}_i(\text{score}_{i,n})$.

4) Weighted sum to form memory tokens
- For each memory slot n, take a weighted sum of token embeddings using weights w_{i,n}.
- Implemented with einsum("bsh,bsn->bnh").
- Result memory_embeddings shape: [batch, num_memory_tokens, hidden_size].
- This preserves token-level diversity because each slot sees a different weight pattern over the sequence (no global mean squeeze).

5) Per-memory refinement (optional)
- If use_mlp=True: each memory vector passes through MLP: Linear(hidden_size→mlp_hidden_dim) → GELU → Linear(mlp_hidden_dim→hidden_size).
- If use_mlp=False: single linear projection per memory token.
- Output shape remains [batch, num_memory_tokens, hidden_size].

6) Normalization
- LayerNorm over hidden_size to stabilize downstream usage.

## Why this design

- No information bottleneck: Instead of collapsing seq_len×hidden_size to a single pooled vector, each memory slot receives a learned distribution over tokens. This keeps multiple latent summaries.
- Slot specialization: Different router weights let slots specialize (e.g., intro, evidence, conclusion) without enforcing hard segmentation.
- Padding safety: Masking before softmax ensures padded positions have zero influence.
- Lightweight: Only one linear router and a small MLP per slot; cost scales with seq_len × num_memory_tokens, not seq_len².

## Shapes at a glance

Let B=batch, S=seq_len (≤128), H=hidden_size, M=num_memory_tokens (8 in debug script):
- encoder_hidden_states: [B, S, H]
- attention_mask: [B, S]
- router scores: [B, S, M]
- weights after softmax: [B, S, M]
- memory_embeddings before MLP: [B, M, H]
- memory_embeddings after MLP/Linear + LayerNorm: [B, M, H]

## Loss signal

The helper compute_mse_loss compares the mean of memory tokens to the masked mean of encoder tokens:
- original_mean = masked mean over S: [B, H]
- compressed_mean = mean over M: [B, H]
- Loss: MSE(compressed_mean, original_mean)

This is a gentle reconstruction objective that encourages memory slots collectively to preserve the overall content distribution.

## Where to look in code

- Document compressor implementation: src/lamb/bridge.py
- Config knobs (compress_rate, use_compressor_mlp, encoder_pool_method): src/lamb/config.py
- Model wiring (where the compressor is constructed): src/lamb/clara_model.py

## Practical tips

- Increase compress_rate if you need more capacity; it scales linearly in cost with seq_len.
- If you observe over-smoothing, keep use_compressor_mlp=True to add per-slot nonlinearity.
- Ensure attention_mask is correctly set; otherwise padding tokens will get non-zero weights and dilute the summaries.
- doc_max_length in the CLI caps encoder sequence length; keep it aligned with your tokenizer truncation.
