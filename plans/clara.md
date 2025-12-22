# Plan: Correct CLaRa Implementation

## Current State Analysis

The current code implements a **LaMB-style vertical memory bridge with KL distillation**, which differs significantly from Apple's CLaRa architecture. 

**Key Differences:**
- Current: Uses vertical bridge to create past key-value caches from compressed context
- CLaRa: Uses memory tokens as learnable embeddings that replace document text in decoder input
- Current: KL distillation between teacher (full context) and student (compressed)
- CLaRa: Direct QA loss + paraphrase loss + optional MSE loss on compressed representations

## Implementation Steps

### 1. Refactor Model Architecture to Match CLaRa's Memory Token Design

**File:** `src/lamb/bridge.py`, `src/lamb/model.py`

**Changes Needed:**

- Remove vertical bridge KV-cache approach from `bridge.py`
- Add memory token embeddings (`<mem_0>`...`<mem_31>`) to tokenizer vocabulary in `model.py`
- Implement compression pipeline:
  ```
  Documents → Encoder (with LoRA) → Mean-pool hidden states → 
  Learnable projection → Memory embeddings [batch, num_docs, num_mem_tokens, hidden_dim]
  ```
- Memory embeddings replace `<mem>` placeholders in decoder input (NOT passed as KV cache)
- Remove `past_key_values` usage; decoder sees memory embeddings as regular token embeddings

**Key Implementation Details:**

```python
class MemoryTokenizer:
    def __init__(self, base_tokenizer, num_memory_tokens=32):
        # Add <mem_0> through <mem_31> to vocabulary
        # Add optional <sep> separator token
        
class DocumentCompressor(nn.Module):
    def __init__(self, hidden_size, num_memory_tokens=32):
        # Learnable projection: hidden_size -> num_memory_tokens x hidden_size
        # Can be implemented as MLP or simple linear projection
        
    def forward(self, encoder_hidden_states):
        # Input: [batch, seq_len, hidden_size]
        # Output: [batch, num_memory_tokens, hidden_size]
        # Mean-pool encoder outputs, then project to memory embeddings
```

**Memory Token Handling:**
- Initialization: Add new tokens to tokenizer, initialize embeddings randomly (std=0.02)
- Optimization: Memory token embeddings in `embed_tokens` layer have `requires_grad=True`
- Usage: Replace `<mem_0>...<mem_31>` placeholders in prompt with actual compressed embeddings

---

### 2. Implement CLaRa Stage 1 Training (Compression Pretraining)

**File:** `src/lamb/train.py`

**Changes Needed:**

- Add stage-based training logic with `--stage` flag (stage1, stage1_2, stage2)
- Implement three loss components:

**A. QA Loss (Primary):**
```python
def compute_qa_loss(model, docs, question, answer):
    # 1. Compress documents
    doc_embeddings = model.compress_documents(docs)  # [batch, num_docs, 32, hidden_dim]
    
    # 2. Build prompt with memory tokens
    prompt = build_prompt_with_memory_tokens(
        question=question,
        num_docs=len(docs),
        num_mem_tokens=32
    )
    # Example: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<background>\n<mem_0>...<mem_31>\n<mem_32>...<mem_63>\n</background>\n\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    
    # 3. Tokenize and replace memory token placeholders
    input_ids, attention_mask = tokenize_with_memory_replacement(prompt, doc_embeddings)
    
    # 4. Forward pass through decoder
    outputs = model.decoder(input_ids=input_ids, attention_mask=attention_mask)
    
    # 5. Compute CE loss only on answer tokens
    labels = mask_prompt_tokens(input_ids, prompt_length)
    loss = F.cross_entropy(outputs.logits[..., :-1, :], labels[..., 1:])
    return loss
```

**B. Paraphrase Loss:**
```python
def compute_paraphrase_loss(model, doc, paraphrase_text):
    # 1. Compress document
    doc_embedding = model.compress_documents([doc])  # [1, 1, 32, hidden_dim]
    
    # 2. Build prompt with random paraphrase instruction
    instruction = random.choice(PARAPHRASE_INSTRUCTIONS).format(docs="<mem_0>...<mem_31>")
    prompt = f"{instruction}\n{paraphrase_text}"
    
    # 3. Decode with memory embeddings
    # (same tokenize + replace + forward + loss as QA)
    return loss
```

**C. MSE Loss (Optional):**
```python
def compute_mse_loss(encoder_hidden_states, compressed_embeddings):
    # Compare mean-pooled original encoder output with compressed representation
    original_mean = encoder_hidden_states.mean(dim=1)  # [batch, hidden_dim]
    compressed_mean = compressed_embeddings.mean(dim=1)  # [batch, hidden_dim]
    return F.mse_loss(compressed_mean, original_mean)
```

**Combined Training Loss:**
```python
total_loss = qa_loss + paraphrase_weight * paraphrase_loss + mse_weight * mse_loss
```

**LoRA Configuration:**
- Encoder adapter: Applied to encoder copy of base model (or same model with adapter switching)
- Decoder adapter: Applied to decoder (generation)
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Rank: 16-32

---

### 3. Implement Data Collation for Multi-Stage Training

**File:** `src/lamb/clara_collate.py` (new file)

**Stage 1 Data Processing:**

```python
def stage1_collate_fn(batch, model, tokenizer, doc_max_length=256, dec_max_length=1024):
    """
    Batch format:
    [
        {
            "data_type": "qa",
            "question": "Q?",
            "docs": ["Doc1", "Doc2", ...],
            "answer": "A"
        },
        {
            "data_type": "paraphrase", 
            "docs": ["Original doc"],
            "answer": "Paraphrased version"
        }
    ]
    """
    
    # 1. Encode all documents (flatten across batch)
    all_docs = []
    for item in batch:
        all_docs.extend(item["docs"][:generation_top_k])  # Limit to top-k docs
    
    doc_encodings = tokenizer(
        all_docs,
        max_length=doc_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 2. Build decoder prompts
    decoder_prompts = []
    prompt_lengths = []  # For label masking
    
    for item in batch:
        if item["data_type"] == "qa":
            prompt = build_qa_prompt_with_memory_placeholders(
                question=item["question"],
                answer=item["answer"],
                num_docs=min(len(item["docs"]), generation_top_k),
                num_mem_tokens=32
            )
        else:  # paraphrase
            prompt = build_paraphrase_prompt_with_memory_placeholders(
                answer=item["answer"],
                num_mem_tokens=32
            )
        
        decoder_prompts.append(prompt)
        prompt_lengths.append(compute_prompt_length(prompt, item["answer"]))
    
    # 3. Tokenize decoder inputs
    dec_encodings = tokenizer(
        decoder_prompts,
        max_length=dec_max_length,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )
    
    # 4. Create labels (mask prompt tokens)
    labels = dec_encodings["input_ids"].clone()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100
    
    return {
        "doc_input_ids": doc_encodings["input_ids"],
        "doc_attention_mask": doc_encodings["attention_mask"],
        "dec_input_ids": dec_encodings["input_ids"],
        "dec_attention_mask": dec_encodings["attention_mask"],
        "labels": labels,
        "data_types": [item["data_type"] for item in batch]
    }
```

**Key Functions Needed:**
- `build_qa_prompt_with_memory_placeholders()`: Create chat-formatted prompt with `<mem_X>` tokens
- `build_paraphrase_prompt_with_memory_placeholders()`: Create paraphrase instruction with memory tokens
- `compute_prompt_length()`: Count tokens before answer section for label masking
- `replace_memory_tokens_with_embeddings()`: Replace token IDs with actual compressed embeddings in model forward pass

---

### 4. Add Stage 2 Retrieval Training (Optional for Initial Implementation)

**File:** `src/lamb/retrieval.py` (new file)

**Components:**

**A. Query Reasoner:**
```python
class QueryReasoner(nn.Module):
    def __init__(self, decoder_model, hidden_size):
        self.decoder = decoder_model
        # Uses query_reasoner_adapter LoRA
        
    def encode_query(self, query_text):
        # Tokenize query
        # Forward through decoder with query_reasoner_adapter
        # Extract last hidden state as query embedding
        return query_embedding  # [batch, hidden_size]
```

**B. Differentiable Top-K:**
```python
def differentiable_topk(query_embeddings, doc_embeddings, k, temperature=0.02):
    """
    Args:
        query_embeddings: [batch, hidden_size]
        doc_embeddings: [batch, num_docs, hidden_size]
        k: number of documents to select
        temperature: softmax temperature for soft selection
    
    Returns:
        soft_weights: [batch, k, num_docs] - differentiable selection weights
        hard_indices: [batch, k] - actual top-k document indices
    """
    # 1. Normalize embeddings
    query_norm = F.normalize(query_embeddings, dim=-1)
    doc_norm = F.normalize(doc_embeddings, dim=-1)
    
    # 2. Compute cosine similarity
    scores = torch.bmm(query_norm.unsqueeze(1), doc_norm.transpose(1, 2)).squeeze(1)
    # scores: [batch, num_docs]
    
    # 3. Soft selection via temperature-scaled softmax
    soft_probs = F.softmax(scores / temperature, dim=-1)
    
    # 4. Hard top-k for indexing
    topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
    
    # 5. Create differentiable selection matrix
    # Uses straight-through estimator or Gumbel-softmax
    selection_weights = create_differentiable_selection(soft_probs, topk_indices, k)
    
    return selection_weights, topk_indices
```

**C. Stage 2 Training Loop:**
```python
def stage2_training_step(model, batch):
    # 1. Encode query
    query_embedding = model.query_reasoner.encode_query(batch["question"])
    
    # 2. Compress all documents
    all_doc_embeddings = model.compress_documents(batch["docs"])
    
    # 3. Differentiable top-k selection
    selection_weights, selected_indices = differentiable_topk(
        query_embedding, 
        all_doc_embeddings, 
        k=5
    )
    
    # 4. Weight selected documents (soft selection)
    selected_embeddings = torch.bmm(
        selection_weights, 
        all_doc_embeddings
    )  # [batch, k, hidden_size]
    
    # 5. Generate answer with selected compressed documents
    # (same as Stage 1 but with selected docs)
    
    # 6. Compute loss
    # Option A (contrastive): Use pos_index for contrastive loss
    # Option B (end-to-end): QA loss with gradient flow through selection
    
    return loss
```

---

### 5. Update Configuration and CLI for CLaRa Training

**File:** `src/lamb/config.py`

**New Configuration:**
```python
@dataclass
class ClaraConfig:
    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    
    # Training stage
    stage: str = "stage1"  # stage1, stage1_2, stage2
    
    # Compression
    compress_rate: int = 32  # Number of memory tokens per document
    doc_max_length: int = 256  # Max tokens per document
    num_encoder_layers: int = 5  # Compressor transformer layers
    
    # LoRA
    encoder_lora_rank: int = 16
    decoder_lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Loss weights
    use_mse_loss: bool = True
    mse_weight: float = 0.1
    use_paraphrase_loss: bool = True
    paraphrase_weight: float = 1.0
    qa_weight: float = 1.0
    
    # Data
    generation_top_k: int = 1  # Number of docs to compress per example
    max_seq_len: int = 1024
    
    # Training
    batch_size: int = 2
    learning_rate: float = 1e-4
    max_steps: int = 10000
    gradient_accumulation_steps: int = 4
    
    # Device
    device: str = field(default_factory=pick_device)
    dtype: torch.dtype = field(default_factory=lambda: pick_dtype(pick_device()))
    attn_implementation: str = field(default_factory=lambda: pick_attn_implementation(pick_device()))
    
    # Checkpointing
    save_steps: int = 500
    checkpoint_dir: str = "checkpoints"
    load_from_checkpoint: str | None = None
    
    # Evaluation
    eval_steps: int = 100
    do_eval: bool = False
```

**File:** `src/lamb/cli.py`

**CLI Updates:**
```python
def main():
    parser = argparse.ArgumentParser()
    
    # Stage selection
    parser.add_argument("--stage", type=str, default="stage1", 
                       choices=["stage1", "stage1_2", "stage2"])
    
    # Compression
    parser.add_argument("--compress_rate", type=int, default=32)
    parser.add_argument("--doc_max_length", type=int, default=256)
    
    # LoRA
    parser.add_argument("--encoder_lora_rank", type=int, default=16)
    parser.add_argument("--decoder_lora_rank", type=int, default=16)
    
    # Loss
    parser.add_argument("--use_mse_loss", action="store_true")
    parser.add_argument("--mse_weight", type=float, default=0.1)
    parser.add_argument("--use_paraphrase_loss", action="store_true")
    
    # Checkpointing
    parser.add_argument("--load_from_checkpoint", type=str, default=None,
                       help="Continue from Stage 1 checkpoint when starting Stage 2")
    
    # ... rest of args
```

---

### 6. Validate Implementation with Debug Subset

**Validation Steps:**

1. **Prepare small dataset:**
   ```bash
   uv run lamb \
     --dataset_limit 32 \
     --dataset_split test \
     --dataset_export_jsonl logs/clara_debug_qa.jsonl \
     --stage stage1 \
     --max_steps 0  # Just export data
   ```

2. **Train Stage 1 for 100 optimizer steps:**
   ```bash
   uv run lamb \
     --stage stage1 \
     --dataset_limit 32 \
     --dataset_split test \
     --model_name Qwen/Qwen3-0.6B \
     --compress_rate 32 \
     --doc_max_length 256 \
     --encoder_lora_rank 16 \
     --decoder_lora_rank 16 \
     --use_mse_loss \
     --use_paraphrase_loss \
     --batch_size 2 \
     --learning_rate 1e-4 \
    --max_steps 100 \
    --save_steps 50 \
     --checkpoint_dir checkpoints/stage1_debug
   ```

3. **Monitor metrics:**
   - QA loss (should decrease)
   - Paraphrase loss (should decrease)
   - MSE loss (may stabilize or slightly increase)
   - Answer token accuracy (should improve)
   - Memory token embedding norms (should be non-zero and changing)

4. **Gradient flow check:**
   ```python
   # After backward pass
   mem_token_grads = model.base_model.get_input_embeddings().weight.grad[vocab_size:]
   print(f"Memory token gradient norm: {mem_token_grads.norm().item()}")
   # Should be > 0 and updating each step
   ```

5. **Compare generated answers:**
   ```bash
   # Before compression training (baseline)
   uv run lamb-generate \
     --model_path Qwen/Qwen3-0.6B \
     --question "Who is the mother of...?" \
     --docs "Doc1" "Doc2" \
     --no_compression
   
   # After compression training
   uv run lamb-generate \
     --model_path checkpoints/stage1_debug/step_100 \
     --question "Who is the mother of...?" \
     --docs "Doc1" "Doc2" \
     --use_compression
   ```

6. **Expected behavior:**
   - Loss should decrease steadily (not flatline or explode)
   - Memory token embeddings should diverge from initialization
   - Generated answers should become more relevant to compressed docs
   - MSE loss indicates compression preserves information

---

## Architecture Decisions & Recommendations

### 1. Encoder Choice: Reuse Base Model vs Separate Encoder?

**Apple's Approach:** Reuses the same LLM for both encoding (compression) and decoding (generation), with different LoRA adapters.

**Pros:**
- Memory efficient (one model instead of two)
- Compatible with Apple's pretrained checkpoints
- Simpler architecture
- Shared representations between encoding and decoding

**Cons:**
- Requires careful adapter switching during forward pass
- Slightly more complex training code

**Recommendation:** Start with same-model approach (encoder_adapter + decoder_adapter). This matches Apple's implementation and allows easier checkpoint compatibility.

**Implementation:**
```python
# During compression
model.base_model.set_adapter("encoder_adapter")
encoder_outputs = model.base_model(doc_input_ids, output_hidden_states=True)

# During generation
model.base_model.set_adapter("decoder_adapter")
decoder_outputs = model.base_model(dec_input_ids, inputs_embeds=with_memory_tokens)
```

---

### 2. Memory Token Initialization Strategy

**Options:**

A. **Random initialization** (std=0.02)
   - Simple, standard approach
   - Works well with proper learning rate

B. **Copy from existing token embeddings**
   - Average of vocabulary embeddings
   - May provide better starting point

C. **Learnable but initialized from document summaries**
   - Compress real documents, use outputs as initialization
   - Most sophisticated but complex

**Recommendation:** Start with **random initialization (std=0.02)** as in Apple's code. This is proven to work and keeps implementation simple. Can experiment with smarter initialization later if convergence is slow.

```python
num_mem_tokens = 32
vocab_size = len(tokenizer)
embed_dim = model.config.hidden_size

# Extend embedding layer
new_embeddings = nn.Embedding(vocab_size + num_mem_tokens, embed_dim)
new_embeddings.weight.data[:vocab_size] = model.get_input_embeddings().weight.data
new_embeddings.weight.data[vocab_size:] = torch.randn(num_mem_tokens, embed_dim) * 0.02

model.set_input_embeddings(new_embeddings)
```

---

### 3. Skip Stage 2 Retrieval for Fast Prototype?

**Stage 2 Complexity:**
- Query reasoner with separate LoRA adapter
- Differentiable top-k implementation
- Contrastive loss with positive document indices
- Gradient flow through soft selection

**Recommendation:** **Implement Stage 1 first and validate compression quality.** Many applications can use:
- Fixed retrieval (BM25, dense retriever) to select documents
- Stage 1 compression to compress selected documents
- Decoder generation on compressed documents

Only add Stage 2 if you need:
- Joint optimization of retrieval and generation
- Learning which documents are most useful for answering queries
- End-to-end trainable RAG system

**Fast Path:**
1. Implement Stage 1 (compression pretraining) - **Priority**
2. Validate on debug subset
3. Train on full dataset
4. Deploy with fixed retrieval + compression
5. Add Stage 2 later if needed

---

## Implementation Priority

### Phase 1 (Core CLaRa Stage 1) - **Implement First**

1. ✅ Memory token vocabulary extension
2. ✅ Document compression module
3. ✅ Stage 1 training loop (QA + paraphrase + MSE loss)
4. ✅ Data collation for Stage 1
5. ✅ LoRA adapter management (encoder + decoder)
6. ✅ Checkpoint saving/loading

### Phase 2 (Validation & Optimization)

7. ⬜ Debug subset training validation
8. ⬜ Gradient flow verification
9. ⬜ Loss curve analysis
10. ⬜ Generation quality comparison

### Phase 3 (Stage 2 Retrieval) - **Optional, Add Later**

11. ⬜ Query reasoner implementation
12. ⬜ Differentiable top-k
13. ⬜ Contrastive loss
14. ⬜ End-to-end training

---

## Key Differences from Current LaMB Implementation

| Aspect | Current LaMB | CLaRa (Target) |
|--------|-------------|----------------|
| Memory representation | Past key-value cache | Learnable token embeddings |
| Training objective | KL distillation (teacher-student) | QA loss + paraphrase + MSE |
| Architecture | Vertical bridge projectors | Document compressor + memory tokens |
| Adapter strategy | Single compressor adapter | Multiple adapters (encoder/decoder/query) |
| Compression target | Intermediate layer states | Final hidden state mean-pooled |
| Integration | `past_key_values` in decoder | Replace token placeholders with embeddings |
| Trainable params | Bridge weights + LoRA | Compressor + memory embeddings + LoRA |

---

## Success Criteria

### Stage 1 Completion:
- [ ] Memory tokens (`<mem_0>`...`<mem_31>`) added to tokenizer
- [ ] Documents compress to 32 fixed-size embeddings
- [ ] QA loss decreases during training
- [ ] Memory token embeddings update (non-zero gradients)
- [ ] Generated answers reference compressed document content
- [ ] Can save/load checkpoints with LoRA adapters + memory embeddings

### Stage 1 Quality Metrics:
- [ ] Answer token accuracy > 30% after 100 steps on debug set
- [ ] QA loss < 2.0 after convergence
- [ ] MSE loss stable (not exploding)
- [ ] Generated text is coherent and relevant to question

### Ready for Production:
- [ ] Trains stably on full `apple/CLaRa_multi_stage` dataset
- [ ] Checkpoint compatibility with different base models
- [ ] Memory efficient (fits on consumer GPU)
- [ ] Inference pipeline for new questions + documents
