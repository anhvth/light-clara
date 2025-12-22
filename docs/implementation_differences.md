# Critical Differences: Our Implementation vs Original CLaRa

## 1. **Memory Token Format & Separator**

### Original CLaRa (.extra/ml-clara)
- Uses `<SEP>` separator token **between documents**
- Memory tokens string: `<mem_0><mem_1>...<mem_31><SEP><mem_0><mem_1>...<mem_31><SEP>...`
- Implemented in [modeling_clara.py#L977-L978](.extra/ml-clara/openrlhf/models/modeling_clara.py#L977-L978):
  ```python
  mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
  docs = mem_tokens_str * self.generation_top_k
  ```
- Creates separator token: `tokenizer.sep_token = '<SEP>'` [#L483](.extra/ml-clara/openrlhf/models/modeling_clara.py#L483)
- Replacement logic accounts for separator: `slot_len = num_embs + (1 if self.sep else 0)` [#L929](.extra/ml-clara/openrlhf/models/modeling_clara.py#L929)

### Our Implementation
- **No separator between documents** ❌
- Memory tokens string: `<mem_0><mem_1>...<mem_31><mem_0><mem_1>...<mem_31>...` (just concatenated)
- Implemented in [src/lamb/clara_collate.py#L25-L44](src/lamb/clara_collate.py#L25-L44):
  ```python
  def build_memory_token_string(num_docs: int, num_mem_tokens_per_doc: int = 32) -> str:
      tokens = []
      for _doc_idx in range(num_docs):
          for mem_idx in range(num_mem_tokens_per_doc):
              tokens.append(f"<mem_{mem_idx}>")
      return "".join(tokens)  # NO SEPARATOR
  ```
- No `<SEP>` token added to tokenizer
- Replacement logic doesn't account for separators [src/lamb/clara_model.py#L276-L308](src/lamb/clara_model.py#L276-L308)

## 2. **Compression Architecture**

### Original CLaRa
- **Uses the decoder model itself as encoder/compressor**
- No separate compressor module; relies on LoRA adapters (`encoder_adapter`)
- Compression method [modeling_clara.py#L538-L579](.extra/ml-clara/openrlhf/models/modeling_clara.py#L538-L579):
  ```python
  def _compr_decoder(self, input_ids, attention_mask):
      self.decoder.set_adapter('encoder_adapter')  # Switch to encoder mode
      
      # Get hidden states from decoder
      emb = self.decoder(
          input_ids=input_ids,
          attention_mask=attention_mask,
          output_hidden_states=True
      ).hidden_states[-1]
      
      # Extract ONLY memory token positions
      mask = torch.isin(input_ids, self.decoder_tokenizer.mem_token_ids_pt)
      return emb[mask].reshape(emb.size(0), -1, emb.size(-1)), mse_loss
  ```
- **Key insight**: Memory tokens are in the INPUT, decoder processes them, then only their hidden states are extracted
- Input format: `<ENC><BOS>Document text<EOS><MEM><MEM>...<MEM>` [modeling_clara.py#L867-L904](.extra/ml-clara/openrlhf/models/modeling_clara.py#L867-L904)

### Our Implementation  
- **Uses a separate `DocumentCompressor` module** 
- Architecture: `AttentionCompressor` with learnable queries [src/lamb/bridge.py#L15-L53](src/lamb/bridge.py#L15-L53):
  ```python
  class AttentionCompressor(nn.Module):
      def __init__(self, dim: int = 1024, num_heads: int = 8, target_len: int = 8):
          self.summary_queries = nn.Parameter(torch.randn(1, target_len, dim))
          self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
      
      def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
          queries = self.summary_queries.repeat(batch_size, 1, 1)
          compressed, _ = self.attn(query=queries, key=x, value=x, ...)
          return compressed
  ```
- Compresses encoder outputs AFTER full document encoding [src/lamb/clara_model.py#L204-L228](src/lamb/clara_model.py#L204-L228)
- **Different design philosophy**: Cross-attention pooling vs. learned token embeddings

## 3. **MSE Loss Calculation**

### Original CLaRa
- Compares **mean of memory token embeddings** vs **mean of non-memory token embeddings**
- From [modeling_clara.py#L556-L575](.extra/ml-clara/openrlhf/models/modeling_clara.py#L556-L575):
  ```python
  mem_mask = mask & attn
  non_mem_mask = (~mask) & attn
  
  mem_mean = (emb * mem_mask.unsqueeze(-1)).sum(dim=1) / mem_len.unsqueeze(-1)
  non_mem_mean = (emb * non_mem_mask.unsqueeze(-1)).sum(dim=1) / non_mem_len.unsqueeze(-1)
  
  mse_loss = F.mse_loss(non_mem_mean, mem_mean, reduction='mean')
  ```
- Forces memory tokens to preserve document info

### Our Implementation
- Compares **mean-pooled encoder outputs** vs **mean-pooled compressed embeddings**
- From [src/lamb/bridge.py#L134-L150](src/lamb/bridge.py#L134-L150):
  ```python
  def compute_mse_loss(self, encoder_hidden_states, memory_embeddings, attention_mask):
      original_mean = _masked_mean(encoder_hidden_states, attention_mask)
      compressed_mean = memory_embeddings.mean(dim=1)
      return functional.mse_loss(compressed_mean, original_mean)
  ```
- Different loss signal—compares average representations

## 4. **Memory Token Naming Convention**

### Original CLaRa
- Can use **either** unique names `<MEM0>, <MEM1>, ...` **or** repeated `<MEM>` tokens
- Controlled by `different_mem_tokens` flag [modeling_clara.py#L465-L474](.extra/ml-clara/openrlhf/models/modeling_clara.py#L465-L474):
  ```python
  if cfg.different_mem_tokens:
      mem_tokens = [f'<MEM{i}>' for i in range(n_mem_tokens)]  # <MEM0>, <MEM1>, ...
  else:
      tokenizer.mem_tokens = ['<MEM>'] * n_mem_tokens  # All same token
  ```

### Our Implementation
- Always uses **unique indexed tokens**: `<mem_0>, <mem_1>, ..., <mem_31>`
- From [src/lamb/clara_model.py#L110-L112](src/lamb/clara_model.py#L110-L112):
  ```python
  mem_tokens = [f"<mem_{i}>" for i in range(num_mem_tokens)]
  ```
- No option for repeated tokens

## 5. **Encoder Input Format**

### Original CLaRa
- Adds special markers to documents before encoding [modeling_clara.py#L876-L891](.extra/ml-clara/openrlhf/models/modeling_clara.py#L876-L891):
  ```python
  inp_enc = [
      self.decoder_tokenizer.enc_token +      # <ENC>
      self.decoder_tokenizer.bos_token +      # <BOS>
      text +                                   # Document
      self.decoder_tokenizer.eos_token        # <EOS>
      for text in texts
  ]
  # Then append memory tokens
  inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(
      inp_enc['input_ids'], inp_enc['attention_mask'], num_mem_tokens, tokenizer
  )
  ```
- Memory tokens are **part of the encoder input sequence**

### Our Implementation
- Encodes documents directly without special wrappers
- Memory tokens are **NOT in encoder input**—only added to decoder prompts
- Compression happens on plain encoder outputs [src/lamb/clara_model.py#L204-L228](src/lamb/clara_model.py#L204-L228)

## 6. **Configuration Flags**

### Original CLaRa Has (We Don't)
- `different_mem_tokens`: Toggle unique vs repeated token names
- `sep`: Whether to use separator tokens
- `optimize_mem_tokens`: Special gradient masking for memory token embeddings
- `compr_model_name`: Option for separate encoder model (BERT, etc.)
- `kbtc_training`: Knowledge-based task completion mode

### We Have (Original Doesn't)
- `use_clara_original`: Toggle between our format and Apple's format
- `encoder_pool_method`: Choice of compression strategy
- `use_compressor_mlp`: Whether to use MLP in compressor

## Summary: Critical Missing Features

1. ❌ **No `<SEP>` token between documents** → Decoder can't distinguish document boundaries
2. ❌ **Completely different compression architecture** → Attention pooling vs learned token states
3. ❌ **Different MSE loss target** → May not preserve information correctly
4. ❌ **Memory tokens not in encoder input** → Original uses them as part of encoding
5. ❌ **No support for repeated memory tokens** → Always uses unique IDs

## Impact Assessment

**High Priority Fixes:**
1. Add `<SEP>` separator between documents in memory token string
2. Consider implementing decoder-as-encoder compression (original architecture)
3. Fix MSE loss to match original (mem vs non-mem token means)

**Medium Priority:**
- Add `sep` config flag to control separator usage
- Support `different_mem_tokens` option for token naming

**Low Priority:**
- Add `optimize_mem_tokens` gradient hook
- Support separate encoder model option
