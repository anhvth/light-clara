import torch
from torch import nn
from torch.nn import functional


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)

    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom


def compute_mse_loss_original(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    mem_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss matching original CLaRa: mean(mem) vs mean(non-mem).

    Args:
        hidden_states: [batch, seq_len, hidden_size] - includes memory token positions
        input_ids: [batch, seq_len] - token IDs with memory tokens at end
        mem_token_ids: [num_mem_tokens] - IDs of memory tokens
        attention_mask: [batch, seq_len]

    Returns:
        mse_loss: scalar
    """
    # Cast to float32 for numerical stability with 4-bit models
    hidden_states = hidden_states.float()

    # Create mask for memory token positions
    mem_token_ids_set = set(mem_token_ids.tolist())
    batch_size, seq_len, _hidden_size = hidden_states.shape

    mem_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
    for i in range(batch_size):
        for j in range(seq_len):
            if input_ids[i, j].item() in mem_token_ids_set:
                mem_mask[i, j] = True

    # Combine with attention mask
    attn = attention_mask.bool()
    mem_mask = mem_mask & attn
    non_mem_mask = (~mem_mask) & attn

    # Compute means
    mem_len = mem_mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)
    non_mem_len = non_mem_mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)

    mem_sum = (hidden_states * mem_mask.unsqueeze(-1)).sum(dim=1)
    non_mem_sum = (hidden_states * non_mem_mask.unsqueeze(-1)).sum(dim=1)

    mem_mean = mem_sum / mem_len
    non_mem_mean = non_mem_sum / non_mem_len

    return functional.mse_loss(non_mem_mean, mem_mean, reduction="mean")


class AttentionCompressor(nn.Module):
    def __init__(self, dim: int = 1024, num_heads: int = 8, target_len: int = 8):
        super().__init__()
        # 1. Create 8 learnable "summary tokens" (Queries)
        self.summary_queries = nn.Parameter(torch.randn(1, target_len, dim))

        # 2. Attention layer
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x shape: (Batch, 128, 1024)
        batch_size = x.shape[0]

        # Expand queries to match batch size: (Batch, 8, 1024)
        queries = self.summary_queries.repeat(batch_size, 1, 1)

        key_padding_mask = None
        if attention_mask is not None:
            # MultiheadAttention expects True where tokens should be ignored.
            key_padding_mask = attention_mask == 0

        # 3. Cross-Attention:
        # Query = Summary Tokens (8)
        # Key/Value = Input Tokens (128)
        compressed, _ = self.attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )

        return compressed  # Shape: (Batch, 8, 1024)


class DocumentCompressor(nn.Module):
    """Compresses documents into fixed-size memory token embeddings.

    Based on Apple's CLaRa architecture:
    - Encodes documents through base model with encoder adapter
    - Routes sequence tokens to memory tokens with learned token-softmax weights
    - Optionally refines each memory token with a per-token MLP
    """

    def __init__(
        self,
        hidden_size: int,
        num_memory_tokens: int = 32,
        use_mlp: bool = True,
        mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_tokens = num_memory_tokens
        self.use_mlp = use_mlp

        self.attention_compressor = AttentionCompressor(
            dim=hidden_size, num_heads=8, target_len=num_memory_tokens
        )

        # if use_mlp:
        #     mlp_hidden_dim = mlp_hidden_dim or hidden_size * 4
        #     self.compress_mlp = nn.Sequential(
        #         nn.Linear(hidden_size, mlp_hidden_dim, bias=False),
        #         nn.GELU(),
        #         nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        #     )
        # else:
        #     # Simple linear projection per memory token
        #     self.compress_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        # # Layer norm for compressed representations
        self.compress_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        print(
            f"[Compressor] {hidden_size} -> {num_memory_tokens} memory tokens "
            f"({'MLP' if use_mlp else 'Linear'}), routing=token_softmax"
        )

    def forward(
        self, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compress encoder outputs to memory token embeddings.

        Args:
            encoder_hidden_states: [batch, seq_len, hidden_size] from encoder
            attention_mask: [batch, seq_len] (1=token, 0=pad). Optional.

        Returns:
            memory_embeddings: [batch, num_memory_tokens, hidden_size]
        """
        memory_embeddings = self.attention_compressor(encoder_hidden_states, attention_mask)
        # import ipdb; ipdb.set_trace()
        # memory_embeddings = (
        #     self.compress_mlp(memory_embeddings)
        #     if self.use_mlp
        #     else self.compress_linear(memory_embeddings)
        # )

        # # Normalize
        # memory_embeddings = self.compress_norm(memory_embeddings)

        return memory_embeddings

    def compute_mse_loss(
        self,
        encoder_hidden_states: torch.Tensor,
        memory_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MSE loss between compressed and original representations.

        Encourages compression to preserve information from encoder outputs.

        Args:
            encoder_hidden_states: [batch, seq_len, hidden_size]
            memory_embeddings: [batch, num_memory_tokens, hidden_size]
            attention_mask: [batch, seq_len] (1=token, 0=pad). Optional.

        Returns:
            mse_loss: scalar tensor
        """
        # Mean-pool both to same shape for comparison
        original_mean = _masked_mean(encoder_hidden_states, attention_mask)  # [batch, hidden_size]
        compressed_mean = memory_embeddings.mean(dim=1)  # [batch, hidden_size]

        return functional.mse_loss(compressed_mean, original_mean)
