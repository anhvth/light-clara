import torch
from torch import nn
from torch.nn import functional


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)

    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom


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


# Usage
# model = AttentionCompressor()
# input_tensor = torch.randn(32, 128, 1024)  # Batch of 32
# output = model(input_tensor)

# print(output.shape)  # torch.Size([32, 8, 1024])


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
        encoder_pool_method: str = "token_softmax",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_tokens = num_memory_tokens
        self.use_mlp = use_mlp
        self.encoder_pool_method = encoder_pool_method

        if encoder_pool_method != "token_softmax":
            print(
                f"[Compressor] encoder_pool_method={encoder_pool_method!r} is deprecated; "
                "using token_softmax routing"
            )

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
