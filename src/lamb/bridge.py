import torch
from torch import nn
from torch.nn import functional


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)

    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom


def _masked_max(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.max(dim=1).values

    # Mask out padding with a large negative value (dtype-safe)
    neg = torch.finfo(x.dtype).min
    x_masked = x.masked_fill(mask.unsqueeze(-1) == 0, neg)
    return x_masked.max(dim=1).values


class DocumentCompressor(nn.Module):
    """Compresses documents into fixed-size memory token embeddings.

    Based on Apple's CLaRa architecture:
    - Encodes documents through base model with encoder adapter
    - Mean-pools encoder hidden states
    - Projects to num_memory_tokens embeddings via learnable projection
    """

    def __init__(
        self,
        hidden_size: int,
        num_memory_tokens: int = 32,
        use_mlp: bool = True,
        mlp_hidden_dim: int | None = None,
        encoder_pool_method: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_tokens = num_memory_tokens
        self.use_mlp = use_mlp
        self.encoder_pool_method = encoder_pool_method

        # Pooling modules (only created when needed)
        if self.encoder_pool_method == "mpl":
            self.pool_mpl = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size, bias=False),
            )
        elif self.encoder_pool_method == "nearn":
            # Learnable attention pooling: score each token -> softmax -> weighted sum
            self.pool_score = nn.Linear(hidden_size, 1, bias=False)
        elif self.encoder_pool_method in {"mean", "max"}:
            pass
        else:
            raise ValueError(
                "encoder_pool_method must be one of: 'mean', 'mpl', 'nearn', 'max' "
                f"(got: {self.encoder_pool_method!r})"
            )

        if use_mlp:
            mlp_hidden_dim = mlp_hidden_dim or hidden_size * 4
            self.compress_mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, num_memory_tokens * hidden_size, bias=False),
            )
        else:
            # Simple linear projection
            self.compress_linear = nn.Linear(
                hidden_size, num_memory_tokens * hidden_size, bias=False
            )

        # Layer norm for compressed representations
        self.compress_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        print(
            f"[Compressor] {hidden_size} -> {num_memory_tokens} memory tokens "
            f"({'MLP' if use_mlp else 'Linear'}), pool={self.encoder_pool_method}"
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
        # Pool across sequence length
        # Shape: [batch, hidden_size]
        if self.encoder_pool_method == "mean":
            pooled = _masked_mean(encoder_hidden_states, attention_mask)
        elif self.encoder_pool_method == "max":
            pooled = _masked_max(encoder_hidden_states, attention_mask)
        elif self.encoder_pool_method == "mpl":
            encoded = self.pool_mpl(encoder_hidden_states)
            pooled = _masked_mean(encoded, attention_mask)
        elif self.encoder_pool_method == "nearn":
            scores = self.pool_score(encoder_hidden_states).squeeze(-1)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = torch.softmax(scores.float(), dim=1).to(dtype=encoder_hidden_states.dtype)
            pooled = (encoder_hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected pool method: {self.encoder_pool_method!r}")

        # Project to memory token space
        compressed = self.compress_mlp(pooled) if self.use_mlp else self.compress_linear(pooled)

        # Reshape to memory tokens
        # Shape: [batch, num_memory_tokens, hidden_size]
        batch_size = encoder_hidden_states.size(0)
        memory_embeddings = compressed.view(batch_size, self.num_memory_tokens, self.hidden_size)

        # Normalize
        memory_embeddings = self.compress_norm(memory_embeddings)

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
