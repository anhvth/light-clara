import torch
from torch import nn
from torch.nn import functional


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)

    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom


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

        # Token-wise router: produce per-memory token weights across the sequence.
        self.router = nn.Linear(hidden_size, num_memory_tokens, bias=False)
        self.router_activation = nn.GELU()

        if use_mlp:
            mlp_hidden_dim = mlp_hidden_dim or hidden_size * 4
            self.compress_mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
            )
        else:
            # Simple linear projection per memory token
            self.compress_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        # Layer norm for compressed representations
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
        # Token-softmax router: map sequence length -> num_memory_tokens without collapsing first
        # Shape: [batch, seq_len, num_memory_tokens]
        scores = self.router_activation(self.router(encoder_hidden_states))
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        weights = torch.softmax(scores.float(), dim=1).to(dtype=encoder_hidden_states.dtype)

        # Weighted sum for each memory token: [batch, num_memory_tokens, hidden_size]
        memory_embeddings = torch.einsum("bsh,bsn->bnh", encoder_hidden_states, weights)

        # Optional per-memory MLP
        memory_embeddings = (
            self.compress_mlp(memory_embeddings)
            if self.use_mlp
            else self.compress_linear(memory_embeddings)
        )

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
