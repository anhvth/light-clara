import torch
from torch import nn
from torch.nn import functional as F


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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_tokens = num_memory_tokens
        self.use_mlp = use_mlp

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
            f"({'MLP' if use_mlp else 'Linear'})"
        )

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress encoder outputs to memory token embeddings.

        Args:
            encoder_hidden_states: [batch, seq_len, hidden_size] from encoder

        Returns:
            memory_embeddings: [batch, num_memory_tokens, hidden_size]
        """
        # Mean-pool across sequence length
        # Shape: [batch, hidden_size]
        pooled = encoder_hidden_states.mean(dim=1)

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
        self, encoder_hidden_states: torch.Tensor, memory_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss between compressed and original representations.

        Encourages compression to preserve information from encoder outputs.

        Args:
            encoder_hidden_states: [batch, seq_len, hidden_size]
            memory_embeddings: [batch, num_memory_tokens, hidden_size]

        Returns:
            mse_loss: scalar tensor
        """
        # Mean-pool both to same shape for comparison
        original_mean = encoder_hidden_states.mean(dim=1)  # [batch, hidden_size]
        compressed_mean = memory_embeddings.mean(dim=1)  # [batch, hidden_size]

        return F.mse_loss(compressed_mean, original_mean)
