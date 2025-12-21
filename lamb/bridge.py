import inspect
from typing import Any, List, Optional

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from lamb.utils import apply_rotary_pos_emb


class VerticalLatentMemoryBridge(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model_config = config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        calculated_head_dim = config.hidden_size // config.num_attention_heads

        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        elif calculated_head_dim == 80:
            print("[Bridge] Detected head_dim mismatch (80). Forcing head_dim=128 to match RoPE.")
            self.head_dim = 128
        else:
            self.head_dim = calculated_head_dim

        self.layer_kv_dim = 2 * self.num_kv_heads * self.head_dim
        print(
            f"[Bridge] Vertical: {self.num_layers} layers x ({self.hidden_size} -> DIRECT -> {self.layer_kv_dim} [HeadDim: {self.head_dim}])"
        )

        self.projectors = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.layer_kv_dim, bias=False)
                for _ in range(self.num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.layer_kv_dim) for _ in range(self.num_layers)]
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        all_layer_hidden_states: List[torch.Tensor],
        rotary_module: Optional[Any] = None,
        position_offset: int = 0,
    ) -> DynamicCache:
        target_dtype = all_layer_hidden_states[0].dtype
        batch_size, seq_len, _ = all_layer_hidden_states[0].shape
        cache = DynamicCache()

        cos, sin = None, None
        if rotary_module is not None:
            dummy_device = all_layer_hidden_states[0].device
            position_ids = torch.arange(
                position_offset,
                position_offset + seq_len,
                device=dummy_device,
                dtype=torch.long,
            ).unsqueeze(0)
            dummy_q = torch.zeros(
                1, 1, seq_len, self.head_dim, device=dummy_device, dtype=target_dtype
            )

            sig = inspect.signature(rotary_module.forward)
            if "seq_len" in sig.parameters:
                cos, sin = rotary_module(dummy_q, seq_len=position_offset + seq_len)
            else:
                cos, sin = rotary_module(dummy_q, position_ids)

            start, end = position_offset, position_offset + seq_len
            if cos.ndim == 4:
                cos, sin = cos[:, :, start:end, :], sin[:, :, start:end, :]
            elif cos.ndim == 3:
                cos, sin = cos[:, start:end, :], sin[:, start:end, :]
            elif cos.ndim == 2:
                cos, sin = cos[start:end, :], sin[start:end, :]

            while cos.ndim < 4:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)

            cos = cos.to(target_dtype)
            sin = sin.to(target_dtype)

        for i in range(self.num_layers):
            layer_state = all_layer_hidden_states[i]
            reconstructed_kv = self.projectors[i](layer_state)
            reconstructed_kv = self.norms[i](reconstructed_kv)

            reconstructed_kv = reconstructed_kv.view(
                batch_size,
                seq_len,
                2,
                self.num_kv_heads,
                self.head_dim,
            )

            k = reconstructed_kv[:, :, 0].permute(0, 2, 1, 3)
            v = reconstructed_kv[:, :, 1].permute(0, 2, 1, 3)

            if cos is not None:
                assert sin is not None  # cos and sin should come together
                _, k = apply_rotary_pos_emb(None, k, cos, sin)

            k = k.to(target_dtype)
            v = v.to(target_dtype)
            cache.update(k, v, layer_idx=i)

        return cache
