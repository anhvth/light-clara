from dataclasses import dataclass, field
from typing import Optional

import torch

from lamb.utils import env_flag, pick_attn_implementation, pick_device, pick_dtype


@dataclass
class LaMBConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    num_memory_tokens: int = 32

    compressor_lora_rank: int = 16
    compressor_lora_alpha: int = 32

    batch_size: int = 1
    learning_rate: float = 5e-5
    kl_temperature: float = 1.0
    ce_alpha: float = 0.5
    device: str = field(default_factory=pick_device)
    dtype: torch.dtype = field(default_factory=lambda: pick_dtype(pick_device()))
    attn_implementation: str = field(
        default_factory=lambda: pick_attn_implementation(pick_device())
    )
    max_seq_len: int = 2048
    max_steps: Optional[int] = None

    debug_every_steps: int = 5
    debug_num_samples: int = 1
    verbose: bool = field(default_factory=lambda: env_flag("LAMB_VERBOSE"))


def default_config() -> LaMBConfig:
    return LaMBConfig()
