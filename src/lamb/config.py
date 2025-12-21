from dataclasses import dataclass, field

import torch

from lamb.utils import env_flag, pick_attn_implementation, pick_device, pick_dtype


@dataclass
class LaMBConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
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
    max_steps: int | None = None

    debug_every_steps: int = 5
    debug_num_samples: int = 1
    verbose: bool = field(default_factory=lambda: env_flag("LAMB_VERBOSE"))

    tensorboard: bool = field(default_factory=lambda: env_flag("LAMB_TENSORBOARD"))
    tensorboard_logdir: str = "logs/tensorboard"
    tensorboard_every_steps: int = 5
    tensorboard_text_every_steps: int = 50

    dataset_size: int = 20000


def default_config() -> LaMBConfig:
    return LaMBConfig()
