import os
from dataclasses import dataclass, field

import torch

from lamb.utils import env_flag, pick_attn_implementation, pick_device, pick_dtype


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got: {v!r}") from e


def pick_max_steps() -> int:
    v = _env_int("LAMB_MAX_STEPS")
    if v is not None:
        return v
    return 200


def pick_debug_every_steps() -> int:
    v = _env_int("LAMB_DEBUG_EVERY")
    if v is not None:
        return v
    return 5


@dataclass
class LaMBConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_memory_tokens: int = 1

    compressor_lora_rank: int = 0
    compressor_lora_alpha: int = 2

    batch_size: int = 1
    learning_rate: float = 1e-4
    kl_temperature: float = 1.0
    ce_alpha: float = 1.0
    device: str = field(default_factory=pick_device)
    dtype: torch.dtype = field(default_factory=lambda: pick_dtype(pick_device()))
    attn_implementation: str = field(
        default_factory=lambda: pick_attn_implementation(pick_device())
    )
    max_seq_len: int = 512
    max_steps: int = 10000

    debug_every_steps: int = 1
    debug_num_samples: int = 1
    verbose: bool = False

    tensorboard: bool = field(default_factory=lambda: env_flag("LAMB_TENSORBOARD"))
    tensorboard_logdir: str = "logs/tensorboard"
    tensorboard_every_steps: int = 5
    tensorboard_text_every_steps: int = 50

    dataset_size: int = 20000

    debug_sample_size: int = 1
    debug_repeat_count: int = 50


def default_config() -> LaMBConfig:
    return LaMBConfig()
