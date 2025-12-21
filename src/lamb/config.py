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
class ClaraConfig:
    """CLaRa (Compressing Language for Retrieval Augmentation) Configuration.

    Supports three training stages:
    - stage1: Compression pretraining (QA + paraphrase + MSE loss)
    - stage1_2: Compression instruction tuning
    - stage2: End-to-end retrieval training (optional)
    """

    # Model
    model_name: str = "Qwen/Qwen3-0.6B"

    # Training stage
    stage: str = "stage1"  # stage1, stage1_2, stage2

    # Compression
    compress_rate: int = 32  # Number of memory tokens per document
    doc_max_length: int = 256  # Max tokens per document
    use_compressor_mlp: bool = True  # Use MLP vs linear projection
    compressor_mlp_hidden_dim: int | None = None  # Default: hidden_size * 4

    # LoRA configuration
    encoder_lora_rank: int = 16
    decoder_lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Loss weights
    use_mse_loss: bool = True
    mse_weight: float = 0.1
    use_paraphrase_loss: bool = True
    paraphrase_weight: float = 1.0
    qa_weight: float = 1.0

    # Data
    generation_top_k: int = 1  # Number of docs to compress per example
    max_seq_len: int = 1024  # Max decoder sequence length

    # Training
    batch_size: int = 2
    learning_rate: float = 1e-4
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Device
    device: str = field(default_factory=pick_device)
    dtype: torch.dtype = field(default_factory=lambda: pick_dtype(pick_device()))
    attn_implementation: str = field(
        default_factory=lambda: pick_attn_implementation(pick_device())
    )

    # Checkpointing
    save_steps: int = 500
    checkpoint_dir: str = "checkpoints"
    load_from_checkpoint: str | None = None

    # Evaluation & Logging
    eval_steps: int = 100
    do_eval: bool = False
    verbose: bool = False

    # TensorBoard
    tensorboard: bool = field(default_factory=lambda: env_flag("LAMB_TENSORBOARD"))
    tensorboard_logdir: str = "logs/tensorboard"
    tensorboard_every_steps: int = 5

    # Debug Mode
    debug_mode: bool = False  # Enable debug features
    debug_every_steps: int = 10  # Generate and print colored output every N steps
    debug_num_samples: int = 3  # Number of samples to generate in debug
    debug_repeat_dataset: int = 0  # Repeat dataset N times (0 = no repeat)


# Backward compatibility alias
LaMBConfig = ClaraConfig


def default_config() -> ClaraConfig:
    return ClaraConfig()
