import os
import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from lamb.utils import pick_attn_implementation, pick_device, pick_dtype


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

    # Tokenizer
    # If set, loads this tokenizer and copies its chat_template onto the main tokenizer.
    tokenizer_template: str = ""

    # Training stage
    stage: str = "stage1"  # stage1, stage1_2, stage2

    # Compression
    compress_rate: int = 32  # Number of memory tokens per document
    doc_max_length: int = 256  # Max tokens per document
    use_compressor_mlp: bool = True  # Use MLP vs linear projection
    compressor_mlp_hidden_dim: int | None = None  # Default: hidden_size * 4
    encoder_pool_method: Literal["token_softmax", "mean", "mpl", "nearn", "max"] = "token_softmax"

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
    # If 0, defaults to learning_rate / 10 (generator is 10x slower than encoder).
    generator_learning_rate: float = 0.0
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Device
    device: str = field(default_factory=pick_device)
    dtype: torch.dtype = field(init=False)
    bf16: bool = False
    fp16: bool = False
    attn_implementation: str = ""

    def __post_init__(self) -> None:
        if self.bf16 and self.fp16:
            raise ValueError("Cannot set both bf16 and fp16")
        if self.bf16 or self.fp16:
            if self.device != "cuda":
                raise ValueError("--bf16/--fp16 require device cuda")
            if self.bf16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError("cuda device does not support bfloat16")
            self.dtype = torch.bfloat16 if self.bf16 else torch.float16
        else:
            self.dtype = pick_dtype(self.device)

        if not self.attn_implementation:
            self.attn_implementation = pick_attn_implementation(self.device)

        # Normalize logging backend.
        report_to = (self.report_to or "").strip().lower()
        valid_report_to = {"tensorboard", "wandb", "none", ""}
        if report_to not in valid_report_to:
            raise ValueError(
                "report_to must be one of: tensorboard, wandb, none (or empty for auto)"
            )

        if not report_to:
            report_to = "none"

        self.report_to = report_to

        if not self.run_name:
            self.run_name = time.strftime("%m-%d-%H-%M")

    # Checkpointing
    save_steps: int = 500
    checkpoint_dir: str = "checkpoints"
    load_from_checkpoint: str | None = None

    # Evaluation & Logging
    eval_steps: int = 100
    do_eval: bool = False
    verbose: bool = False
    # Logging backend (huggingface-style).
    report_to: str = ""
    run_name: str = ""
    log_dir: str = "logs/runs"
    # Logging cadence in optimizer update steps (i.e., parameter updates).
    log_every_steps: int = 5
    log_text_every_steps: int = 0

    # Weights & Biases
    wandb_project: str = "clara"
    wandb_entity: str = ""

    # Debug Mode
    debug_mode: bool = False  # Enable debug features
    # Debug cadence in optimizer update steps (i.e., parameter updates). If
    # gradient_accumulation_steps > 1, debug output appears every
    # debug_every_steps * gradient_accumulation_steps dataloader batches.
    debug_every_steps: int = 10
    debug_num_samples: int = 3  # Number of samples to generate in debug
    debug_repeat_dataset: int = 0  # Repeat dataset N times (0 = no repeat)

    # Dataset (for CLI)
    dataset_name: str = "apple/CLaRa_multi_stage"
    dataset_split: str = "test"
    dataset_streaming: bool = True
    dataset_limit: int = 64
    dataset_export_jsonl: str = ""


# Backward compatibility alias
LaMBConfig = ClaraConfig


def default_config() -> ClaraConfig:
    return ClaraConfig()
