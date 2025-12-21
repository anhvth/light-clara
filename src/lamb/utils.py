import os
from typing import Optional, Tuple

import torch
from torch.nn import functional

try:
    import flash_attn  # type: ignore
except ImportError:
    flash_attn = None


def env_flag(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def pick_device() -> str:
    forced = os.environ.get("LAMB_DEVICE")
    if forced:
        forced = forced.strip().lower()
        if forced in {"cuda", "mps", "cpu"}:
            if forced == "cuda" and torch.cuda.is_available():
                return "cuda"
            if (
                forced == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"
            if forced == "cpu":
                return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    if device == "mps":
        return torch.float16 if env_flag("LAMB_MPS_FP16") else torch.float32
    return torch.float32


def sdpa_available() -> bool:
    return hasattr(functional, "scaled_dot_product_attention")


def flash_attn_available() -> bool:
    if env_flag("LAMB_DISABLE_FLASH_ATTN"):
        return False
    try:
        return flash_attn is not None
    except Exception:
        return False


def pick_attn_implementation(device: str) -> str:
    if device == "cuda" and flash_attn_available():
        return "flash_attention_2"
    if sdpa_available():
        return "sdpa"
    return "eager"


def apply_rotary_pos_emb(
    q: Optional[torch.Tensor], k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q, k_embed
