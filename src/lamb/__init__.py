from .bridge import DocumentCompressor
from .clara_collate import make_stage1_collate_fn, stage1_collate_fn
from .clara_model import ClaraModel
from .clara_train import train_stage1
from .config import ClaraConfig, LaMBConfig, default_config

__all__ = [
    # CLaRa (new architecture)
    "ClaraConfig",
    "ClaraModel",
    "DocumentCompressor",
    "make_stage1_collate_fn",
    "stage1_collate_fn",
    "train_stage1",
    # Legacy config (backward compat)
    "LaMBConfig",
    "default_config",
]
