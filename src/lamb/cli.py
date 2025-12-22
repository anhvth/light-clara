"""CLI for CLaRa training.

Supports multi-stage training:
 Stage 1: Compression pretraining (QA + paraphrase + MSE loss)
 Stage 1.2: Compression instruction tuning
 Stage 2: End-to-end retrieval training (future)
"""

import argparse
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import MISSING, Field, fields
from typing import Any

import unsloth  # noqa: F401

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from lamb.clara_data import export_debug_jsonl, iter_clara_examples
from lamb.clara_model import ClaraModel
from lamb.clara_train import train_stage1
from lamb.config import ClaraConfig


def _maybe_print_env_diagnostics() -> None:
    if os.environ.get("LAMB_ENV_DIAG", "").strip().lower() not in {"1", "true", "yes", "y", "on"}:
        return

    print("[Env] Diagnostics")
    print(f"[Env] python={sys.executable}")
    print(f"[Env] version={sys.version.splitlines()[0]}")
    print(f"[Env] prefix={sys.prefix}")
    print(f"[Env] sys.path[0:3]={sys.path[:3]}")

    try:
        import numpy as np  # type: ignore

        print(f"[Env] numpy={np.__version__} file={getattr(np, '__file__', '')}")
    except Exception as e:
        print(f"[Env] numpy import failed: {type(e).__name__}: {e}")


def _format_clara_sft(tokenizer: Any, *, question: str, docs: list[str], answer: str) -> str:
    background = "\n\n".join(docs)
    user = f"<background>\n{background}\n</background>\n\nQuestion: {question}\n"
    result = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "Answer the question using the provided background.",
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
    )
    return str(result)


def prepare_dataset(config: ClaraConfig) -> list[dict[str, Any]]:
    """Load CLaRa dataset examples.

    Returns list of dicts with keys: data_type, question, docs, answer
    """
    dataset_name = str(getattr(config, "dataset_name", "apple/CLaRa_multi_stage"))
    dataset_split = str(getattr(config, "dataset_split", "test"))
    dataset_streaming = bool(getattr(config, "dataset_streaming", True))
    dataset_limit = int(getattr(config, "dataset_limit", 64))
    export_path = str(getattr(config, "dataset_export_jsonl", ""))

    if export_path:
        n = export_debug_jsonl(
            dataset_export_jsonl=export_path,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            dataset_streaming=dataset_streaming,
            dataset_limit=dataset_limit,
        )
        print(f"[Data] Exported {n} examples to {export_path}")

    examples = list(
        iter_clara_examples(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            dataset_streaming=dataset_streaming,
            dataset_limit=dataset_limit,
        )
    )
    if not examples:
        raise RuntimeError(
            f"No examples loaded from {dataset_name} split={dataset_split} streaming={dataset_streaming}"
        )

    # Convert to dict format for collate function
    dataset = []
    for idx, ex in enumerate(examples):
        dataset.append(
            {
                "idx": idx,
                "data_type": ex.data_type,
                "question": ex.question,
                "docs": ex.docs,
                "answer": ex.answer,
            }
        )

    print(
        f"[Data] Loaded {len(dataset)} examples from {dataset_name} "
        f"(split={dataset_split}, streaming={dataset_streaming})"
    )
    print(f"[Data] Sample: q='{dataset[0]['question'][:80]}...' docs={len(dataset[0]['docs'])}\n")
    return dataset


def _print_config(config_fields: Sequence[Field[Any]], config: ClaraConfig) -> None:
    if tabulate is None:
        print("[CLaRa] Install tabulate (uv add tabulate) to see the config table.")
        return

    rows = [(cfg_field.name, getattr(config, cfg_field.name)) for cfg_field in config_fields]
    print("\n[CLaRa Config]")
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="github"))


def _main(config: ClaraConfig) -> None:
    print(f"[CLaRa] Running on {platform.node()}")
    print(
        f"[CLaRa] Device: {config.device}, dtype: {config.dtype}, attn: {config.attn_implementation}"
    )
    print(f"[CLaRa] Stage: {config.stage}\n")

    # Create model
    model = ClaraModel(config)

    # Load dataset
    dataset = prepare_dataset(config)

    # Train based on stage
    if config.stage in ("stage1", "stage1_2"):
        train_stage1(model, dataset, config)
    elif config.stage == "stage2":
        raise NotImplementedError("Stage 2 training not yet implemented")
    else:
        raise ValueError(f"Unknown stage: {config.stage}")


def main() -> None:
    _maybe_print_env_diagnostics()
    parser = argparse.ArgumentParser(description="CLaRa training CLI")
    config_fields = fields(ClaraConfig)

    for cfg_field in config_fields:
        name = cfg_field.name
        default = cfg_field.default if cfg_field.default is not MISSING else None
        if cfg_field.default_factory is not MISSING:
            default = cfg_field.default_factory()

        if cfg_field.type in (str, int, float, bool):
            parser.add_argument(
                f"--{name}", type=cfg_field.type, default=default, help=f"Set {name}"
            )

    # Extra dataset args (kept here to avoid overloading the config dataclass).
    # Note: dataset args are now in ClaraConfig to avoid duplication.

    args = parser.parse_args()

    config = ClaraConfig()
    for cfg_field in config_fields:
        if hasattr(args, cfg_field.name):
            setattr(config, cfg_field.name, getattr(args, cfg_field.name))

    _print_config(config_fields, config)

    _main(config)


if __name__ == "__main__":
    main()
