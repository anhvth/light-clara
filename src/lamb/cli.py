"""CLI for CLaRa training.

Supports multi-stage training:
- Stage 1: Compression pretraining (QA + paraphrase + MSE loss)
- Stage 1.2: Compression instruction tuning
- Stage 2: End-to-end retrieval training (future)
"""
# pyright: reportMissingTypeStubs=false

import argparse
import platform
from dataclasses import MISSING, fields
from typing import Any

from lamb.clara_data import export_debug_jsonl, iter_clara_examples
from lamb.clara_model import ClaraModel
from lamb.clara_train import train_stage1
from lamb.config import ClaraConfig


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
            out_path=export_path,
            dataset_name=dataset_name,
            split=dataset_split,
            streaming=dataset_streaming,
            limit=dataset_limit,
        )
        print(f"[Data] Exported {n} examples to {export_path}")

    examples = list(
        iter_clara_examples(
            dataset_name=dataset_name,
            split=dataset_split,
            streaming=dataset_streaming,
            limit=dataset_limit,
        )
    )
    if not examples:
        raise RuntimeError(
            f"No examples loaded from {dataset_name} split={dataset_split} streaming={dataset_streaming}"
        )

    # Convert to dict format for collate function
    dataset = []
    for ex in examples:
        dataset.append(
            {
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
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="apple/CLaRa_multi_stage",
        help="HF dataset name (default: apple/CLaRa_multi_stage)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="HF split to use; use test for quick iteration.",
    )
    parser.add_argument(
        "--dataset_streaming",
        type=bool,
        default=True,
        help="Use streaming to avoid full download.",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=64,
        help="Max examples to stream/load.",
    )
    parser.add_argument(
        "--dataset_export_jsonl",
        type=str,
        default="",
        help="If set, export the streamed subset to JSONL at this path.",
    )

    args = parser.parse_args()

    config = ClaraConfig()
    for cfg_field in config_fields:
        if hasattr(args, cfg_field.name):
            setattr(config, cfg_field.name, getattr(args, cfg_field.name))

    # Attach extra dataset args to config
    config.dataset_name = args.dataset_name
    config.dataset_split = args.dataset_split
    config.dataset_streaming = args.dataset_streaming
    config.dataset_limit = args.dataset_limit
    config.dataset_export_jsonl = args.dataset_export_jsonl

    _main(config)


if __name__ == "__main__":
    main()
