"""CLI for LaMB training"""
# pyright: reportMissingTypeStubs=false

import argparse
import os
import platform
from typing import Any, Dict

from datasets import load_dataset

from lamb.config import LaMBConfig, default_config
from lamb.model import LaMBModel
from lamb.train import train


def _format_translation(tokenizer: Any, translation: Dict[str, str]) -> str:
    result = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a translator from English to Vietnamese"},
            {"role": "user", "content": translation["en"]},
            {"role": "assistant", "content": translation["vi"]},
        ],
        tokenize=False,
    )
    return str(result)


def prepare_dataset(tokenizer: Any, dataset_size: int) -> tuple:
    ds = load_dataset("opus100", "en-vi", split="train")

    def format_ds(x: Dict[str, Any]) -> Dict[str, str]:
        translation = x["translation"]
        return {"content": _format_translation(tokenizer, translation)}

    formatted_ds = ds.select(range(dataset_size)).map(format_ds)
    print(f"Sample formatted data: {formatted_ds[0]['content'][:2000]}...\n")
    return formatted_ds, ds[0]


def build_config(args: argparse.Namespace) -> LaMBConfig:
    cfg = default_config()
    cfg.model_name = args.model_name or cfg.model_name
    cfg.batch_size = args.batch_size or cfg.batch_size
    cfg.learning_rate = args.learning_rate or cfg.learning_rate
    cfg.max_seq_len = args.max_seq_len or cfg.max_seq_len
    cfg.debug_every_steps = args.debug_every_steps or cfg.debug_every_steps

    max_steps_env = os.environ.get("LAMB_MAX_STEPS")
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    elif max_steps_env is not None:
        cfg.max_steps = int(max_steps_env)
    else:
        cfg.max_steps = 200

    debug_every_env = os.environ.get("LAMB_DEBUG_EVERY")
    if debug_every_env is not None and args.debug_every_steps is None:
        cfg.debug_every_steps = int(debug_every_env)

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LaMB translator")
    parser.add_argument("--model-name", type=str, default=None, help="HF model name")
    parser.add_argument("--dataset-size", type=int, default=20000, help="Number of training samples to use")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Truncation length")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many steps")
    parser.add_argument("--debug-every-steps", type=int, default=None, help="Debug interval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)

    print(
        f"Running on {platform.node()} using device: {config.device}, dtype: {config.dtype}, "
        f"attn: {config.attn_implementation}"
    )
    model = LaMBModel(config)
    tokenizer = model.tokenizer

    formatted_ds, sample = prepare_dataset(tokenizer, dataset_size=args.dataset_size)
    debug_examples = [{"en": sample["translation"]["en"], "vi": sample["translation"]["vi"]}]

    train(model, formatted_ds, tokenizer, config, debug_examples=debug_examples)


if __name__ == "__main__":
    main()
