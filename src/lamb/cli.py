"""CLI for LaMB training"""
# pyright: reportMissingTypeStubs=false

import argparse
import os
import platform
from dataclasses import MISSING, fields
from typing import Any

from datasets import load_dataset

from lamb.config import LaMBConfig
from lamb.model import LaMBModel
from lamb.train import train


def _format_translation(tokenizer: Any, translation: dict[str, str]) -> str:
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

    def format_ds(x: dict[str, Any]) -> dict[str, str]:
        translation = x["translation"]
        return {"content": _format_translation(tokenizer, translation)}

    formatted_ds = ds.select(range(dataset_size)).map(format_ds)
    print(f"Sample formatted data: {formatted_ds[0]['content'][:2000]}...\n")
    return formatted_ds, ds[0]


def _main(config: LaMBConfig) -> None:
    # Apply environment variable overrides
    max_steps_env = os.environ.get("LAMB_MAX_STEPS")
    if config.max_steps is None:
        if max_steps_env is not None:
            config.max_steps = int(max_steps_env)
        else:
            config.max_steps = 200

    debug_every_env = os.environ.get("LAMB_DEBUG_EVERY")
    if debug_every_env is not None and config.debug_every_steps == 5:  # assuming default
        config.debug_every_steps = int(debug_every_env)

    print(
        f"Running on {platform.node()} using device: {config.device}, dtype: {config.dtype}, "
        f"attn: {config.attn_implementation}"
    )
    model = LaMBModel(config)
    tokenizer = model.tokenizer

    formatted_ds, sample = prepare_dataset(tokenizer, dataset_size=config.dataset_size)
    debug_examples = [{"en": sample["translation"]["en"], "vi": sample["translation"]["vi"]}]

    train(model, formatted_ds, tokenizer, config, debug_examples=debug_examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="LaMB training CLI")
    config_fields = fields(LaMBConfig)

    for field in config_fields:
        name = field.name
        default = field.default if field.default is not MISSING else None
        if field.default_factory is not MISSING:
            default = field.default_factory()

        # Add arguments for simple types
        if field.type in (str, int, float, bool):
            parser.add_argument(f"--{name}", type=field.type, default=default, help=f"Set {name}")

    args = parser.parse_args()

    config = LaMBConfig()
    for field in config_fields:
        if hasattr(args, field.name):
            setattr(config, field.name, getattr(args, field.name))

    _main(config)


if __name__ == "__main__":
    main()
