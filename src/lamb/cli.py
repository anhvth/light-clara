"""CLI for LaMB training"""
# pyright: reportMissingTypeStubs=false

import argparse
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


def prepare_dataset(tokenizer: Any, config: LaMBConfig) -> tuple:
    ds = load_dataset("opus100", "en-vi", split="train")

    def format_ds(x: dict[str, Any]) -> dict[str, str]:
        translation = x["translation"]
        return {"content": _format_translation(tokenizer, translation)}

    if config.debug_repeat_count > 0:
        ids = list(range(config.debug_sample_size))
        formatted_ds = ds.select(ids * config.debug_repeat_count).map(format_ds)
    else:
        formatted_ds = ds.select(range(config.dataset_size)).map(format_ds)
    print(f"Sample formatted data: {formatted_ds[0]['content'][:2000]}...\n")
    return formatted_ds, ds[0]


def _main(config: LaMBConfig) -> None:
    print(
        f"Running on {platform.node()} using device: {config.device}, dtype: {config.dtype}, "
        f"attn: {config.attn_implementation}"
    )
    model = LaMBModel(config)
    tokenizer = model.tokenizer

    formatted_ds, sample = prepare_dataset(tokenizer, config)
    debug_examples = [{"en": sample["translation"]["en"], "vi": sample["translation"]["vi"]}]

    train(model, formatted_ds, tokenizer, config, debug_examples=debug_examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="LaMB training CLI")
    config_fields = fields(LaMBConfig)

    for cfg_field in config_fields:
        name = cfg_field.name
        default = cfg_field.default if cfg_field.default is not MISSING else None
        if cfg_field.default_factory is not MISSING:
            default = cfg_field.default_factory()

        if cfg_field.type in (str, int, float, bool):
            parser.add_argument(
                f"--{name}", type=cfg_field.type, default=default, help=f"Set {name}"
            )

    args = parser.parse_args()

    config = LaMBConfig()
    for cfg_field in config_fields:
        if hasattr(args, cfg_field.name):
            setattr(config, cfg_field.name, getattr(args, cfg_field.name))

    _main(config)


if __name__ == "__main__":
    main()
