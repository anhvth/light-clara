from typing import TYPE_CHECKING, Any, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lamb.debug import debug_reproduce_training

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from lamb.config import LaMBConfig
    from lamb.model import LaMBModel


def train(  # noqa: PLR0912,PLR0915
    model: "LaMBModel",
    dataset: Any,
    tokenizer: "PreTrainedTokenizer",
    config: "LaMBConfig",
    *,
    debug_examples: Optional[List[dict]] = None,
) -> None:
    print(f"\n[Train] Starting Epoch on {len(dataset)} samples...")
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=config.learning_rate)
    model.train()

    def collate_fn(batch: List[dict]) -> List[str]:
        return [b["content"] for b in batch]

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    step = 0
    optimizer.zero_grad()
    running_loss, running_acc = 0.0, 0.0

    max_steps = int(getattr(config, "max_steps", 0) or 0)

    try:
        dl_len = len(dataloader)
    except TypeError:
        dl_len = None

    pbar_total: Optional[int] = None
    if max_steps and dl_len is not None:
        pbar_total = min(int(dl_len), int(max_steps))
    elif max_steps:
        pbar_total = int(max_steps)
    elif dl_len is not None:
        pbar_total = int(dl_len)

    pbar = tqdm(dataloader, desc="Train", dynamic_ncols=True, total=pbar_total)

    for batch_txt in pbar:
        debug_txt: Optional[str] = None
        batch_loss, batch_correct, batch_total = 0.0, 0, 0
        for txt in batch_txt:
            debug_txt = txt
            splitter = "<|im_start|>assistant\n"
            if splitter not in txt:
                splitter = "assistant\n"
            try:
                ctx_str, tgt_str = txt.split(splitter, 1)
            except Exception:
                mid = len(txt) // 2
                ctx_str, tgt_str = txt[:mid], txt[mid:]

            ctx_str += splitter
            ctx = tokenizer.encode(ctx_str, add_special_tokens=False)
            tgt = tokenizer.encode(tgt_str, add_special_tokens=False)

            full = ctx + tgt
            if len(full) > config.max_seq_len:
                full = full[: config.max_seq_len]
                split_idx = [min(len(ctx), config.max_seq_len - 1)]
            else:
                split_idx = [len(ctx)]

            inp = torch.tensor([full]).to(config.device)
            loss, metrics = model(inp, split_idx, return_metrics=True)
            loss = loss / config.batch_size

            if not torch.isfinite(loss):
                print("\n[Train] Non-finite loss encountered; stopping.")
                continue

            loss.backward()
            batch_loss += loss.item()
            batch_correct += int(metrics.get("agree_correct", 0))
            batch_total += int(metrics.get("agree_total", 0))

        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if max_steps and step >= max_steps:
            break

        if (
            debug_examples
            and config.debug_every_steps
            and step % int(config.debug_every_steps) == 0
        ):
            was_training = model.training
            try:
                if debug_txt is not None and debug_reproduce_training(
                    model, txt=debug_txt, verbose=config.verbose
                ):
                    print(f"[Train] Overfit success at step={step}; stopping.")
                    return
            finally:
                if was_training:
                    model.train()

        avg_loss = float(batch_loss)
        avg_acc = (float(batch_correct) / float(batch_total)) if batch_total else 0.0
        momentum = 0.9
        running_loss = momentum * running_loss + (1 - momentum) * avg_loss
        running_acc = momentum * running_acc + (1 - momentum) * avg_acc

        pbar.set_postfix(loss=f"{running_loss:.4f}", agree_acc=f"{running_acc * 100:.2f}%")

        if step % 5 == 0 and config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
