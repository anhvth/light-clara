import json
import os
import time
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lamb.debug import debug_reproduce_training

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:
    SummaryWriter = None

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
    debug_examples: list[dict] | None = None,
) -> None:
    print(f"\n[Train] Starting Epoch on {len(dataset)} samples...")
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=config.learning_rate)
    model.train()

    writer = None
    tb_enabled = bool(getattr(config, "tensorboard", False))
    tb_logdir = str(getattr(config, "tensorboard_logdir", "logs/tensorboard"))
    tb_every_steps = int(getattr(config, "tensorboard_every_steps", 0) or 0)
    tb_text_every_steps = int(getattr(config, "tensorboard_text_every_steps", 0) or 0)

    if tb_enabled:
        if SummaryWriter is None:
            print(
                "[TB] TensorBoard logging requested but unavailable. "
                "Install with: uv add tensorboard. "
            )
            writer = None
        else:
            run_dir = os.path.join(tb_logdir, time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=run_dir)
            print(f"[TB] Logging to {run_dir}")

            try:
                cfg_txt = json.dumps(config.__dict__, default=str, indent=2)
            except Exception:
                cfg_txt = str(config)
            writer.add_text("run/config", cfg_txt, global_step=0)
            writer.add_text(
                "run/env",
                f"device={config.device} dtype={config.dtype} attn={getattr(config, 'attn_implementation', None)}",
                global_step=0,
            )

    def collate_fn(batch: list[dict]) -> list[str]:
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

    pbar_total: int | None = None
    if max_steps and dl_len is not None:
        pbar_total = min(int(dl_len), int(max_steps))
    elif max_steps:
        pbar_total = int(max_steps)
    elif dl_len is not None:
        pbar_total = int(dl_len)

    pbar = tqdm(dataloader, desc="Train", dynamic_ncols=True, total=pbar_total)

    for batch_txt in pbar:
        debug_txt: str | None = None
        batch_loss, batch_correct, batch_total = 0.0, 0, 0
        batch_kl, batch_ce, batch_alpha = 0.0, 0.0, 0.0
        batch_count = 0
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

            scale = 1.0 / float(config.batch_size)
            batch_kl += float(metrics.get("kl_loss", 0.0)) * scale
            batch_ce += float(metrics.get("ce_loss", 0.0)) * scale
            batch_alpha += float(metrics.get("ce_alpha", 0.0))
            batch_count += 1

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
                dbg_stats: dict[str, Any] = {}
                ok = False
                if debug_txt is not None:
                    ok = debug_reproduce_training(
                        model,
                        txt=debug_txt,
                        verbose=config.verbose,
                        stats_out=dbg_stats,
                    )

                if writer is not None and dbg_stats:
                    if "teacher_forced_acc" in dbg_stats:
                        writer.add_scalar(
                            "dbg/teacher_forced_acc", dbg_stats["teacher_forced_acc"], step
                        )
                    if "lm_loss" in dbg_stats:
                        writer.add_scalar("dbg/lm_loss", dbg_stats["lm_loss"], step)
                    if "greedy_matches" in dbg_stats:
                        writer.add_scalar(
                            "dbg/greedy_matches", float(bool(dbg_stats["greedy_matches"])), step
                        )

                if ok:
                    print(f"[Train] Overfit success at step={step}; stopping.")
                    return
            finally:
                if was_training:
                    model.train()

        avg_loss = float(batch_loss)
        avg_acc = (float(batch_correct) / float(batch_total)) if batch_total else 0.0
        avg_kl = (float(batch_kl) / float(batch_count)) if batch_count else 0.0
        avg_ce = (float(batch_ce) / float(batch_count)) if batch_count else 0.0
        avg_alpha = (float(batch_alpha) / float(batch_count)) if batch_count else 0.0
        momentum = 0.9
        running_loss = momentum * running_loss + (1 - momentum) * avg_loss
        running_acc = momentum * running_acc + (1 - momentum) * avg_acc

        pbar.set_postfix(loss=f"{running_loss:.4f}", agree_acc=f"{running_acc * 100:.2f}%")

        if writer is not None and tb_every_steps and (step % tb_every_steps == 0):
            try:
                lr = float(optimizer.param_groups[0].get("lr", 0.0))
            except Exception:
                lr = 0.0

            writer.add_scalar("train/loss_step", avg_loss, step)
            writer.add_scalar("train/loss_smoothed", running_loss, step)
            writer.add_scalar("train/agree_acc_step", avg_acc, step)
            writer.add_scalar("train/agree_acc_smoothed", running_acc, step)
            writer.add_scalar("train/kl_loss_step", avg_kl, step)
            writer.add_scalar("train/ce_loss_step", avg_ce, step)
            writer.add_scalar("train/ce_alpha", avg_alpha, step)
            writer.add_scalar("train/lr", lr, step)

            if tb_text_every_steps and (step % tb_text_every_steps == 0) and debug_txt is not None:
                writer.add_text("samples/raw", debug_txt[:2000], step)

        if step % 5 == 0 and config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if writer is not None:
        writer.flush()
        writer.close()
