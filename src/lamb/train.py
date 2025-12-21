import json
import os
import time
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lamb.debug import debug_reproduce_training

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from lamb.config import LaMBConfig
    from lamb.model import LaMBModel


def train(  # noqa: PLR0915
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
            print(  # type: ignore[unreachable]
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

    # Histories for last 100 steps
    loss_history: list[float] = []
    token_acc_history: list[float] = []
    teacher_student_acc_history: list[float] = []

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
        batch_loss = 0.0
        batch_token_correct, batch_token_total = 0, 0
        batch_teacher_student_correct, batch_teacher_student_total = 0, 0
        saw_teacher_student = False
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
            batch_token_correct += int(metrics.get("token_correct", 0))
            batch_token_total += int(metrics.get("token_total", 0))

            if "teacher_student_total" in metrics:
                saw_teacher_student = True
                batch_teacher_student_correct += int(metrics.get("teacher_student_correct", 0))
                batch_teacher_student_total += int(metrics.get("teacher_student_total", 0))

            scale = 1.0 / float(config.batch_size)
            if "kl_loss" in metrics:
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
                if debug_txt is not None:
                    debug_reproduce_training(
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

                # if ok and False:
                #     print(f"[Train] Overfit success at step={step}; stopping.")
                #     return
            finally:
                if was_training:
                    model.train()

        avg_loss = float(batch_loss)
        avg_token_acc = (
            (float(batch_token_correct) / float(batch_token_total)) if batch_token_total else 0.0
        )
        avg_teacher_student_acc = (
            (float(batch_teacher_student_correct) / float(batch_teacher_student_total))
            if batch_teacher_student_total
            else 0.0
        )

        # Update histories
        loss_history.append(avg_loss)
        token_acc_history.append(avg_token_acc)
        if saw_teacher_student:
            teacher_student_acc_history.append(avg_teacher_student_acc)

        # Keep only last 100
        if len(loss_history) > 100:
            loss_history.pop(0)
        if len(token_acc_history) > 100:
            token_acc_history.pop(0)
        if len(teacher_student_acc_history) > 100:
            teacher_student_acc_history.pop(0)

        # Compute running averages
        running_avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0
        running_avg_token_acc = (
            sum(token_acc_history) / len(token_acc_history) if token_acc_history else 0.0
        )
        running_avg_teacher_student_acc = (
            sum(teacher_student_acc_history) / len(teacher_student_acc_history)
            if teacher_student_acc_history
            else 0.0
        )
        avg_kl = (float(batch_kl) / float(batch_count)) if batch_count else 0.0
        avg_ce = (float(batch_ce) / float(batch_count)) if batch_count else 0.0
        avg_alpha = (float(batch_alpha) / float(batch_count)) if batch_count else 0.0

        postfix: dict[str, str] = {
            "loss": f"{avg_loss:.4f}",
            "loss_avg": f"{running_avg_loss:.4f}",
            "token_acc": f"{avg_token_acc * 100:.2f}%",
            "token_acc_avg": f"{running_avg_token_acc * 100:.2f}%",
        }
        if saw_teacher_student:
            postfix["teacher_student_acc"] = f"{avg_teacher_student_acc * 100:.2f}%"
            postfix["teacher_student_acc_avg"] = f"{running_avg_teacher_student_acc * 100:.2f}%"
        pbar.set_postfix(ordered_dict=postfix)

        if writer is not None and tb_every_steps and (step % tb_every_steps == 0):
            try:
                lr = float(optimizer.param_groups[0].get("lr", 0.0))
            except Exception:
                lr = 0.0

            writer.add_scalar("train/loss_step", avg_loss, step)
            writer.add_scalar("train/loss_smoothed", running_avg_loss, step)
            writer.add_scalar("train/token_acc_step", avg_token_acc, step)
            writer.add_scalar("train/token_acc_smoothed", running_avg_token_acc, step)
            if saw_teacher_student:
                writer.add_scalar("train/teacher_student_acc_step", avg_teacher_student_acc, step)
                writer.add_scalar(
                    "train/teacher_student_acc_smoothed", running_avg_teacher_student_acc, step
                )
            if saw_teacher_student:
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
