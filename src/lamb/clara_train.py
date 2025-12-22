"""CLaRa Training Loop - Stage 1 (Compression Pretraining)."""

import contextlib
import io
import os
import re
import sys
import time
from typing import IO, Any, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lamb.clara_collate import make_stage1_collate_fn
from lamb.clara_model import ClaraModel
from lamb.config import ClaraConfig
from lamb.debug import debug_reproduce_clara

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore

try:
    import ipdb
except ImportError:
    ipdb = None  # type: ignore

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", str(s))


class _Tee(io.TextIOBase):
    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            with contextlib.suppress(Exception):
                st.flush()


def train_stage1(
    model: ClaraModel,
    dataset: Any,
    config: ClaraConfig,
) -> None:
    """Train CLaRa Stage 1: Compression Pretraining.

    Combines three losses:
    - QA loss: Answer generation from compressed documents
    - Paraphrase loss: Paraphrase generation from compressed documents
    - MSE loss (optional): Compression preserves encoder representations
    """
    # Handle debug mode dataset repetition
    original_dataset_size = len(dataset)
    if config.debug_mode and config.debug_repeat_dataset > 0:
        print(f"\x1b[93m[Debug Mode] Repeating dataset {config.debug_repeat_dataset}x\x1b[0m")
        dataset = dataset * (config.debug_repeat_dataset + 1)

    print(
        f"\n[Train Stage 1] Starting on {len(dataset)} samples (original: {original_dataset_size})..."
    )
    print(f"[Train Stage 1] Compress rate: {config.compress_rate} tokens/doc")
    print(
        f"[Train Stage 1] Losses: QA={config.qa_weight}, Paraphrase={config.paraphrase_weight}, MSE={config.mse_weight if config.use_mse_loss else 0}"
    )
    if config.use_clara_original:
        print("[Train Stage 1] Using original CLaRa stage-1 formatting (data + loss)")
    if config.debug_mode:
        debug_batch_interval = config.debug_every_steps * max(1, config.gradient_accumulation_steps)
        suffix = (
            f" ({debug_batch_interval} dataloader batches)"
            if debug_batch_interval != config.debug_every_steps
            else ""
        )
        print(
            f"\x1b[92m[Debug Mode] Enabled - will show color-coded tokens every {config.debug_every_steps} optimizer step(s){suffix}\x1b[0m"
        )

    # Setup optimizer with different learning rates:
    # - encoder side: encoder_adapter + compressor + memory token embeddings
    # - generator side: decoder_adapter (default 10x slower)
    encoder_lr = float(getattr(config, "learning_rate", 0.0) or 0.0)
    if encoder_lr <= 0:
        raise ValueError(f"learning_rate must be > 0, got {encoder_lr}")

    gen_lr_cfg = float(getattr(config, "generator_learning_rate", 0.0) or 0.0)
    generator_lr = gen_lr_cfg if gen_lr_cfg > 0 else (encoder_lr / 10.0)

    encoder_params: list[torch.nn.Parameter] = []
    generator_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, param in model.base_model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        if "decoder_adapter" in name:
            generator_params.append(param)
        else:
            encoder_params.append(param)

    for param in model.compressor.parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        encoder_params.append(param)

    if not encoder_params and not generator_params:
        raise RuntimeError("No trainable parameters found for optimizer")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": generator_params, "lr": generator_lr},
        ],
        weight_decay=0.01,
    )

    print(
        f"[Train Stage 1] LR encoder={encoder_lr:g} generator={generator_lr:g} "
        f"(generator is {'custom' if gen_lr_cfg > 0 else 'learning_rate/10'})"
    )

    # Setup data loader
    collate_fn = make_stage1_collate_fn(
        tokenizer=model.tokenizer,
        doc_max_length=config.doc_max_length,
        dec_max_length=config.max_seq_len,
        generation_top_k=config.generation_top_k,
        num_mem_tokens=config.compress_rate,
        use_clara_original=config.use_clara_original,
        use_sep_token=config.use_sep_token or config.use_clara_original,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    report_to = str(getattr(config, "report_to", "none") or "none").lower()
    run_name = str(getattr(config, "run_name", "") or time.strftime("%m-%d-%H-%M"))

    writer: SummaryWriter | None = None
    wandb_run: Any | None = None
    log_every = int(getattr(config, "log_every_steps", 0) or 0)

    if report_to == "tensorboard":
        if SummaryWriter is None:
            print(
                "[TensorBoard] Logging requested but SummaryWriter unavailable. Install with: uv add tensorboard"
            )
        else:
            run_dir = os.path.join(getattr(config, "log_dir", "logs/runs"), run_name)
            os.makedirs(run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=run_dir)
            print(f"[TensorBoard] Logging to {run_dir}")

            writer.add_scalar("train/lr_encoder", encoder_lr, global_step=0)
            writer.add_scalar("train/lr_generator", generator_lr, global_step=0)
    elif report_to == "wandb":
        if wandb is None:
            print("[W&B] Logging requested but wandb is not installed. Install with: uv add wandb")
        else:
            safe_config = {
                key: (val if isinstance(val, (str, int, float, bool)) else str(val))
                for key, val in config.__dict__.items()
            }
            wandb_run = wandb.init(
                project=config.wandb_project or "clara",
                entity=config.wandb_entity or None,
                name=run_name,
                config=safe_config,
            )
            wandb_run.log(
                {
                    "train/lr_encoder": encoder_lr,
                    "train/lr_generator": generator_lr,
                },
                step=0,
            )
            print(
                f"[W&B] Logging to project={config.wandb_project or 'clara'} run={wandb_run.name}"
            )
    elif report_to != "none":
        print(f"[Logging] Unknown report_to={report_to}; logging disabled")

    def _log_scalars(metrics: dict[str, float], step: int) -> None:
        if writer is not None:
            for key, val in metrics.items():
                writer.add_scalar(key, val, step)
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    def _log_text(name: str, value: str, step: int) -> None:
        if writer is not None:
            writer.add_text(name, value, step)
        if wandb_run is not None:
            wandb_run.log({name: value}, step=step)

    # Training loop
    model.train()
    global_step = 0
    optimizer.zero_grad()

    # Metrics tracking
    running_qa_loss = 0.0
    running_paraphrase_loss = 0.0
    running_mse_loss = 0.0
    running_total_loss = 0.0

    pbar = tqdm(dataloader, desc="Stage1 Training", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        if global_step >= config.max_steps:
            break

        # Move batch to device
        doc_input_ids = batch["doc_input_ids"].to(config.device)
        doc_attention_mask = batch["doc_attention_mask"].to(config.device)
        dec_input_ids = batch["dec_input_ids"].to(config.device)
        dec_attention_mask = batch["dec_attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)
        data_types = batch["data_types"]
        num_docs_per_sample = batch["num_docs_per_sample"]

        batch_size = dec_input_ids.size(0)

        # Compress documents
        memory_embeddings, encoder_hidden_states = model.compress_documents(
            doc_input_ids, doc_attention_mask
        )

        if config.use_clara_original:
            if len(set(num_docs_per_sample)) > 1:
                raise ValueError("Original CLaRa format expects a fixed number of docs per sample")
            num_docs = num_docs_per_sample[0] if num_docs_per_sample else 1
            mem = memory_embeddings.reshape(batch_size, num_docs, -1, memory_embeddings.size(-1))
            batch_memory_embeddings_tensor = mem.reshape(batch_size, -1, mem.size(-1))
        else:
            # Reshape memory embeddings to [batch, num_docs * compress_rate, hidden_size]
            # Currently: [batch * total_docs, compress_rate, hidden_size]
            # Need to split by num_docs_per_sample
            batch_memory_embeddings = []
            doc_offset = 0
            for num_docs in num_docs_per_sample:
                sample_mem = memory_embeddings[doc_offset : doc_offset + num_docs]
                # Flatten: [num_docs, compress_rate, hidden_size] -> [num_docs * compress_rate, hidden_size]
                sample_mem_flat = sample_mem.reshape(-1, sample_mem.size(-1))
                batch_memory_embeddings.append(sample_mem_flat)
                doc_offset += num_docs

            # Pad to same length
            max_mem_tokens = max(m.size(0) for m in batch_memory_embeddings)
            padded_memory_embeddings = []
            for mem_tensor in batch_memory_embeddings:
                padded = mem_tensor
                if padded.size(0) < max_mem_tokens:
                    pad_size = max_mem_tokens - padded.size(0)
                    pad = torch.zeros(
                        pad_size, padded.size(1), dtype=padded.dtype, device=padded.device
                    )
                    padded = torch.cat([padded, pad], dim=0)
                padded_memory_embeddings.append(padded)

            batch_memory_embeddings_tensor = torch.stack(padded_memory_embeddings)

        # Forward through decoder with memory embeddings
        outputs = model.forward_with_memory(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            memory_embeddings=batch_memory_embeddings_tensor,  # will be replace(embeding(input_ids), memory)
            labels=labels,
        )

        # Compute per-sample losses
        qa_loss = torch.tensor(0.0, device=config.device)
        paraphrase_loss = torch.tensor(0.0, device=config.device)
        mse_loss = torch.tensor(0.0, device=config.device)

        num_qa = sum(1 for dt in data_types if dt != "paraphrase")
        num_paraphrase = sum(1 for dt in data_types if dt == "paraphrase")

        # Main decoder loss (already computed)
        decoder_loss = outputs["loss"]

        if config.use_clara_original:
            qa_loss = decoder_loss * config.qa_weight
            paraphrase_loss = torch.tensor(0.0, device=config.device)
        # Split into QA and paraphrase
        elif num_qa > 0 and num_paraphrase > 0:
            # Mixed batch - approximate split
            qa_loss = decoder_loss * (num_qa / batch_size) * config.qa_weight
            paraphrase_loss = (
                decoder_loss * (num_paraphrase / batch_size) * config.paraphrase_weight
            )
        elif num_paraphrase > 0:
            paraphrase_loss = decoder_loss * config.paraphrase_weight
        else:
            qa_loss = decoder_loss * config.qa_weight

        # MSE loss between compressed and encoder representations
        if config.use_mse_loss:
            if config.use_clara_original:
                # Original CLaRa MSE: mean(mem_tokens) vs mean(non_mem_tokens)
                # Need to reconstruct input_ids with memory tokens
                # encoder_hidden_states has shape [batch*docs, doc_seq_len + num_mem_tokens, hidden_size]
                # We need the corresponding input_ids
                num_mem_tokens = config.compress_rate
                batch_size = doc_input_ids.size(0)
                mem_token_ids = (
                    model.mem_token_ids.unsqueeze(0).expand(batch_size, -1).to(doc_input_ids.device)
                )
                input_ids_with_mem = torch.cat([doc_input_ids, mem_token_ids], dim=1)
                attention_mask_with_mem = torch.cat(
                    [
                        doc_attention_mask,
                        torch.ones(batch_size, num_mem_tokens, device=doc_attention_mask.device),
                    ],
                    dim=1,
                )

                mse_loss = (
                    compute_mse_loss_original(
                        encoder_hidden_states,
                        input_ids_with_mem,
                        model.mem_token_ids,
                        attention_mask_with_mem,
                    )
                    * config.mse_weight
                )
            else:
                # Custom MSE: mean-pooled comparison
                mse_loss = (
                    model.compressor.compute_mse_loss(
                        encoder_hidden_states,
                        memory_embeddings,
                        attention_mask=doc_attention_mask,
                    )
                    * config.mse_weight
                )

        # Total loss
        total_loss = qa_loss + paraphrase_loss + mse_loss

        # Backward and optimize
        if config.gradient_accumulation_steps > 1:
            total_loss = total_loss / config.gradient_accumulation_steps

        total_loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Track metrics
            running_qa_loss += qa_loss.item()
            running_paraphrase_loss += paraphrase_loss.item()
            running_mse_loss += mse_loss.item()
            running_total_loss += total_loss.item() * config.gradient_accumulation_steps

            # Log to TensorBoard/W&B
            if log_every and global_step % log_every == 0:
                avg_qa = running_qa_loss / log_every
                avg_para = running_paraphrase_loss / log_every
                avg_mse = running_mse_loss / log_every
                avg_total = running_total_loss / log_every

                _log_scalars(
                    {
                        "train/qa_loss": avg_qa,
                        "train/paraphrase_loss": avg_para,
                        "train/mse_loss": avg_mse,
                        "train/total_loss": avg_total,
                        "train/lr_encoder": encoder_lr,
                        "train/lr_generator": generator_lr,
                    },
                    global_step,
                )

                running_qa_loss = 0.0
                running_paraphrase_loss = 0.0
                running_mse_loss = 0.0
                running_total_loss = 0.0

            # Update progress bar
            pbar.set_postfix(
                {
                    "step": global_step,
                    "loss": f"{total_loss.item():.4f}",
                    "qa": f"{qa_loss.item():.3f}",
                    "para": f"{paraphrase_loss.item():.3f}",
                    "mse": f"{mse_loss.item():.4f}",
                }
            )

            # Debug generation
            if config.debug_mode and global_step % config.debug_every_steps == 0:
                tb_chunks: list[str] = []
                for debug_idx in range(min(config.debug_num_samples, batch_size)):
                    buf = io.StringIO()
                    tee = _Tee(sys.stdout, buf)
                    try:
                        with contextlib.redirect_stdout(cast(IO[str], tee)):
                            debug_reproduce_clara(model, batch, config, sample_idx=debug_idx)
                    except Exception as e:
                        print(f"\x1b[91m[Debug] Generation failed: {e}\x1b[0m")
                        buf.write(f"\n[Debug] Generation failed: {e}\n")
                    tb_chunks.append(_strip_ansi(buf.getvalue()))

                # Exactly one text event per step.
                if tb_chunks:
                    full = "".join(tb_chunks).strip()
                    max_chars = 30_000
                    if len(full) > max_chars:
                        full = full[:max_chars] + f"\n…(+{len(full) - max_chars} chars truncated)"
                    if full:
                        _log_text("debug/step", full, global_step)

            # Save checkpoint
            if config.save_steps > 0 and global_step % config.save_steps == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, f"step_{global_step}")
                model.save_checkpoint(checkpoint_path)

    # Save final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "final")
    model.save_checkpoint(final_path)

    if writer:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    print(f"\n[Train Stage 1] Completed {global_step} optimizer step(s)")
