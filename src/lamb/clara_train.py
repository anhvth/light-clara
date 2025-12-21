"""CLaRa Training Loop - Stage 1 (Compression Pretraining)."""

import os
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lamb.clara_collate import make_stage1_collate_fn
from lamb.clara_model import ClaraModel
from lamb.config import ClaraConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore


def _ansi_fg_rgb(r: int, g: int, b: int) -> str:
    """Generate ANSI escape code for RGB foreground color."""
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"\x1b[38;2;{r};{g};{b}m"


def _ansi_reset() -> str:
    """ANSI reset code."""
    return "\x1b[0m"


def _score_to_rgb(score: float) -> tuple[int, int, int]:
    """Convert probability score to RGB color (red=0, green=1)."""
    score = float(score)
    score = max(0.0, min(1.0, score))
    # Red for low scores, green for high scores
    r = round(255 * (1.0 - score))
    g = round(255 * score)
    b = 0
    return r, g, b


def _decode_to_str(tokenizer: Any, ids: Any) -> str:
    """Decode token IDs to string."""
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    if isinstance(decoded, list):
        return " ".join(decoded)
    return str(decoded)


@torch.no_grad()
def debug_reproduce_clara(
    model: ClaraModel,
    batch: dict[str, Any],
    config: ClaraConfig,
    sample_idx: int = 0,
    max_print_tokens: int = 100,
) -> None:
    """Debug CLaRa training with color-coded token-by-token output.

    Shows gold answer tokens colored by model's probability (red=low, green=high).
    """
    was_training = model.training
    model.eval()
    tokenizer = model.tokenizer

    # Get one sample from batch
    doc_input_ids = batch["doc_input_ids"][sample_idx : sample_idx + 1].to(config.device)
    doc_attention_mask = batch["doc_attention_mask"][sample_idx : sample_idx + 1].to(config.device)
    dec_input_ids = batch["dec_input_ids"][sample_idx : sample_idx + 1].to(config.device)
    labels = batch["labels"][sample_idx : sample_idx + 1].to(config.device)

    # Compress documents
    memory_embeddings, _ = model.compress_documents(doc_input_ids, doc_attention_mask)
    memory_embeddings_flat = memory_embeddings.view(1, -1, memory_embeddings.size(-1))

    # Forward through decoder
    outputs = model.forward_with_memory(
        input_ids=dec_input_ids,
        attention_mask=torch.ones_like(dec_input_ids),
        memory_embeddings=memory_embeddings_flat,
        labels=None,  # Don't compute loss, just get logits
    )

    logits = outputs["logits"][:, :-1, :]  # Shift for next-token prediction
    gold_ids_list = dec_input_ids[0].tolist()
    gold_next = gold_ids_list[1:]  # Next tokens to predict

    # Get model predictions
    pred_next = logits.argmax(dim=-1)[0].tolist()

    # Compute probabilities for gold tokens
    gold_scores: list[float] = []
    if gold_next:
        logits0 = logits[0].float()
        probs0 = torch.softmax(logits0, dim=-1)
        gold_idx = torch.tensor(gold_next, device=probs0.device, dtype=torch.long)
        gold_scores_t = probs0.gather(-1, gold_idx.unsqueeze(-1)).squeeze(-1)
        gold_scores = gold_scores_t.cpu().tolist()

    # Compute accuracy
    matches = [
        int(pred_next[i] == gold_next[i]) for i in range(min(len(gold_next), len(pred_next)))
    ]
    acc = (sum(matches) / len(matches)) if matches else 0.0

    # Compute loss on answer tokens only
    valid_mask = labels[0] != -100
    if valid_mask.any():
        answer_logits = logits[0][valid_mask[1:]]  # Skip first token, match with labels
        answer_labels = labels[0][valid_mask]
        ce_loss = torch.nn.functional.cross_entropy(answer_logits, answer_labels).item()
    else:
        ce_loss = 0.0

    print("\n" + "=" * 80)
    print(f"[Debug] Sample {sample_idx}")
    print("=" * 80)
    print(f"[Debug] Answer token accuracy: {acc * 100:.2f}% ({sum(matches)}/{len(matches)})")
    print(f"[Debug] Answer CE loss: {ce_loss:.6f}")

    # Color-coded output: show prompt + answer tokens colored by probability
    # Find where answer starts (first non -100 label)
    label_mask = labels[0] != -100
    if label_mask.any():
        answer_start_idx = label_mask.nonzero(as_tuple=True)[0][0].item()

        # Prompt (uncolored)
        prompt_ids = gold_ids_list[:answer_start_idx]
        prompt_text = _decode_to_str(tokenizer, prompt_ids)

        # Answer tokens (colored by model probability)
        answer_ids = gold_ids_list[answer_start_idx:]
        # Adjust gold_scores index (it starts from position 0 in shifted logits)
        # We need scores for tokens starting at answer_start_idx
        answer_scores_start = answer_start_idx - 1  # Because logits are shifted by 1

        colored_parts: list[str] = []
        n = min(len(answer_ids), max_print_tokens, len(gold_scores) - answer_scores_start)
        for i in range(n):
            tok_id = answer_ids[i]
            tok_txt = _decode_to_str(tokenizer, [tok_id])
            score_idx = answer_scores_start + i
            if 0 <= score_idx < len(gold_scores):
                score = gold_scores[score_idx]
                r, g, b = _score_to_rgb(score)
                colored_parts.append(f"{_ansi_fg_rgb(r, g, b)}{tok_txt}{_ansi_reset()}")
            else:
                colored_parts.append(tok_txt)

        suffix = ""
        if n < len(answer_ids):
            suffix = f"{_ansi_reset()}…(+{len(answer_ids) - n} tokens)"

        print(
            "\n[Debug] Answer colored by P(gold token) - red=low confidence, green=high confidence:\n"
            f"{prompt_text}{''.join(colored_parts)}{suffix}\n"
        )

    print("=" * 80 + "\n")

    if was_training:
        model.train()


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
    if config.debug_mode:
        print(
            f"\x1b[92m[Debug Mode] Enabled - will show color-coded tokens every {config.debug_every_steps} steps\x1b[0m"
        )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    # Setup data loader
    collate_fn = make_stage1_collate_fn(
        tokenizer=model.tokenizer,
        doc_max_length=config.doc_max_length,
        dec_max_length=config.max_seq_len,
        generation_top_k=config.generation_top_k,
        num_mem_tokens=config.compress_rate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Setup TensorBoard
    writer = None
    if config.tensorboard and SummaryWriter is not None:
        run_dir = os.path.join(config.tensorboard_logdir, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)
        print(f"[TensorBoard] Logging to {run_dir}")

    # Training loop
    model.train()
    global_step = 0
    optimizer.zero_grad()

    # Metrics tracking
    running_qa_loss = 0.0
    running_paraphrase_loss = 0.0
    running_mse_loss = 0.0
    running_total_loss = 0.0
    log_every = config.tensorboard_every_steps

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

        # Reshape memory embeddings to [batch, num_docs * compress_rate, hidden_size]
        # Currently: [batch * total_docs, compress_rate, hidden_size]
        # Need to split by num_docs_per_sample
        batch_memory_embeddings = []
        doc_offset = 0
        for num_docs in num_docs_per_sample:
            sample_mem = memory_embeddings[doc_offset : doc_offset + num_docs]
            # Flatten: [num_docs, compress_rate, hidden_size] -> [num_docs * compress_rate, hidden_size]
            sample_mem_flat = sample_mem.view(-1, sample_mem.size(-1))
            batch_memory_embeddings.append(sample_mem_flat)
            doc_offset += num_docs

        # Pad to same length
        max_mem_tokens = max(m.size(0) for m in batch_memory_embeddings)
        padded_memory_embeddings = []
        for mem in batch_memory_embeddings:
            if mem.size(0) < max_mem_tokens:
                pad_size = max_mem_tokens - mem.size(0)
                pad = torch.zeros(pad_size, mem.size(1), dtype=mem.dtype, device=mem.device)
                mem = torch.cat([mem, pad], dim=0)
            padded_memory_embeddings.append(mem)

        batch_memory_embeddings_tensor = torch.stack(padded_memory_embeddings)

        # Forward through decoder with memory embeddings
        outputs = model.forward_with_memory(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            memory_embeddings=batch_memory_embeddings_tensor,
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

        # Split into QA and paraphrase
        if num_qa > 0 and num_paraphrase > 0:
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
            mse_loss = (
                model.compressor.compute_mse_loss(encoder_hidden_states, memory_embeddings)
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

            # Log to tensorboard
            if writer and global_step % log_every == 0:
                avg_qa = running_qa_loss / log_every
                avg_para = running_paraphrase_loss / log_every
                avg_mse = running_mse_loss / log_every
                avg_total = running_total_loss / log_every

                writer.add_scalar("train/qa_loss", avg_qa, global_step)
                writer.add_scalar("train/paraphrase_loss", avg_para, global_step)
                writer.add_scalar("train/mse_loss", avg_mse, global_step)
                writer.add_scalar("train/total_loss", avg_total, global_step)
                writer.add_scalar("train/learning_rate", config.learning_rate, global_step)

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
            import ipdb

            ipdb.set_trace()
            if config.debug_mode and global_step % config.debug_every_steps == 0:
                for debug_idx in range(min(config.debug_num_samples, batch_size)):
                    try:
                        debug_reproduce_clara(model, batch, config, sample_idx=debug_idx)
                    except Exception as e:
                        print(f"\x1b[91m[Debug] Generation failed: {e}\x1b[0m")

            # Save checkpoint
            if config.save_steps > 0 and global_step % config.save_steps == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, f"step_{global_step}")
                model.save_checkpoint(checkpoint_path)

    # Save final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "final")
    model.save_checkpoint(final_path)

    if writer:
        writer.close()

    print(f"\n[Train Stage 1] Completed {global_step} steps")
