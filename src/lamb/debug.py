import contextlib
import re
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional

if TYPE_CHECKING:
    from lamb.clara_model import ClaraModel
    from lamb.config import ClaraConfig
    from lamb.model import LaMBModel


def _decode_to_str(tokenizer: Any, ids: Any) -> str:
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    if isinstance(decoded, list):
        return " ".join(decoded)
    return str(decoded)


def _ansi_fg_rgb(r: int, g: int, b: int) -> str:
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"\x1b[38;2;{r};{g};{b}m"


def _ansi_reset() -> str:
    return "\x1b[0m"


def _score_to_rgb(score: float) -> tuple[int, int, int]:
    score = float(score)
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    # Color = score*green + (1-score)*red.
    # score=0 -> red, score=1 -> green.
    r = round(255 * (1.0 - score))
    g = round(255 * score)
    b = 0
    return r, g, b


def _preview_text(s: str, *, max_chars: int = 600) -> str:
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"…(+{len(s) - max_chars} chars)"


def _compact_mem_tokens_in_prompt(prompt_text: str) -> str:
    """Compact `<mem_0><mem_1>...` runs to `<mem_...>*N` for debug printing."""

    def repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        count = len(re.findall(r"<mem_\d+>", inner))
        return f"<background>\n<mem_...>*{count}\n</background>"

    # Only operate on the <background> ... </background> block to avoid accidental
    # changes elsewhere.
    return re.sub(
        r"<background>\n(.*?)\n</background>",
        repl,
        prompt_text,
        flags=re.DOTALL,
    )


def _normalize_for_contains(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _append_table(lines: list[str], title: str, rows: list[tuple[str, str]]) -> None:
    """Append a simple aligned key/value table into the log buffer."""

    if not rows:
        return

    lines.append(title)
    pad = max(len(k) for k, _ in rows)
    for k, v in rows:
        lines.append(f"  {k.ljust(pad)} : {v}")


def _answer_in_docs(answer: str, docs: list[str]) -> tuple[bool, int | None]:
    """Best-effort check whether the literal answer appears in any doc.

    This is intentionally conservative for very short answers (e.g. "yes").
    """

    ans = _normalize_for_contains(answer)
    if not ans or len(ans) < 4:
        return False, None

    for i, d in enumerate(docs):
        if ans in _normalize_for_contains(d):
            return True, i
    return False, None


@torch.no_grad()
def debug_reproduce_training(
    model: "LaMBModel",
    *,
    txt: str,
    max_print_tokens: int = 32,
    verbose: bool = False,
    stats_out: Any = None,
) -> bool:
    log: list[str] = []
    tokenizer = model.tokenizer
    splitter = "<|im_start|>assistant\n"
    if splitter not in txt:
        splitter = "assistant\n"

    if splitter in txt:
        ctx_raw, tgt_str = txt.split(splitter, 1)
    else:
        mid = len(txt) // 2
        ctx_raw, tgt_str = txt[:mid], txt[mid:]

    ctx_with_splitter = ctx_raw + splitter
    ctx_ids = tokenizer.encode(ctx_with_splitter, add_special_tokens=False)
    tgt_ids = tokenizer.encode(tgt_str, add_special_tokens=False)

    ctx_t = torch.tensor([ctx_ids], device=model.config.device)
    ctx_mask = (ctx_t != tokenizer.pad_token_id).long()
    layer_latents = model.compress(ctx_t, ctx_mask)
    past = model.bridge(layer_latents, rotary_module=model.rotary_emb)

    if model.has_compressor:
        model.base_model.disable_adapters()

    tgt_t = torch.tensor([tgt_ids], device=model.config.device)
    out = model.base_model(input_ids=tgt_t, past_key_values=past, use_cache=False)
    logits = out.logits[:, :-1, :]
    pred_next = logits.argmax(dim=-1)[0].tolist()
    gold_next = tgt_ids[1:]

    # Probability that the student assigns to the *gold* next token at each position.
    # These correspond to the same tokens used by CE loss (t1..).
    gold_scores: list[float] = []
    gold_ranks: list[int] = []
    if gold_next:
        logits0 = logits[0].float()
        probs0 = torch.softmax(logits0, dim=-1)
        gold_idx = torch.tensor(gold_next, device=probs0.device, dtype=torch.long)
        gold_scores_t = probs0.gather(-1, gold_idx.unsqueeze(-1)).squeeze(-1)
        gold_scores = gold_scores_t.detach().cpu().tolist()

        # Rank (1 = best) of the gold token under the student's distribution.
        gold_logits_t = logits0.gather(-1, gold_idx.unsqueeze(-1)).squeeze(-1)
        gold_ranks_t = (logits0 > gold_logits_t.unsqueeze(-1)).sum(dim=-1) + 1
        gold_ranks = gold_ranks_t.detach().cpu().tolist()

    compare_n = min(len(gold_next), len(pred_next), int(max_print_tokens))
    matches = [
        int(pred_next[i] == gold_next[i]) for i in range(min(len(gold_next), len(pred_next)))
    ]
    acc = (sum(matches) / len(matches)) if matches else 0.0

    lm_loss = functional.cross_entropy(
        logits.view(-1, logits.size(-1)), torch.tensor(gold_next, device=logits.device)
    ).item()
    if verbose:
        log.append("================ TRAINING-REPRO DEBUG ================")
        log.append(f"[Dbg] Context chars: {len(ctx_with_splitter)}, target chars: {len(tgt_str)}")
        log.append(f"[Dbg] Context tokens: {len(ctx_ids)}, target tokens: {len(tgt_ids)}")
        log.append(f"[Dbg] Context head: {ctx_with_splitter[:120].replace(chr(10), '↩')}...")
        log.append(f"[Dbg] Target head: {tgt_str[:120].replace(chr(10), '↩')}...")
        log.append("")

    summary_rows: list[tuple[str, str]] = [
        ("acc", f"{acc * 100:.2f}% ({sum(matches)}/{len(matches)})"),
        ("lm_loss", f"{lm_loss:.6f}"),
    ]
    if tgt_ids:
        summary_rows.append(("t0", f"{tgt_ids[0]} -> {_decode_to_str(tokenizer, [tgt_ids[0]])}"))
    _append_table(log, "[Dbg] Teacher-forced metrics", summary_rows)
    log.append("")

    if tgt_ids and gold_next and gold_scores:
        history_ids = [*ctx_ids, tgt_ids[0]]
        history_text = _decode_to_str(tokenizer, history_ids)

        n = min(int(max_print_tokens), len(gold_next), len(gold_scores))
        colored_parts: list[str] = []
        for i in range(n):
            tok_id = gold_next[i]
            tok_txt = _decode_to_str(tokenizer, [tok_id])
            r, g, b = _score_to_rgb(float(gold_scores[i]))
            colored_parts.append(f"{_ansi_fg_rgb(r, g, b)}{tok_txt}{_ansi_reset()}")

        suffix = ""
        if n < len(gold_next):
            suffix = f"{_ansi_reset()}…(+{len(gold_next) - n} tokens)"

        log.append(
            "[Dbg] Gold target colored by student P(gold) (red=0, green=1). "
            "Note: t0 (often '<think>') is unscored; coloring starts at t1 (CE shift)."
        )
        log.append(f"{history_text}{''.join(colored_parts)}{suffix}")

        if verbose and gold_ranks:
            log.append("[Dbg] Token scores (first positions):")
            for i in range(n):
                g_id = gold_next[i]
                g_s = _decode_to_str(tokenizer, [g_id]).replace("\n", "↩")
                p_id = pred_next[i] if i < len(pred_next) else -1
                p_s = _decode_to_str(tokenizer, [p_id]).replace("\n", "↩") if p_id >= 0 else ""
                score = float(gold_scores[i])
                rank = int(gold_ranks[i]) if i < len(gold_ranks) else -1
                log.append(
                    f"  {i:02d}: P(gold)={score:7.4f} rank={rank:6d} | gold:{g_s!r} pred:{p_s!r}"
                )

        log.append("")

    if stats_out is not None:
        with contextlib.suppress(Exception):
            stats_out["teacher_forced_acc"] = float(acc)
            stats_out["lm_loss"] = float(lm_loss)
            stats_out["teacher_forced_matches"] = int(sum(matches))
            stats_out["teacher_forced_total"] = len(matches)

    if verbose:
        log.append("[Dbg] Token-by-token (gold_next vs pred_next):")
        for i in range(compare_n):
            g = gold_next[i]
            p = pred_next[i]
            g_s = _decode_to_str(tokenizer, [g]).replace("\n", "↩")
            p_s = _decode_to_str(tokenizer, [p]).replace("\n", "↩")
            ok = "=" if g == p else "≠"
            log.append(f"  {i:02d}: {g:6d} {ok} {p:6d} | gold:{g_s!r} pred:{p_s!r}")
        log.append("")

    greedy_matches = False
    if len(tgt_ids) >= 1:
        seed = tgt_ids[0]
        max_new = max(1, 2 * len(tgt_ids))
        gen_ids = [seed]
        next_in = torch.tensor([[seed]], device=model.config.device)
        gen_past = past

        stop_ids = set()
        for tok in ["<|im_end|>", "</s>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != tokenizer.unk_token_id and tid is not None:
                stop_ids.add(int(tid))
        if tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))

        for _ in range(max_new):
            o = model.base_model(input_ids=next_in, past_key_values=gen_past, use_cache=True)
            gen_past = o.past_key_values
            nxt = int(o.logits[:, -1, :].argmax(dim=-1).item())
            gen_ids.append(nxt)
            if nxt in stop_ids:
                break
            next_in = torch.tensor([[nxt]], device=model.config.device)

        full_match = gen_ids[: len(tgt_ids)] == tgt_ids
        greedy_matches = bool(full_match)

        if verbose:
            gold_preview = _decode_to_str(tokenizer, tgt_ids[: min(len(tgt_ids), 64)])
            gen_preview = _decode_to_str(tokenizer, gen_ids[: min(len(gen_ids), 64)])
            log.append(f"[Dbg] Gold decode (head): {gold_preview[:300].replace(chr(10), '↩')}")
            log.append(
                f"[Dbg] Greedy decode (seeded) head: {gen_preview[:300].replace(chr(10), '↩')}"
            )
            log.append(
                f"[Dbg] Greedy seeded full-match (prefix length {len(tgt_ids)}): {full_match}"
            )
            log.append("")

    if verbose and tgt_ids:
        recon = [tgt_ids[0], *pred_next[: len(gold_next)]]
        recon_txt = _decode_to_str(tokenizer, recon)
        log.append(
            f"[Dbg] Teacher-forced recon decode (head): {recon_txt[:300].replace(chr(10), '↩')}"
        )
        log.append("")

    success = (acc >= 0.999) and (lm_loss < 0.1) and (len(gold_next) > 0)

    if stats_out is not None:
        with contextlib.suppress(Exception):
            stats_out["greedy_matches"] = bool(greedy_matches)

    log.append(f"[Dbg] Teacher-forced SUCCESS: {success} (acc={acc:.4f}, lm_loss={lm_loss:.6f})")
    log.append(f"[Dbg] Greedy match: {greedy_matches}")
    if verbose:
        log.append("======================================================\n")

    print("\n".join(log))

    return bool(success)


@torch.no_grad()
def debug_reproduce_clara(
    model: "ClaraModel",
    batch: dict[str, Any],
    config: "ClaraConfig",
    sample_idx: int = 0,
    max_print_tokens: int = 100,
) -> None:
    """Debug CLaRa training with color-coded token-by-token output.

    Shows gold answer tokens colored by model's probability (red=low, green=high).
    """
    log: list[str] = []
    was_training = model.training
    model.eval()
    tokenizer = model.tokenizer

    # Get one sample from batch
    num_docs_per_sample = batch.get("num_docs_per_sample")
    if not isinstance(num_docs_per_sample, list) or not num_docs_per_sample:
        num_docs_per_sample = [1] * int(batch["dec_input_ids"].size(0))

    doc_offset = sum(int(n) for n in num_docs_per_sample[:sample_idx])
    num_docs = int(num_docs_per_sample[sample_idx])
    doc_input_ids = batch["doc_input_ids"][doc_offset : doc_offset + num_docs].to(config.device)
    doc_attention_mask = batch["doc_attention_mask"][doc_offset : doc_offset + num_docs].to(
        config.device
    )
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

    # Compute accuracy only on answer tokens (where labels != -100)
    # labels are aligned with input_ids, but we're predicting next tokens
    # So we need to check labels[1:] against our predictions
    valid_mask = labels[0][1:] != -100  # Skip first position
    if valid_mask.any():
        valid_indices = valid_mask.nonzero(as_tuple=True)[0].tolist()
        matches = [int(pred_next[i] == gold_next[i]) for i in valid_indices if i < len(pred_next)]
        acc = (sum(matches) / len(matches)) if matches else 0.0
    else:
        matches = []
        acc = 0.0

    # Compute loss on answer tokens only
    valid_mask = labels[0] != -100
    if valid_mask.any():
        answer_logits = logits[0][valid_mask[1:]]  # Skip first token, match with labels
        answer_labels = labels[0][valid_mask]
        ce_loss = torch.nn.functional.cross_entropy(answer_logits, answer_labels).item()
    else:
        ce_loss = 0.0

    log.append("\n" + "=" * 80)
    log.append(f"[Debug] Sample {batch['indices'][sample_idx]}")
    log.append("=" * 80)

    questions = batch.get("questions")
    if isinstance(questions, list) and sample_idx < len(questions):
        q = str(questions[sample_idx]).strip()
        if q:
            log.append(f"[Debug] Question (gold): {_preview_text(q, max_chars=400)}")

    answers = batch.get("answers")
    gold_answer = ""
    if isinstance(answers, list) and sample_idx < len(answers):
        gold_answer = str(answers[sample_idx]).strip()
        if gold_answer:
            log.append(f"[Debug] Answer (gold): {_preview_text(gold_answer, max_chars=300)}")

    docs_per_sample = batch.get("docs_per_sample")
    if isinstance(docs_per_sample, list) and sample_idx < len(docs_per_sample):
        docs = docs_per_sample[sample_idx]
        if isinstance(docs, list) and docs:
            log.append("[Debug] Background (raw docs):")
            for i, d in enumerate(docs):
                d_str = _preview_text(str(d).strip())
                log.append(f"  [Doc {i}] {d_str}")
            log.append("")

            if gold_answer:
                found, doc_idx = _answer_in_docs(gold_answer, [str(x) for x in docs])
                if found:
                    log.append(f"[Debug] Answer literal found in docs: True (doc={doc_idx})\n")
                else:
                    log.append("[Debug] Answer literal found in docs: False\n")

    _append_table(
        log,
        "[Debug] Answer metrics",
        [
            ("token_acc", f"{acc * 100:.2f}% ({sum(matches)}/{len(matches)})"),
            ("ce_loss", f"{ce_loss:.6f}"),
        ],
    )

    # Color-coded output: show prompt + answer tokens colored by probability
    # Find where answer starts (first non -100 label)
    label_mask = labels[0] != -100
    if label_mask.any():
        answer_start_idx = label_mask.nonzero(as_tuple=True)[0][0].item()

        # Prompt (uncolored)
        prompt_ids = gold_ids_list[:answer_start_idx]
        prompt_text = _decode_to_str(tokenizer, prompt_ids)
        prompt_text = _compact_mem_tokens_in_prompt(prompt_text)

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

        log.append(
            "\n[Debug] Answer colored by P(gold token) - red=low confidence, green=high confidence:\n"
            f"{prompt_text}{''.join(colored_parts)}{suffix}\n"
        )

    log.append("=" * 80 + "\n")

    print("\n".join(log))

    if was_training:
        model.train()


def print_decode(tokenizer: Any, x: torch.Tensor) -> None:
    """Debug-print a token sequence using the provided tokenizer.

    Important: callers must pass the *same tokenizer instance* used to build the
    sequence. This is required for custom tokens like `<mem_*>` to be visible.
    """

    # If batched, show the first example.
    if x.ndim == 2:
        x = x[0]

    ids = x.detach().to("cpu").tolist()
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    token_view = " ".join(f"{tok}" for tok, tid in zip(tokens, ids, strict=False))

    print(f"\x1b[94m[Debug Decode] {decoded}\x1b[0m")
    print(f"\x1b[90m[Debug Tokens] {token_view}\x1b[0m")
