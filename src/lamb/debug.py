import contextlib
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional

if TYPE_CHECKING:
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


@torch.no_grad()
def debug_reproduce_training(  # noqa: PLR0912,PLR0915
    model: "LaMBModel",
    *,
    txt: str,
    max_print_tokens: int = 32,
    verbose: bool = False,
    stats_out: Any = None,
) -> bool:
    tokenizer = model.tokenizer
    was_training = model.training
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

    if verbose:
        print("\n================ TRAINING-REPRO DEBUG ================")
        print(f"[Dbg] Context chars: {len(ctx_with_splitter)}, target chars: {len(tgt_str)}")
        print(f"[Dbg] Context tokens: {len(ctx_ids)}, target tokens: {len(tgt_ids)}")
        print(f"[Dbg] Context head: {ctx_with_splitter[:120].replace(chr(10), '↩')}...")
        print(f"[Dbg] Target head: {tgt_str[:120].replace(chr(10), '↩')}...")

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

    if verbose and tgt_ids:
        t0 = tgt_ids[0]
        t0_dec = _decode_to_str(tokenizer, [t0]).replace(chr(10), "↩")
        print(f"[Dbg] t0 (FIRST target token): {t0} -> {t0_dec!r}")
        print(
            "[Dbg] The table below compares predicted NEXT token vs gold NEXT token, so it starts at t1 (not t0)."
        )

    print(f"[Dbg] Teacher-forced next-token acc: {acc * 100:.2f}% ({sum(matches)}/{len(matches)})")
    print(f"[Dbg] LM loss (gold CE on t1..): {lm_loss:.6f}")

    # High-quality token debug: show the history (ctx + t0) uncolored, then the tokens we
    # compute CE on (t1..) colored by the student's probability assigned to that gold token.
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

        print(
            "[Dbg] Gold target colored by student P(gold) (red=0, green=1). "
            "Note: t0 (often '<think>') is unscored; coloring starts at t1 (CE shift).\n"
            f"{history_text}{''.join(colored_parts)}{suffix}"
        )

        if verbose and gold_ranks:
            print("[Dbg] Token scores (first positions):")
            for i in range(n):
                g_id = gold_next[i]
                g_s = _decode_to_str(tokenizer, [g_id]).replace("\n", "↩")
                p_id = pred_next[i] if i < len(pred_next) else -1
                p_s = _decode_to_str(tokenizer, [p_id]).replace("\n", "↩") if p_id >= 0 else ""
                score = float(gold_scores[i])
                rank = int(gold_ranks[i]) if i < len(gold_ranks) else -1
                print(f"  {i:02d}: P(gold)={score:7.4f} rank={rank:6d} | gold:{g_s!r} pred:{p_s!r}")

    if stats_out is not None:
        with contextlib.suppress(Exception):
            stats_out["teacher_forced_acc"] = float(acc)
            stats_out["lm_loss"] = float(lm_loss)
            stats_out["teacher_forced_matches"] = int(sum(matches))
            stats_out["teacher_forced_total"] = len(matches)

    if verbose:
        print("[Dbg] Token-by-token (gold_next vs pred_next):")
        for i in range(compare_n):
            g = gold_next[i]
            p = pred_next[i]
            g_s = _decode_to_str(tokenizer, [g]).replace("\n", "↩")
            p_s = _decode_to_str(tokenizer, [p]).replace("\n", "↩")
            ok = "=" if g == p else "≠"
            print(f"  {i:02d}: {g:6d} {ok} {p:6d} | gold:{g_s!r} pred:{p_s!r}")

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

        if verbose:
            gold_preview = _decode_to_str(tokenizer, tgt_ids[: min(len(tgt_ids), 64)])
            gen_preview = _decode_to_str(tokenizer, gen_ids[: min(len(gen_ids), 64)])
            print(f"[Dbg] Gold decode (head): {gold_preview[:300].replace(chr(10), '↩')}")
            print(f"[Dbg] Greedy decode (seeded) head: {gen_preview[:300].replace(chr(10), '↩')}")

        full_match = gen_ids[: len(tgt_ids)] == tgt_ids
        if verbose:
            print(f"[Dbg] Greedy seeded full-match (prefix length {len(tgt_ids)}): {full_match}")

    if verbose and tgt_ids:
        recon = [tgt_ids[0], *pred_next[: len(gold_next)]]
        recon_txt = _decode_to_str(tokenizer, recon)
        print(f"[Dbg] Teacher-forced recon decode (head): {recon_txt[:300].replace(chr(10), '↩')}")

    success = (acc >= 0.999) and (lm_loss < 0.1) and (len(gold_next) > 0)

    greedy_matches = False
    if len(tgt_ids) >= 1:
        greedy_matches = gen_ids[: len(tgt_ids)] == tgt_ids

    if stats_out is not None:
        with contextlib.suppress(Exception):
            stats_out["greedy_matches"] = bool(greedy_matches)

    print(f"[Dbg] Teacher-forced SUCCESS: {success} (acc={acc:.4f}, lm_loss={lm_loss:.6f})")
    print(f"[Dbg] Greedy match: {greedy_matches}")
    if verbose:
        print("======================================================\n")

    if was_training:
        model.train()
    return bool(success)
