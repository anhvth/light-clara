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


@torch.no_grad()
def debug_reproduce_training(  # noqa: PLR0912,PLR0915
    model: "LaMBModel",
    *,
    txt: str,
    max_print_tokens: int = 32,
    verbose: bool = False,
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

    model.base_model.disable_adapters()

    tgt_t = torch.tensor([tgt_ids], device=model.config.device)
    out = model.base_model(input_ids=tgt_t, past_key_values=past, use_cache=False)
    logits = out.logits[:, :-1, :]
    pred_next = logits.argmax(dim=-1)[0].tolist()
    gold_next = tgt_ids[1:]

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

    print(f"[Dbg] Teacher-forced SUCCESS: {success} (acc={acc:.4f}, lm_loss={lm_loss:.6f})")
    print(f"[Dbg] Greedy match: {greedy_matches}")
    if verbose:
        print("======================================================\n")

    if was_training:
        model.train()
    return bool(success)
