from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from lamb.model import LaMBModel


@torch.no_grad()
def generate_student(  # noqa: PLR0912,PLR0915
    model: "LaMBModel",
    *,
    en_text: str,
    expected_vi: Optional[str] = None,
    system_prompt: str = "You are a translator from English to Vietnamese",
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    model.eval()
    tokenizer = model.tokenizer

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": en_text},
        {"role": "assistant", "content": expected_vi or ""},
    ]

    full_txt = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(full_txt, str):
        raise TypeError("Tokenizer apply_chat_template did not return a string; set tokenize=False")

    splitter = "<|im_start|>assistant\n"
    if splitter not in full_txt:
        splitter = "assistant\n"
    if splitter not in full_txt:
        raise RuntimeError("Could not find an assistant splitter in the chat template output.")

    ctx_str, tgt_str = full_txt.split(splitter, 1)

    if max_new_tokens is None:
        tgt_len = len(tokenizer.encode(tgt_str, add_special_tokens=False))
        max_new_tokens = max(1, int(tgt_len) * 2)
        print(f"[Gen] Target length: {tgt_len} tokens, max_new_tokens: {max_new_tokens}")

    ctx_with_splitter = ctx_str + splitter
    context_ids = tokenizer.encode(ctx_with_splitter, add_special_tokens=False)
    context_ids_t = torch.tensor([context_ids], device=model.config.device)
    context_mask = (context_ids_t != tokenizer.pad_token_id).long()

    target_ids = tokenizer.encode(tgt_str, add_special_tokens=False)
    print(f"\n[Gen] Context: {ctx_str[:100]}... |\n\n---\nTarget: {tgt_str[:100]}...")
    print(f"[Gen] Context tokens: {len(context_ids)}, Target tokens: {len(target_ids)}")
    print(f"[Gen] Target token IDs: {target_ids}")
    print(f"[Gen] Target decoded: {tokenizer.decode(target_ids)}")

    layer_latents = model.compress(context_ids_t, context_mask)
    past_key_values = model.bridge(layer_latents, rotary_module=model.rotary_emb)

    model.base_model.disable_adapters()

    if len(target_ids) == 0:
        raise RuntimeError("Target encoded to 0 tokens; cannot generate.")
    seed_id = int(target_ids[0])
    input_ids = torch.tensor([[seed_id]], device=model.config.device)

    generated_ids: list[int] = []

    def _sample_next(logits: torch.Tensor) -> int:
        if not do_sample or temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        logits_local = logits / float(temperature)
        probs = torch.softmax(logits_local, dim=-1)

        if top_p is not None and 0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            keep = cdf <= top_p
            keep[..., 0] = True
            filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            next_in_sorted = torch.multinomial(filtered, num_samples=1)
            next_token = sorted_idx.gather(-1, next_in_sorted)
            return int(next_token.item())

        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    stop_token_ids = set()
    for tok in ["<|im_end|>", "</s>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id and tid is not None:
            stop_token_ids.add(int(tid))
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(int(tokenizer.eos_token_id))

    next_input = input_ids
    for i in range(int(max_new_tokens)):
        out = model.base_model(
            input_ids=next_input, past_key_values=past_key_values, use_cache=True
        )
        past_key_values = out.past_key_values
        next_logits = out.logits[:, -1, :].float()
        next_id = _sample_next(next_logits)
        if next_id in stop_token_ids:
            print(f"[Gen] Stopped at token {i}/{max_new_tokens}: hit stop token {next_id}")
            break
        generated_ids.append(next_id)
        next_input = torch.tensor([[next_id]], device=model.config.device)

    print(f"[Gen] Generated {len(generated_ids)} tokens: {generated_ids[:20]}...")

    decoded = tokenizer.decode([seed_id, *generated_ids], skip_special_tokens=True)
    if isinstance(decoded, list):
        decoded = " ".join(decoded)
    generated_text = str(decoded).strip()

    expected_text = tgt_str
    if "<|im_end|>" in expected_text:
        expected_text = expected_text.split("<|im_end|>", 1)[0]
    expected_text = expected_text.strip()

    print("\n=== Debug Generation (EN→VI) ===")
    print(f"EN: {en_text[:500]}" + ("..." if len(en_text) > 500 else ""))
    if expected_text:
        print("\n[Expected VI]")
        print(expected_text[:2000] + ("..." if len(expected_text) > 2000 else ""))
    print("\n[Generated VI]")
    print(generated_text[:2000] + ("..." if len(generated_text) > 2000 else ""))
    print("\n==============================\n")

    return generated_text
