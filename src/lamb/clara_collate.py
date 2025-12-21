"""Data collation functions for CLaRa training stages."""

import random
from typing import Any

from transformers import PreTrainedTokenizer

# Paraphrase instruction templates (from Apple's CLaRa)
PARAPHRASE_INSTRUCTIONS = [
    "Background: {docs} means the same as",
    "Background: {docs} Can you put the above sentences in your own terms?",
    "Background: {docs} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {docs}\n(2)",
    "Background: {docs} is a paraphrase of what?",
    "Background: {docs} Could you give me a different version of the background sentences above?",
    "In other words, background: {docs} is just another way of saying:",
    "You're getting across the same point whether you say background: {docs} or",
    "Background: {docs} After unpacking the ideas in the background information above, we got:",
    "Background: {docs} Please offer a restatement of the background sentences I've just read.",
    "Background: {docs}, which also means:",
    "Strip away the mystery, and you'll find background: {docs} is simply another rendition of:",
    "The essence of background: {docs} is captured again in the following statement:",
]


def build_memory_token_string(num_docs: int, num_mem_tokens_per_doc: int = 32) -> str:
    """Build string with memory token placeholders.

    Args:
        num_docs: Number of documents
        num_mem_tokens_per_doc: Memory tokens per document (default: 32)

    Returns:
        String like "<mem_0><mem_1>...<mem_31><mem_32>...<mem_63>" for 2 docs
    """
    tokens = []
    for doc_idx in range(num_docs):
        for mem_idx in range(num_mem_tokens_per_doc):
            global_idx = doc_idx * num_mem_tokens_per_doc + mem_idx
            tokens.append(f"<mem_{global_idx}>")
    return "".join(tokens)


def build_qa_prompt(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
    num_docs: int,
    num_mem_tokens: int = 32,
    system_prompt: str = "Answer the question using the provided background.",
) -> tuple[str, int]:
    """Build QA prompt with memory token placeholders.

    Returns:
        (prompt_text, prompt_length) where prompt_length is tokens before answer
    """
    mem_tokens_str = build_memory_token_string(num_docs, num_mem_tokens)

    user_content = f"<background>\n{mem_tokens_str}\n</background>\n\nQuestion: {question}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Compute prompt length (everything before answer)
    # Find the assistant marker
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker not in full_text:
        assistant_marker = "assistant\n"

    if assistant_marker in full_text:
        prompt_part = full_text.split(assistant_marker)[0] + assistant_marker
        prompt_tokens = tokenizer.encode(prompt_part, add_special_tokens=False)
        prompt_length = len(prompt_tokens)
    else:
        # Fallback: estimate
        prompt_length = len(tokenizer.encode(user_content, add_special_tokens=False)) + 10

    return full_text, prompt_length


def build_paraphrase_prompt(
    tokenizer: PreTrainedTokenizer,
    paraphrase_text: str,
    num_mem_tokens: int = 32,
) -> tuple[str, int]:
    """Build paraphrase prompt with memory token placeholders.

    Returns:
        (prompt_text, prompt_length) where prompt_length is tokens before paraphrase
    """
    mem_tokens_str = build_memory_token_string(1, num_mem_tokens)

    # Randomly select instruction template
    instruction_template = random.choice(PARAPHRASE_INSTRUCTIONS)
    instruction = instruction_template.format(docs=mem_tokens_str)

    # Simple format: instruction + paraphrase
    full_text = f"{instruction}\n{paraphrase_text}"

    # Prompt length is the instruction part
    instruction_tokens = tokenizer.encode(instruction + "\n", add_special_tokens=False)
    prompt_length = len(instruction_tokens)

    return full_text, prompt_length


def stage1_collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    doc_max_length: int = 256,
    dec_max_length: int = 1024,
    generation_top_k: int = 1,
    num_mem_tokens: int = 32,
) -> dict[str, Any]:
    """Collate function for Stage 1 training (compression pretraining).

    Batch items should have:
        - data_type: "qa" or "paraphrase" (or "unknown" treated as QA)
        - question: str
        - docs: list[str]
        - answer: str

    Returns:
        dict with:
            - doc_input_ids: [batch * num_docs, doc_seq_len]
            - doc_attention_mask: [batch * num_docs, doc_seq_len]
            - dec_input_ids: [batch, dec_seq_len]
            - dec_attention_mask: [batch, dec_seq_len]
            - labels: [batch, dec_seq_len]
            - data_types: list[str]
            - num_docs_per_sample: list[int]
    """
    len(batch)

    # Collect all documents across batch
    all_docs = []
    num_docs_per_sample = []

    for item in batch:
        docs = item.get("docs", [])[:generation_top_k]
        if not docs:
            docs = [""]  # Empty doc placeholder
        all_docs.extend(docs)
        num_docs_per_sample.append(len(docs))

    # Encode documents
    doc_encodings = tokenizer(
        all_docs,
        max_length=doc_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,
    )

    # Build decoder prompts
    decoder_texts = []
    prompt_lengths = []
    data_types = []

    for item in batch:
        data_type = item.get("data_type", "unknown")
        question = item.get("question", "")
        answer = item.get("answer", "")
        num_docs = min(len(item.get("docs", [])), generation_top_k)
        if num_docs == 0:
            num_docs = 1

        data_types.append(data_type)

        if data_type == "paraphrase":
            # Paraphrase format
            prompt_text, prompt_len = build_paraphrase_prompt(tokenizer, answer, num_mem_tokens)
        else:
            # QA format (default for "qa" and "unknown")
            prompt_text, prompt_len = build_qa_prompt(
                tokenizer, question, answer, num_docs, num_mem_tokens
            )

        decoder_texts.append(prompt_text)
        prompt_lengths.append(prompt_len)

    # Encode decoder inputs
    dec_encodings = tokenizer(
        decoder_texts,
        max_length=dec_max_length,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=False,
    )

    # Create labels (mask prompt tokens, only train on answer)
    labels = dec_encodings["input_ids"].clone()
    for i, prompt_len in enumerate(prompt_lengths):
        # Mask everything before the answer
        labels[i, :prompt_len] = -100

    # Also mask padding tokens
    labels[dec_encodings["attention_mask"] == 0] = -100

    return {
        "doc_input_ids": doc_encodings["input_ids"],
        "doc_attention_mask": doc_encodings["attention_mask"],
        "dec_input_ids": dec_encodings["input_ids"],
        "dec_attention_mask": dec_encodings["attention_mask"],
        "labels": labels,
        "data_types": data_types,
        "num_docs_per_sample": num_docs_per_sample,
    }


def make_stage1_collate_fn(
    tokenizer: PreTrainedTokenizer,
    doc_max_length: int = 256,
    dec_max_length: int = 1024,
    generation_top_k: int = 1,
    num_mem_tokens: int = 32,
):
    """Create a collate function with fixed parameters."""

    def collate_fn(batch):
        return stage1_collate_fn(
            batch,
            tokenizer=tokenizer,
            doc_max_length=doc_max_length,
            dec_max_length=dec_max_length,
            generation_top_k=generation_top_k,
            num_mem_tokens=num_mem_tokens,
        )

    return collate_fn
