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


def build_memory_token_string(
    num_docs: int,
    num_mem_tokens_per_doc: int = 32,
    sep_token: str = "",
) -> str:
    """Build string with memory token placeholders.

    Args:
        num_docs: Number of documents
        num_mem_tokens_per_doc: Memory tokens per document (default: 32)
        sep_token: Separator to insert between documents (e.g., "<SEP>")

    Returns:
        String like "<mem_0>...<mem_31><SEP><mem_0>...<mem_31><SEP>" for 2 docs (with sep)
        or "<mem_0>...<mem_31><mem_0>...<mem_31>" for 2 docs (without sep).

        Important: We intentionally *reuse* the same `<mem_i>` tokens for each
        document. The decoder sees repeated placeholders, and we replace them in
        left-to-right order with the flattened `memory_embeddings`.
    """
    doc_tokens = []
    for _doc_idx in range(num_docs):
        mem_tokens = [f"<mem_{mem_idx}>" for mem_idx in range(num_mem_tokens_per_doc)]
        doc_tokens.append("".join(mem_tokens))

    if sep_token:
        # Add separator after each document
        return sep_token.join(doc_tokens) + sep_token
    else:
        return "".join(doc_tokens)


def build_qa_prompt(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
    num_docs: int,
    num_mem_tokens: int = 32,
    system_prompt: str = "Answer the question using the provided background.",
    sep_token: str = "",
) -> tuple[str, int]:
    """Build QA prompt with memory token placeholders.

    Returns:
        (prompt_text, prompt_length) where prompt_length is tokens before answer
    """
    mem_tokens_str = build_memory_token_string(num_docs, num_mem_tokens, sep_token=sep_token)

    user_content = f"<background>\n{mem_tokens_str}\n</background>\n\nQuestion: {question}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert isinstance(full_text, str), f"Expected str, got {type(full_text)}"

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
    sep_token: str = "",
) -> tuple[str, int]:
    """Build paraphrase prompt with memory token placeholders.

    Returns:
        (prompt_text, prompt_length) where prompt_length is tokens before paraphrase
    """
    mem_tokens_str = build_memory_token_string(1, num_mem_tokens, sep_token=sep_token)

    # Randomly select instruction template
    instruction_template = random.choice(PARAPHRASE_INSTRUCTIONS)
    instruction = instruction_template.format(docs=mem_tokens_str)

    # Simple format: instruction + paraphrase
    full_text = f"{instruction}\n{paraphrase_text}"

    # Prompt length is the instruction part
    instruction_tokens = tokenizer.encode(instruction + "\n", add_special_tokens=False)
    prompt_length = len(instruction_tokens)

    return full_text, prompt_length


def _original_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    user_prompt: str,
    assistant_content: str,
) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Chat template for prompt length (no assistant content)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_length = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    messages.append({"role": "assistant", "content": assistant_content})
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return full_text, prompt_length


def build_original_qa_prompt(
    tokenizer: PreTrainedTokenizer,
    question: str | list[str],
    answer: str | list[str],
    num_docs: int,
    num_mem_tokens: int,
    sep_token: str = "<SEP>",
) -> tuple[str, int]:
    """Build original CLaRa QA prompt with support for multiple QA pairs.

    Args:
        question: Single question string or list of questions
        answer: Single answer string or list of answers
        num_docs: Number of documents
        num_mem_tokens: Memory tokens per document
        sep_token: Separator token between documents

    Returns:
        (full_text, prompt_length) where prompt_length excludes the answer(s)
    """
    mem_tokens_str = build_memory_token_string(num_docs, num_mem_tokens, sep_token=sep_token)
    system_prompt = (
        "You are a helpful assistant. Given a document, your task is to generate some single "
        "questions to cover all key information of the document and answer them sequentially."
    )
    user_prompt = f"Background:\n{mem_tokens_str}"

    # Handle multiple QA pairs (original format) or single pair (fallback)
    if isinstance(question, list) and isinstance(answer, list):
        # Multiple QA pairs - join them with newlines
        qa_lines = [f"Question: {q}\nAnswer: {a}" for q, a in zip(question, answer, strict=False)]
        qa_block = "\n".join(qa_lines)
    elif isinstance(question, list):
        # List of questions but single answer - use first question
        qa_block = f"Question: {question[0]}\nAnswer: {answer}"
    elif isinstance(answer, list):
        # Single question but list of answers - use first answer
        qa_block = f"Question: {question}\nAnswer: {answer[0]}"
    else:
        # Single question and answer
        qa_block = f"Question: {question}\nAnswer: {answer}"

    return _original_chat_prompt(tokenizer, system_prompt, user_prompt, qa_block)


def build_original_paraphrase_prompt(
    tokenizer: PreTrainedTokenizer,
    paraphrase_text: str,
    num_docs: int,
    num_mem_tokens: int,
    sep_token: str = "<SEP>",
) -> tuple[str, int]:
    mem_tokens_str = build_memory_token_string(num_docs, num_mem_tokens, sep_token=sep_token)
    system_prompt = (
        "You are a helpful assistant. Your task is follow the instructions to paraphrase the "
        "background information."
    )
    user_prompt = random.choice(PARAPHRASE_INSTRUCTIONS).format(docs=mem_tokens_str)
    return _original_chat_prompt(tokenizer, system_prompt, user_prompt, paraphrase_text)


def stage1_collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    doc_max_length: int = 256,
    dec_max_length: int = 1024,
    generation_top_k: int = 1,
    num_mem_tokens: int = 32,
    sep_token: str = "",
) -> dict[str, Any]:
    """Collate function for Stage 1 training (compression pretraining).

    Batch items should have:
        - data_type: "qa" or "paraphrase" (or "unknown" treated as QA)
        - question: str | list[str] (can be multiple questions for multi-QA)
        - docs: list[str]
        - answer: str | list[str] (can be multiple answers for multi-QA)

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
    docs_per_sample: list[list[str]] = []
    questions: list[str | list[str]] = []
    answers: list[str | list[str]] = []

    for item in batch:
        docs = item.get("docs", [])[:generation_top_k]
        if not docs:
            docs = [""]  # Empty doc placeholder
        all_docs.extend(docs)
        num_docs_per_sample.append(len(docs))
        docs_per_sample.append(list(docs))

        # Preserve list format if present
        q = item.get("question", "")
        a = item.get("answer", "")
        questions.append(q)
        answers.append(a)

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
    indices = []

    for item in batch:
        data_type = item.get("data_type", "unknown")
        question = item.get("question", "")
        answer = item.get("answer", "")
        idx = item.get("idx", 0)
        num_docs = min(len(item.get("docs", [])), generation_top_k)
        if num_docs == 0:
            num_docs = 1

        data_types.append(data_type)
        indices.append(idx)

        if data_type == "paraphrase":
            # Paraphrase format
            prompt_text, prompt_len = build_paraphrase_prompt(
                tokenizer, answer, num_mem_tokens, sep_token=sep_token
            )
        else:
            # QA format (default for "qa" and "unknown")
            prompt_text, prompt_len = build_qa_prompt(
                tokenizer, question, answer, num_docs, num_mem_tokens, sep_token=sep_token
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
    labels = dec_encodings["input_ids"].clone()  # type: ignore
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
        "docs_per_sample": docs_per_sample,
        "questions": questions,
        "answers": answers,
        "indices": indices,
    }


def stage1_collate_fn_original(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    doc_max_length: int = 256,
    dec_max_length: int = 1024,
    generation_top_k: int = 1,
    num_mem_tokens: int = 32,
    sep_token: str = "<SEP>",
) -> dict[str, Any]:
    """Collate function that mirrors Apple's original stage-1 formatting.

    The original pipeline assumes a fixed number of documents per sample
    (`generation_top_k`) and uses chat-template prompts with system/user roles.

    Supports multiple QA pairs per sample (questions/answers as lists).
    """

    docs_per_sample: list[list[str]] = []
    questions: list[str | list[str]] = []
    answers: list[str | list[str]] = []
    data_types: list[str] = []
    indices: list[int] = []

    for item in batch:
        docs = list(item.get("docs", []))
        if not docs:
            docs = [""]

        # Truncate/pad to the expected top-k count for compatibility with the
        # original CLaRa generator.
        if len(docs) > generation_top_k:
            docs = docs[:generation_top_k]
        elif len(docs) < generation_top_k:
            docs.extend([""] * (generation_top_k - len(docs)))

        docs_per_sample.append(docs)
        # Preserve list format if present
        questions.append(item.get("question", ""))
        answers.append(item.get("answer", ""))
        data_types.append(item.get("data_type", "qa"))
        indices.append(int(item.get("idx", 0)))

    flat_docs = [doc for docs in docs_per_sample for doc in docs]
    doc_encodings = tokenizer(
        flat_docs,
        max_length=doc_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,
    )

    decoder_texts: list[str] = []
    prompt_lengths: list[int] = []

    for question, answer, data_type in zip(questions, answers, data_types, strict=False):
        if data_type == "paraphrase":
            prompt_text, prompt_len = build_original_paraphrase_prompt(
                tokenizer,
                paraphrase_text=answer,
                num_docs=generation_top_k,
                num_mem_tokens=num_mem_tokens,
                sep_token=sep_token,
            )
        else:
            prompt_text, prompt_len = build_original_qa_prompt(
                tokenizer,
                question=question,
                answer=answer,
                num_docs=generation_top_k,
                num_mem_tokens=num_mem_tokens,
                sep_token=sep_token,
            )

        decoder_texts.append(prompt_text)
        prompt_lengths.append(prompt_len)

    dec_encodings = tokenizer(
        decoder_texts,
        max_length=dec_max_length,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=False,
    )

    labels = dec_encodings["input_ids"].clone()  # type: ignore
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100
    labels[dec_encodings["attention_mask"] == 0] = -100

    return {
        "doc_input_ids": doc_encodings["input_ids"],
        "doc_attention_mask": doc_encodings["attention_mask"],
        "dec_input_ids": dec_encodings["input_ids"],
        "dec_attention_mask": dec_encodings["attention_mask"],
        "labels": labels,
        "data_types": data_types,
        "num_docs_per_sample": [generation_top_k for _ in batch],
        "docs_per_sample": docs_per_sample,
        "questions": questions,
        "answers": answers,
        "indices": indices,
    }


def make_stage1_collate_fn(
    tokenizer: PreTrainedTokenizer,
    doc_max_length: int = 256,
    dec_max_length: int = 1024,
    generation_top_k: int = 1,
    num_mem_tokens: int = 32,
    use_clara_original: bool = False,
    use_sep_token: bool = False,
):
    """Create a collate function with fixed parameters."""

    def collate_fn(batch):
        sep = "<SEP>" if use_sep_token else ""
        if use_clara_original:
            return stage1_collate_fn_original(
                batch,
                tokenizer=tokenizer,
                doc_max_length=doc_max_length,
                dec_max_length=dec_max_length,
                generation_top_k=generation_top_k,
                num_mem_tokens=num_mem_tokens,
                sep_token=sep,
            )

        return stage1_collate_fn(
            batch,
            tokenizer=tokenizer,
            doc_max_length=doc_max_length,
            dec_max_length=dec_max_length,
            generation_top_k=generation_top_k,
            num_mem_tokens=num_mem_tokens,
            sep_token=sep,
        )

    return collate_fn
