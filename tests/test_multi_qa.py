"""Test multi-QA functionality for CLaRa original format."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import only what we need to avoid triggering config table
from transformers import AutoTokenizer

from lamb.clara_collate import build_original_qa_prompt
from lamb.clara_data import normalize_clara_record


def build_memory_token_string(num_docs: int, num_mem_tokens: int, sep_token: str = "") -> str:
    """Build memory token string (copied from clara_collate to avoid imports)."""
    doc_tokens = []
    for _doc_idx in range(num_docs):
        mem_tokens = [f"<mem_{mem_idx}>" for mem_idx in range(num_mem_tokens)]
        doc_tokens.append("".join(mem_tokens))

    if sep_token:
        return sep_token.join(doc_tokens) + sep_token
    else:
        return "".join(doc_tokens)


def test_multi_qa_data_normalization():
    """Test that normalize_clara_record preserves multiple questions/answers."""
    # Simulate stage-1 data with multiple QA pairs
    record = {
        "data_type": "qa",
        "question": ["What is the capital?", "What is the population?", "What is the language?"],
        "answers": ["Paris", "67 million", "French"],
        "docs": ["France is a country in Europe.", "Paris is the capital of France."],
    }

    example = normalize_clara_record(record)

    assert isinstance(example.question, list), "Questions should be preserved as list"
    assert isinstance(example.answer, list), "Answers should be preserved as list"
    assert len(example.question) == 3, f"Expected 3 questions, got {len(example.question)}"
    assert len(example.answer) == 3, f"Expected 3 answers, got {len(example.answer)}"

    print("✓ Data normalization preserves multiple QA pairs")
    print(f"  Questions: {example.question}")
    print(f"  Answers: {example.answer}")


def test_single_qa_data_normalization():
    """Test that single QA pairs work as before."""
    record = {
        "data_type": "qa",
        "question": "What is the capital?",
        "answer": "Paris",
        "docs": ["France is a country in Europe."],
    }

    example = normalize_clara_record(record)

    assert isinstance(example.question, str), "Single question should be string"
    assert isinstance(example.answer, str), "Single answer should be string"
    assert example.question == "What is the capital?"
    assert example.answer == "Paris"

    print("✓ Single QA pairs still work correctly")


def test_multi_qa_prompt_building():
    """Test that prompt builder correctly formats multiple QA pairs."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.add_tokens(["<SEP>"] + [f"<mem_{i}>" for i in range(8)], special_tokens=True)

    questions = ["What is the capital?", "What is the population?", "What is the language?"]
    answers = ["Paris", "67 million", "French"]

    prompt_text, _prompt_len = build_original_qa_prompt(
        tokenizer=tokenizer,
        question=questions,
        answer=answers,
        num_docs=3,
        num_mem_tokens=8,
        sep_token="<SEP>",
    )

    print("\n" + "=" * 80)
    print("MULTI-QA PROMPT:")
    print("=" * 80)
    print(prompt_text)
    print("=" * 80)

    # Verify all QA pairs are present
    assert "Question: What is the capital?" in prompt_text
    assert "Answer: Paris" in prompt_text
    assert "Question: What is the population?" in prompt_text
    assert "Answer: 67 million" in prompt_text
    assert "Question: What is the language?" in prompt_text
    assert "Answer: French" in prompt_text

    # Verify they're in assistant turn
    assert "<|im_start|>assistant" in prompt_text

    # Count QA pairs
    qa_count = prompt_text.count("Question:")
    assert qa_count == 3, f"Expected 3 QA pairs, found {qa_count}"

    print(f"\n✓ Multi-QA prompt correctly formatted with {qa_count} QA pairs")
    print(f"✓ Prompt length (before answers): {_prompt_len} tokens")


def test_single_qa_prompt_building():
    """Test that single QA pairs still work."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.add_tokens(["<SEP>"] + [f"<mem_{i}>" for i in range(8)], special_tokens=True)

    prompt_text, _prompt_len = build_original_qa_prompt(
        tokenizer=tokenizer,
        question="What is the capital?",
        answer="Paris",
        num_docs=1,
        num_mem_tokens=8,
        sep_token="<SEP>",
    )

    assert "Question: What is the capital?" in prompt_text
    assert "Answer: Paris" in prompt_text

    qa_count = prompt_text.count("Question:")
    assert qa_count == 1, f"Expected 1 QA pair, found {qa_count}"

    print("\n✓ Single QA prompt works correctly")


if __name__ == "__main__":
    test_multi_qa_data_normalization()
    test_single_qa_data_normalization()
    test_multi_qa_prompt_building()
    test_single_qa_prompt_building()
    print("\n✅ All multi-QA tests passed!")
