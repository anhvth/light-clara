"""Quick integration test for separator tokens in training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer

from lamb.clara_collate import build_original_qa_prompt
from lamb.config import ClaraConfig


def test_separator_in_prompt():
    """Test that separators appear correctly in prompts."""
    config = ClaraConfig(
        model_name="Qwen/Qwen3-0.6B",
        use_clara_original=True,
        use_sep_token=True,
        compress_rate=8,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Add <SEP> token
    tokenizer.add_tokens(["<SEP>"] + [f"<mem_{i}>" for i in range(8)], special_tokens=True)

    # Build a prompt with 3 docs
    prompt_text, prompt_len = build_original_qa_prompt(
        tokenizer=tokenizer,
        question="What is the capital of France?",
        answer="Paris",
        num_docs=3,
        num_mem_tokens=8,
        sep_token="<SEP>",
    )

    print("=" * 80)
    print("PROMPT TEXT:")
    print("=" * 80)
    print(prompt_text)
    print("=" * 80)

    # Verify separators are present
    assert "<SEP>" in prompt_text, "Separator not found in prompt!"
    assert prompt_text.count("<SEP>") == 3, (
        f"Expected 3 separators, found {prompt_text.count('<SEP>')}"
    )

    # Verify memory tokens are present
    assert "<mem_0>" in prompt_text
    assert "<mem_7>" in prompt_text

    # Tokenize and verify
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    sep_id = tokenizer.convert_tokens_to_ids("<SEP>")
    sep_count = tokens.count(sep_id)

    print(f"\n✓ Prompt contains {prompt_text.count('<SEP>')} <SEP> tokens (text)")
    print(f"✓ Tokenized prompt contains {sep_count} <SEP> tokens (IDs)")
    print(f"✓ Prompt length (before answer): {prompt_len} tokens")
    print(f"✓ Total length: {len(tokens)} tokens")

    return True


if __name__ == "__main__":
    test_separator_in_prompt()
    print("\n✅ Integration test passed!")
