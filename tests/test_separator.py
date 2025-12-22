"""Unit tests for CLaRa separator token functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lamb.clara_collate import build_memory_token_string


def test_separator_insertion():
    """Test that separator tokens are correctly inserted between documents."""
    # Test without separator
    tokens_no_sep = build_memory_token_string(3, 4, sep_token="")
    assert "<SEP>" not in tokens_no_sep
    assert tokens_no_sep == "<mem_0><mem_1><mem_2><mem_3>" * 3
    print("✓ No separator: tokens repeated correctly")

    # Test with separator
    tokens_with_sep = build_memory_token_string(3, 4, sep_token="<SEP>")
    assert "<SEP>" in tokens_with_sep
    assert tokens_with_sep.count("<SEP>") == 3  # One after each document
    
    expected = "<mem_0><mem_1><mem_2><mem_3><SEP>" * 3
    assert tokens_with_sep == expected
    print(f"✓ With separator: {tokens_with_sep}")

    # Test single document
    tokens_single = build_memory_token_string(1, 2, sep_token="<SEP>")
    assert tokens_single == "<mem_0><mem_1><SEP>"
    print(f"✓ Single document: {tokens_single}")

    # Test original style (20 docs, 8 tokens each)
    tokens_original = build_memory_token_string(20, 8, sep_token="<SEP>")
    assert tokens_original.count("<SEP>") == 20
    assert tokens_original.count("<mem_0>") == 20  # Reused across docs
    print(f"✓ Original style (20 docs): {len(tokens_original)} chars, {tokens_original.count('<SEP>')} separators")


def test_memory_token_reuse():
    """Test that memory tokens are reused across documents (not unique per doc)."""
    tokens = build_memory_token_string(3, 4, sep_token="<SEP>")
    
    # Each document should use the same memory token IDs
    doc1 = "<mem_0><mem_1><mem_2><mem_3>"
    assert tokens.startswith(doc1 + "<SEP>")
    
    # Verify reuse pattern
    parts = tokens.split("<SEP>")[:-1]  # Remove trailing empty string
    assert len(parts) == 3
    assert all(part == doc1 for part in parts)
    print("✓ Memory tokens correctly reused across documents")


if __name__ == "__main__":
    test_separator_insertion()
    test_memory_token_reuse()
    print("\n✅ All separator token tests passed!")
