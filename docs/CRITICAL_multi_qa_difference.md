# CRITICAL: Multi-QA Training Format Difference

## Discovery Date: 2025-12-22

## The Issue

Our implementation uses:
- **1 question** per sample
- **1 answer** per sample  
- 20 documents per sample

Original CLaRa uses:
- **MULTIPLE questions** per sample (List[str])
- **MULTIPLE answers** per sample (List[str])
- 20 documents per sample

## Evidence from Original Code

### Function Signature
```python
# .extra/ml-clara/openrlhf/models/modeling_clara.py:990
def _blend_qa_prompt(self, docs: str, query: List[str], answer: List[str]) -> Tuple[int, str]:
```

### Prompt Construction
```python
# .extra/ml-clara/openrlhf/models/modeling_clara.py:999
qa_lines = [f"Question: {q}\nAnswer: {a}" for q, a in zip(query, answer)]
query_answer = "\n".join(qa_lines)
assistant_prompt = [{"role": "assistant", "content": query_answer}]
```

### Example Output Format
```
<|im_start|>system
You are a helpful assistant. Given a document, your task is to generate some single 
questions to cover all key information of the document and answer them sequentially.
<|im_end|>
<|im_start|>user
Background:
<mem_0>...<mem_31><SEP><mem_0>...<mem_31><SEP>...(20 docs)
<|im_end|>
<|im_start|>assistant
Question: What is the capital of France?
Answer: Paris
Question: Who is the president?
Answer: Emmanuel Macron
Question: What is the population?
Answer: 67 million
<|im_end|>
```

## Impact

This explains why the original system prompt says:
> "your task is to generate some single **questions** to cover all key information"

The plural "questions" indicates multiple Q&A pairs are expected!

## What We Need to Fix

1. **Dataset format**: Our data should provide `List[str]` for questions and answers, not single strings
2. **Collate function**: Should handle lists and join them with newlines
3. **Prompt builder**: Should iterate over Q&A pairs and format them

## Current Status

❌ Our implementation only trains on 1 Q&A pair per sample
✅ Original trains on MULTIPLE Q&A pairs per sample

This is a **major difference** that affects:
- Training efficiency (more Q&A pairs per forward pass)
- Model behavior (learns to generate multiple related Q&As)
- Data utilization (better coverage of document content)

## Next Steps

1. Check our dataset - does it have multiple Q&A pairs available?
2. Update collate functions to handle List[str] for questions/answers
3. Update prompt builders to join multiple Q&A pairs
4. Test with original data format
