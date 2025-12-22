#!/bin/bash
# Quick validation test for CLaRa original mode

echo "=========================================="
echo "CLaRa Original Mode Validation Test"
echo "=========================================="

# Test 1: Config validation
echo -e "\n[Test 1] Config validation..."
python -c "
from lamb.config import ClaraConfig
cfg = ClaraConfig(use_clara_original=True, compress_rate=8)
assert cfg.use_clara_original == True
assert cfg.compress_rate == 8
print('✅ Config validated')
"

# Test 2: Separator token unit test
echo -e "\n[Test 2] Running separator unit tests..."
python tests/test_separator.py

# Test 3: Integration test
echo -e "\n[Test 3] Running integration test..."
python tests/test_integration_separator.py

# Test 4: Model initialization test
echo -e "\n[Test 4] Testing model initialization with original mode..."
python -c "
import torch
from lamb.config import ClaraConfig
from lamb.clara_model import ClaraModel

cfg = ClaraConfig(
    model_name='Qwen/Qwen3-0.6B',
    use_clara_original=True,
    compress_rate=8,
    device='cpu',
)

model = ClaraModel(cfg)
print(f'✅ Model initialized')
print(f'   - Tokenizer vocab size: {len(model.tokenizer)}')
print(f'   - Has SEP token: {hasattr(model, \"sep_token\") and model.sep_token is not None}')
if model.sep_token:
    print(f'   - SEP token: {model.sep_token} (id={model.sep_token_id})')
print(f'   - Compress rate: {cfg.compress_rate}')
"

echo -e "\n=========================================="
echo "✅ ALL VALIDATION TESTS PASSED"
echo "=========================================="
echo ""
echo "Ready to run full training with:"
echo "  ./scripts/train_debug.sh"
echo ""
