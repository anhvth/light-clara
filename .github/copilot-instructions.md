# LaMB Vertical - AI Coding Guidelines

## Architecture Overview

**LaMB (Latent Memory Bridge)** is a memory-augmented language model for RAG that compresses long contexts into fixed-size memory tokens using a vertical bridge mechanism.

### Core Components

- **`LaMBModel`** (`lamb/model.py`): Main model wrapping a base transformer with PEFT LoRA adapters and memory bridge
- **`VerticalLatentMemoryBridge`** (`lamb/bridge.py`): Compresses layer-wise hidden states into KV memory tokens for efficient attention
- **Training**: Uses KL divergence between teacher (full context) and student (memory-compressed) forward passes

### Key Patterns

**PEFT Integration**: Base models are typed as `Any` due to dynamic adapter methods. Use `cast(torch.nn.Module, model)` for device operations.

```python
self.base_model: Any  # PEFT-enhanced model with adapter methods
cast(torch.nn.Module, self.base_model).to(device=device)
```

**Configuration**: Uses dataclasses with factory functions for device/dtype detection. Environment variables override defaults.

```python
device: str = field(default_factory=pick_device)
dtype: torch.dtype = field(default_factory=lambda: pick_dtype(pick_device()))
```

**Text Processing**: Training splits chat-formatted text at assistant markers for context/target separation.

```python
splitter = "<|im_start|>assistant\n"
ctx_str, tgt_str = txt.split(splitter, 1)
ctx_str += splitter  # Include splitter in context
```

## Development Workflow

**Dependency Management**: Use `uv` exclusively. Never use `pip` directly.

```bash
uv sync                    # Install dependencies
uv sync --group dev       # Add development tools
uv run lamb --help        # Run CLI commands
```

**Code Quality**: Strict type checking and formatting enforced.

```bash
uv run ruff check lamb/   # Lint (includes import sorting)
uv run mypy lamb/         # Type check
uv run black lamb/        # Format
uv run isort lamb/        # Sort imports
```

**Validation**: Use the error reporting script for comprehensive checks.

```bash
./script/report_error.sh   # Syntax + linting validation
```

## Project Conventions

**Type Annotations**: Python 3.8+ compatible - use `Union[X, Y]` not `X | Y`, `List[int]` not `list[int]`.

**Error Handling**: Training gracefully handles non-finite losses by skipping batches, not crashing.

**Device Management**: Always use `pick_device()`, `pick_dtype()`, `pick_attn_implementation()` for cross-platform compatibility.

**Memory Tokens**: Configurable via `num_memory_tokens` (default: 32). Bridge compresses context into this fixed size.

**LoRA Configuration**: Applied to attention and MLP layers for compression. Only these parameters are trainable.

**Debugging**: Use `debug_reproduce_training()` for overfitting validation during training steps.</content>
<parameter name="filePath">/Users/anhvth/projects/ml-clara-v2/.github/copilot-instructions.md