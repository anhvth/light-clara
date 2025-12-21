# LaMB Vertical

Latent Memory Bridge for RAG - A PyTorch implementation of memory-augmented language models.

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
uv sync
```

### Using pip

```bash
pip install -e .
```

## Usage

Train a LaMB model:

```bash
uv run lamb --model-name Qwen/Qwen3-4B-Instruct-2507 --dataset-size 20000 --max-steps 100
```

Or activate the virtual environment and run:

```bash
source .venv/bin/activate
lamb --model-name Qwen/Qwen3-4B-Instruct-2507 --dataset-size 20000 --max-steps 100
```

## Development

Install development dependencies:

```bash
uv sync --extra dev
```

### Code Quality Automation

This project uses **ruff** exclusively for all code quality tasks (formatting, linting, import sorting). Use the provided scripts for automated code improvements:

```bash
# Main script: Auto-format, auto-fix, and generate report
./scripts/improve_code.sh

# Quick format only
./scripts/format_code.sh

# Generate error report without changes
./scripts/report_error.sh
```

See [scripts/README.md](scripts/README.md) for detailed documentation.

### Manual Commands

```bash
# Format code
uv run ruff format lamb/

# Auto-fix linting issues
uv run ruff check --fix lamb/

# Check without fixing
uv run ruff check lamb/
```

### Run Tests

```bash
uv run pytest
```

## Project Structure

```
lamb/
├── __init__.py      # Package exports
├── bridge.py        # VerticalLatentMemoryBridge implementation
├── cli.py           # Command-line interface
├── config.py        # Configuration classes
├── debug.py         # Debugging utilities
├── generate.py      # Text generation functions
├── model.py         # LaMBModel implementation
├── train.py         # Training loop
└── utils.py         # Utility functions
```

## License

MIT License