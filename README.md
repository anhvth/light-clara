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
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Format code:

```bash
uv run black lamb/
uv run isort lamb/
```

Lint code:

```bash
uv run ruff check lamb/
uv run mypy lamb/
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