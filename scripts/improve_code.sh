set -euo pipefail

if command -v improve-code >/dev/null 2>&1; then
	exec improve-code "$@"
fi

if command -v uv >/dev/null 2>&1; then
	exec uv run improve-code "$@"
fi

echo "improve-code is not installed and uv is not available." >&2
echo "Install via dotfiles: ~/dotfiles/setup.sh (or: cd ~/dotfiles/custom-tools/improve-code && uv pip install -e .)" >&2
exit 1