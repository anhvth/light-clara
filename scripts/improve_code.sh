set -euo pipefail

run_fallback() {
	echo "[improve_code.sh] Falling back to ruff (improve-code unavailable)." >&2
	uv run ruff format .
	uv run ruff check --fix .
	uv run ruff check .
}

# Some environments may have an `improve-code` shim on PATH which is not executable.
# Prefer `uv run` when available, but fall back to ruff if it fails.
if command -v uv >/dev/null 2>&1; then
	if uv run improve-code "$@"; then
		exit 0
	fi
	run_fallback
	exit 0
fi

if command -v improve-code >/dev/null 2>&1; then
	tool_path="$(command -v improve-code)"
	if [ -x "$tool_path" ]; then
		"$tool_path" "$@" && exit 0
	fi
fi

echo "improve-code is not installed and uv is not available." >&2
echo "Install via dotfiles: ~/dotfiles/setup.sh (or: cd ~/dotfiles/custom-tools/improve-code && uv pip install -e .)" >&2
exit 1