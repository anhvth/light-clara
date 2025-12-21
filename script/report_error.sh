#!/usr/bin/env bash
# Lightweight error report generator: compiles all Python files to catch syntax issues
# and writes the log to logs/report_<timestamp>.md.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT="${LOG_DIR}/report_${TS}.md"

run_checks() {
  python - <<'PY'
import pathlib
import py_compile
import subprocess
import sys
from datetime import datetime

root = pathlib.Path.cwd()
python_files = [p for p in root.rglob('*.py') if not any(
    part in p.parts for part in ['.venv', 'venv', '__pycache__', '.git', 'node_modules']
)]
errors: list[tuple[pathlib.Path, str, str]] = []

# Check syntax with py_compile
for path in python_files:
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        errors.append((path, "SyntaxError", str(exc)))

# Check linting with ruff
try:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "--output-format", "json", str(root / "lamb")],
        capture_output=True,
        text=True,
        cwd=root
    )
    if result.returncode != 0:
        import json
        try:
            ruff_output = json.loads(result.stdout)
            for violation in ruff_output:
                file_path = pathlib.Path(violation["filename"])
                code = violation["code"]
                message = violation["message"]
                errors.append((file_path, f"Ruff({code})", message))
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            errors.append((root / "lamb", "RuffError", result.stdout.strip()))
except FileNotFoundError:
    errors.append((root, "ToolMissing", "ruff not available"))

print(f"# Error Report")
print(f"- Generated: {datetime.now().isoformat()}")
print(f"- Root: {root}")
print(f"- Files scanned: {len(python_files)}")
print(f"- Errors: {len(errors)}\n")

if errors:
    # Group errors by file
    errors_by_file = {}
    for path, error_type, message in errors:
        if path not in errors_by_file:
            errors_by_file[path] = []
        errors_by_file[path].append((error_type, message))

    for path in sorted(errors_by_file.keys()):
        print(f"## {path.relative_to(root)}")
        for error_type, message in errors_by_file[path]:
            print(f"- **{error_type}**: {message}")
        print()
else:
    print("No syntax or linting errors detected.")

sys.exit(1 if errors else 0)
PY
}

STATUS=0
if ! OUTPUT="$(run_checks 2>&1)"; then
  STATUS=$?
fi

{
  echo "${OUTPUT}" | tee "${REPORT}"
} >/dev/null

if [[ ${STATUS} -ne 0 ]]; then
  echo "Check failed; see ${REPORT}" >&2
else
  echo "Check succeeded; report at ${REPORT}" >&2
fi

exit ${STATUS}
