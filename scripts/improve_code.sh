#!/usr/bin/env bash
# Comprehensive code quality script
# Usage: ./scripts/improve_code.sh [command]
# Commands:
#   improve (default): format, auto-fix, and report
#   format: format code only
#   report: generate report only
#   view: view latest report

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

COMMAND="${1:-improve}"

case "${COMMAND}" in
    "format")
        echo "🎨 Formatting code with ruff..."
        uv run ruff format src/lamb/
        echo "🔧 Sorting imports with isort..."
        uv run isort src/lamb/
        echo "✅ Code formatting and import sorting complete!"
        ;;
    "report")
        TS="$(date +%Y%m%d_%H%M%S)"
        REPORT="${LOG_DIR}/report_${TS}.md"
        echo "📊 Generating error report..."
        python - <<'PY' | tee "${REPORT}"
import pathlib
import py_compile
import subprocess
import sys
from datetime import datetime

root = pathlib.Path.cwd()
python_files = [p for p in root.rglob('*.py') if not any(
    part in p.parts for part in ['.venv', 'venv', '__pycache__', '.git', 'node_modules']
)]

print("# Error Report")
print(f"- Generated: {datetime.now().isoformat()}")
print(f"- Root: {root}")
print(f"- Files scanned: {len(python_files)}")
print()

errors = []

# Check syntax with py_compile
print("## Syntax Validation")
for path in python_files:
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        errors.append((path, "SyntaxError", str(exc)))

if errors:
    print(f"❌ Found {len(errors)} syntax error(s)")
else:
    print("✅ No syntax errors detected")
print()

# Check linting with ruff
print("## Linting Issues")
try:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "--output-format", "json", str(root / "lamb")],
        capture_output=True,
        text=True,
        cwd=root
    )
    
    if result.stdout.strip():
        import json
        try:
            ruff_output = json.loads(result.stdout)
            
            # Separate by severity
            lint_errors = []
            lint_warnings = []
            for violation in ruff_output:
                file_path = pathlib.Path(violation["filename"])
                code = violation["code"]
                message = violation["message"]
                line = violation.get("location", {}).get("row", "?")
                
                # Simple heuristic: codes starting with E or F are errors, others are warnings
                if code.startswith(("E", "F")):
                    lint_errors.append((file_path, code, message, line))
                else:
                    lint_warnings.append((file_path, code, message, line))
            
            if lint_errors:
                print(f"❌ Found {len(lint_errors)} linting error(s):")
                errors_by_file = {}
                for path, code, message, line in lint_errors:
                    if path not in errors_by_file:
                        errors_by_file[path] = []
                    errors_by_file[path].append((code, message, line))
                
                for path in sorted(errors_by_file.keys()):
                    print(f"### {path.relative_to(root)}")
                    for code, message, line in errors_by_file[path]:
                        print(f"- **Line {line}** [{code}]: {message}")
                    print()
            else:
                print("✅ No critical linting errors")
            
            if lint_warnings:
                print(f"⚠️  Found {len(lint_warnings)} linting warning(s):")
                warnings_by_file = {}
                for path, code, message, line in lint_warnings:
                    if path not in warnings_by_file:
                        warnings_by_file[path] = []
                    warnings_by_file[path].append((code, message, line))
                
                for path in sorted(warnings_by_file.keys()):
                    print(f"### {path.relative_to(root)}")
                    for code, message, line in warnings_by_file[path]:
                        print(f"- **Line {line}** [{code}]: {message}")
                    print()
            else:
                print("✅ No linting warnings")
        except json.JSONDecodeError:
            print("⚠️  Could not parse ruff output")
            print(result.stdout)
            errors.append((root / "lamb", "RuffError", result.stdout.strip()))
except FileNotFoundError:
    print("❌ ruff not available - please install dev dependencies: uv sync --group dev")
    errors.append((root, "ToolMissing", "ruff not available"))
    sys.exit(1)

# Summary
total_issues = len(errors) + len(lint_errors) if 'lint_errors' in locals() else 0 + len(lint_warnings) if 'lint_warnings' in locals() else 0
print("## Summary")
print(f"- **Syntax Errors**: {len([e for e in errors if e[1] == 'SyntaxError'])}")
print(f"- **Lint Errors**: {len(lint_errors) if 'lint_errors' in locals() else 0}")
print(f"- **Lint Warnings**: {len(lint_warnings) if 'lint_warnings' in locals() else 0}")
print(f"- **Total Issues**: {total_issues}")
print()

if total_issues == 0:
    print("🎉 **All checks passed! Code is clean.**")
    sys.exit(0)
else:
    print("⚠️  **Manual fixes required for the issues above.**")
    sys.exit(1)
PY

        STATUS=$?
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if [[ ${STATUS} -eq 0 ]]; then
            echo "✅ Report generated successfully."
            echo "📄 Report saved to: ${REPORT}"
        else
            echo "⚠️  Issues found."
            echo "📄 Full report saved to: ${REPORT}"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ;;
    "view")
        # Find the most recent improve or report file
        LATEST_IMPROVE=$(find "${LOG_DIR}" -name "improve_*.md" -type f 2>/dev/null | sort -r | head -1)
        LATEST_REPORT=$(find "${LOG_DIR}" -name "report_*.md" -type f 2>/dev/null | sort -r | head -1)

        # Determine which is more recent
        LATEST=""
        if [[ -n "${LATEST_IMPROVE}" && -n "${LATEST_REPORT}" ]]; then
            if [[ "${LATEST_IMPROVE}" > "${LATEST_REPORT}" ]]; then
                LATEST="${LATEST_IMPROVE}"
            else
                LATEST="${LATEST_REPORT}"
            fi
        elif [[ -n "${LATEST_IMPROVE}" ]]; then
            LATEST="${LATEST_IMPROVE}"
        elif [[ -n "${LATEST_REPORT}" ]]; then
            LATEST="${LATEST_REPORT}"
        fi

        if [[ -z "${LATEST}" ]]; then
            echo "No reports found in ${LOG_DIR}"
            exit 1
        fi

        echo "📄 Showing: ${LATEST}"
        echo ""
        cat "${LATEST}"
        ;;
    "improve"|*)
        TS="$(date +%Y%m%d_%H%M%S)"
        REPORT="${LOG_DIR}/improve_${TS}.md"

        echo "🔧 Running automated code improvements..."
        echo ""

        # Step 1: Format code
        echo "📝 Step 1: Auto-formatting code..."
        if uv run ruff format src/lamb/; then
            echo "✅ Formatting complete"
        else
            echo "⚠️  Formatting had issues (continuing...)"
        fi
        echo ""

        # Step 1.5: Sort imports
        echo "🔧 Step 1.5: Sorting imports with isort..."
        if uv run isort src/lamb/; then
            echo "✅ Import sorting complete"
        else
            echo "⚠️  Import sorting had issues (continuing...)"
        fi
        echo ""

        # Step 2: Auto-fix linting issues
        echo "🔍 Step 2: Auto-fixing linting issues..."
        if uv run ruff check --fix src/lamb/; then
            echo "✅ Auto-fix complete"
        else
            echo "⚠️  Some issues could not be auto-fixed (continuing...)"
        fi
        echo ""

        # Step 3: Generate comprehensive report
        echo "📊 Step 3: Generating error report..."
        python - <<'PY' | tee "${REPORT}"
import pathlib
import py_compile
import subprocess
import sys
from datetime import datetime

root = pathlib.Path.cwd()
python_files = [p for p in root.rglob('*.py') if not any(
    part in p.parts for part in ['.venv', 'venv', '__pycache__', '.git', 'node_modules']
)]

print("# Code Improvement Report")
print(f"- Generated: {datetime.now().isoformat()}")
print(f"- Root: {root}")
print(f"- Files scanned: {len(python_files)}")
print()

# Collect all issues
syntax_errors = []
lint_errors = []
lint_warnings = []

# Check syntax with py_compile
print("## Syntax Validation")
for path in python_files:
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        syntax_errors.append((path, str(exc)))

if syntax_errors:
    print(f"❌ Found {len(syntax_errors)} syntax error(s):")
    print()
    for path, message in syntax_errors:
        print(f"### {path.relative_to(root)}")
        print(f"```")
        print(message)
        print(f"```")
        print()
else:
    print("✅ No syntax errors detected")
print()

# Check linting with ruff (remaining issues after auto-fix)
print("## Linting Issues (Not Auto-fixable)")
try:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "--output-format", "json", str(root / "lamb")],
        capture_output=True,
        text=True,
        cwd=root
    )
    
    if result.stdout.strip():
        import json
        try:
            ruff_output = json.loads(result.stdout)
            
            # Separate by severity
            for violation in ruff_output:
                file_path = pathlib.Path(violation["filename"])
                code = violation["code"]
                message = violation["message"]
                line = violation.get("location", {}).get("row", "?")
                
                # Simple heuristic: codes starting with E or F are errors, others are warnings
                if code.startswith(("E", "F")):
                    lint_errors.append((file_path, code, message, line))
                else:
                    lint_warnings.append((file_path, code, message, line))
        except json.JSONDecodeError:
            print("⚠️  Could not parse ruff output")
            print(result.stdout)
except FileNotFoundError:
    print("❌ ruff not available - please install dev dependencies: uv sync --group dev")
    sys.exit(1)

if lint_errors:
    print(f"❌ Found {len(lint_errors)} linting error(s):")
    print()
    errors_by_file = {}
    for path, code, message, line in lint_errors:
        if path not in errors_by_file:
            errors_by_file[path] = []
        errors_by_file[path].append((code, message, line))
    
    for path in sorted(errors_by_file.keys()):
        print(f"### {path.relative_to(root)}")
        for code, message, line in errors_by_file[path]:
            print(f"- **Line {line}** [{code}]: {message}")
        print()
else:
    print("✅ No critical linting errors")
print()

if lint_warnings:
    print(f"⚠️  Found {len(lint_warnings)} linting warning(s):")
    print()
    warnings_by_file = {}
    for path, code, message, line in lint_warnings:
        if path not in warnings_by_file:
            warnings_by_file[path] = []
        warnings_by_file[path].append((code, message, line))
    
    for path in sorted(warnings_by_file.keys()):
        print(f"### {path.relative_to(root)}")
        for code, message, line in warnings_by_file[path]:
            print(f"- **Line {line}** [{code}]: {message}")
        print()
else:
    print("✅ No linting warnings")
print()

# Summary
total_issues = len(syntax_errors) + len(lint_errors) + len(lint_warnings)
print("## Summary")
print(f"- **Syntax Errors**: {len(syntax_errors)}")
print(f"- **Lint Errors**: {len(lint_errors)}")
print(f"- **Lint Warnings**: {len(lint_warnings)}")
print(f"- **Total Issues**: {total_issues}")
print()

if total_issues == 0:
    print("🎉 **All checks passed! Code is clean.**")
    sys.exit(0)
else:
    print("⚠️  **Manual fixes required for the issues above.**")
    sys.exit(1)
PY

        STATUS=$?

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if [[ ${STATUS} -eq 0 ]]; then
            echo "✅ All improvements complete! Code is clean."
            echo "📄 Report saved to: ${REPORT}"
        else
            echo "⚠️  Some issues require manual attention."
            echo "📄 Full report saved to: ${REPORT}"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ;;
esac
