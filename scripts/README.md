# Development Scripts

This directory contains utility scripts for code quality automation using **ruff** as the single tool for all code quality tasks (formatting, linting, import sorting).

## Scripts Overview

### 🚀 `improve_code.sh` - Main Automation Script

**The primary script** that handles all code quality tasks with different commands.

```bash
# Full improvement (format + fix + report)
./scripts/improve_code.sh

# Or specify a command:
./scripts/improve_code.sh improve  # same as default
./scripts/improve_code.sh format   # format only
./scripts/improve_code.sh report   # report only
./scripts/improve_code.sh view     # view latest report
```

**Commands:**

#### `improve` (default)
**What it does:**
1. **Auto-formats** all Python code with ruff
2. **Auto-fixes** linting issues that can be fixed automatically
3. **Generates a report** of remaining issues that need manual attention

**Output:**
- Console output showing progress and summary
- Detailed report saved to `logs/improve_YYYYMMDD_HHMMSS.md`

**Use this script before:**
- Committing code
- Creating pull requests
- After making significant changes

#### `format`
Quick formatting without linting checks.

**What it does:**
- Formats all Python files in `lamb/` directory
- Applies consistent style (quotes, indentation, line breaks)

**When to use:**
- Quick style fixes
- Before reviewing code
- When you just need formatting

#### `report`
Generates a report without making any changes.

**What it does:**
- Checks for syntax errors
- Checks for linting issues
- Generates report without fixing anything

**Output:**
- Report saved to `logs/report_YYYYMMDD_HHMMSS.md`

**When to use:**
- To see what needs fixing without making changes
- For CI/CD pipelines
- To verify code state

#### `view`
Display the most recent code quality report.

**What it does:**
- Finds the most recent report or improvement log
- Displays it in the terminal

**When to use:**
- To quickly check the latest code quality status
- After running improve_code.sh or report_error.sh
- To review what issues remain

---

## Ruff Configuration

All code quality is managed through ruff configuration in `pyproject.toml`:

- **Formatting**: Replaces black
- **Linting**: Replaces flake8, pylint
- **Import Sorting**: Replaces isort
- **Type Hints**: Basic checks (replaces basic mypy functionality)

### Ruff Rules Enabled

- `E`, `W`: pycodestyle errors and warnings
- `F`: pyflakes
- `I`: isort (import sorting)
- `B`: flake8-bugbear
- `C4`: flake8-comprehensions
- `UP`: pyupgrade
- `N`: pep8-naming
- `PL`: pylint
- `RUF`: ruff-specific rules
- `SIM`: flake8-simplify

## Manual Commands

If you need to run specific tasks:

```bash
# Format code
uv run ruff format lamb/

# Check linting (without fixing)
uv run ruff check lamb/

# Auto-fix linting issues
uv run ruff check --fix lamb/

# Show fixable issues
uv run ruff check --fix --diff lamb/

# Check specific file
uv run ruff check lamb/model.py
```

## Understanding Reports

Reports are saved to `logs/` directory and include:

1. **Syntax Errors**: Code that won't compile
2. **Lint Errors**: Critical issues (E, F rules)
3. **Lint Warnings**: Style and complexity issues (N, PL, SIM rules)

### Common Warnings

- **PLR0912**: Too many branches - simplify conditional logic
- **PLR0915**: Too many statements - break function into smaller parts
- **N812**: Naming convention - use lowercase for modules
- **SIM102**: Simplify nested if statements
- **PLC0415**: Import at top of file - move imports to module level
- **RUF005**: Use unpacking instead of concatenation

## Best Practices

1. **Run `improve_code.sh` frequently** during development
2. Review the generated report for warnings
3. Address complexity warnings (PLR0912, PLR0915) for maintainability
4. Keep reports in `logs/` for reference (already gitignored)
5. Use `improve_code.sh format` before quick commits
6. Use `improve_code.sh report` in CI/CD pipelines

## Integration with Development Workflow

```bash
# Before committing
./scripts/improve_code.sh

# If report shows issues:
# 1. Review logs/improve_*.md
# 2. Fix critical errors manually
# 3. Consider addressing warnings
# 4. Re-run improve_code.sh to verify
./scripts/improve_code.sh
```

## Troubleshooting

**"ruff not available"**
```bash
uv sync --extra dev
```

**"Permission denied"**
```bash
chmod +x scripts/*.sh
```

**Too many warnings**
- Focus on errors first (E, F codes)
- Warnings are informational - address gradually
- Adjust `pyproject.toml` to ignore specific rules if needed
