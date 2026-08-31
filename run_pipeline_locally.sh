#!/usr/bin/env bash
# ==============================================================================
#  Local CI-Parity Quality & Validation Pipeline (Linux & macOS)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
    VENV_PY="$SCRIPT_DIR/venv/bin/python"
fi

if [ ! -f "$VENV_PY" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_PY. Run ./install_dependencies.sh first." >&2
    exit 1
fi

invoke_poetry() {
    "$VENV_PY" -m poetry "$@"
}

echo "=== AI Hybrid VHS Audio Restorer Quality Pipeline ==="

echo "==> Step 1: Verify Poetry"
invoke_poetry --version

echo "==> Step 2: Validate Poetry Lockfile"
invoke_poetry check --lock

echo "==> Step 3: Install Test & Development Dependencies"
# CI uses `poetry sync`, which prunes anything outside the selected groups. Locally
# that would strip the `ml` group (torch, audio-separator) that real restoration
# runs need, so this installs without pruning. Everything the gate checks is
# identical; only the extra runtime packages differ.
invoke_poetry install -v --only main,dev --no-root

PWSH_BIN="$(command -v pwsh || true)"
if [ -x /usr/bin/pwsh ]; then
    PWSH_BIN="/usr/bin/pwsh"
fi

if [ -n "$PWSH_BIN" ]; then
    echo "==> Step 4: Run PowerShell Lint (via pwsh)"
    "$PWSH_BIN" -NoProfile -Command '& ./.github/scripts/Invoke-PowerShellLint.ps1 -ScriptPaths @("./install_dependencies.ps1", "./run_pipeline_locally.ps1", "./scripts/install_pr_review_tooling.ps1", "./scripts/package_windows_installer.ps1", "./.github/scripts/Invoke-PowerShellLint.ps1")'
else
    echo "==> Step 4: pwsh not found in PATH — skipping PowerShell script lint"
fi

echo "==> Step 5: Run Ruff"
invoke_poetry run ruff check modules tests restore_audio_hybrid.py scripts/apply_patches.py

echo "==> Step 6: Run Black (format check)"
while IFS= read -r -d '' python_file; do
    invoke_poetry run black --check "$python_file"
done < <(git ls-files -z '*.py')

echo "==> Step 7: Run isort (import ordering check)"
invoke_poetry run isort --check-only --diff modules tests restore_audio_hybrid.py scripts/apply_patches.py

echo "==> Step 8: Run Taplo (TOML format check)"
toml_files=()
while IFS= read -r -d '' file; do
    toml_files+=("$file")
done < <(git ls-files -z '*.toml')
if [ ${#toml_files[@]} -gt 0 ]; then
    invoke_poetry run taplo fmt --check "${toml_files[@]}"
fi

echo "==> Step 9: Run Flake8"
invoke_poetry run flake8 modules tests restore_audio_hybrid.py scripts/apply_patches.py

echo "==> Step 10: Run Pylint"
invoke_poetry run pylint --errors-only modules tests restore_audio_hybrid.py scripts/apply_patches.py

echo "==> Step 11: Run Bandit Security Scan"
invoke_poetry run bandit -ll -ii -r modules restore_audio_hybrid.py scripts/apply_patches.py

echo "==> Step 12: Run pip-audit Dependency Scan"
invoke_poetry run pip-audit

echo "==> Step 13: Run Radon Cyclomatic Complexity & Maintainability Gates"
invoke_poetry run radon cc modules tests/conftest.py tests/unit tests/integration restore_audio_hybrid.py scripts/apply_patches.py -s > radon-report.txt
invoke_poetry run radon cc modules tests/conftest.py tests/unit tests/integration restore_audio_hybrid.py scripts/apply_patches.py -j -O radon-cc-report.json
invoke_poetry run radon mi modules tests/conftest.py tests/unit tests/integration restore_audio_hybrid.py scripts/apply_patches.py -s > radon-mi-report.txt
invoke_poetry run radon mi modules tests/conftest.py tests/unit tests/integration restore_audio_hybrid.py scripts/apply_patches.py -j -O radon-mi-report.json
invoke_poetry run radon raw modules tests/conftest.py tests/unit tests/integration restore_audio_hybrid.py scripts/apply_patches.py > radon-raw-report.txt
invoke_poetry run radon hal modules tests/conftest.py tests/unit tests/integration restore_audio_hybrid.py scripts/apply_patches.py > radon-hal-report.txt

# Run both gates and combine the exit codes, as CI does, so a CC failure does not
# hide an MI failure behind `set -e`.
cc_gate_status=0
mi_gate_status=0
invoke_poetry run python tests/tooling/radon_cc_gate.py radon-cc-report.json || cc_gate_status=$?
invoke_poetry run python tests/tooling/radon_mi_gate.py radon-mi-report.json || mi_gate_status=$?
if [ "$cc_gate_status" -ne 0 ] || [ "$mi_gate_status" -ne 0 ]; then
    exit 1
fi

echo "==> Step 14: Run Markdown Format Check & Lint"
# Every tracked Markdown file, not just the top level of each directory: a plain
# directory argument to pymarkdown does not recurse, which left the skill and
# workflow documents unchecked.
md_list_tmp="$(mktemp "${TMPDIR:-/tmp}/md_files.XXXXXX")"
if ! git ls-files -z '*.md' > "$md_list_tmp"; then
    echo "[ERROR] Failed to enumerate tracked Markdown files via git ls-files." >&2
    rm -f "$md_list_tmp"
    exit 1
fi

md_files=()
while IFS= read -r -d '' file; do
    md_files+=("$file")
done < "$md_list_tmp"
rm -f "$md_list_tmp"

if [ ${#md_files[@]} -eq 0 ]; then
    echo "[ERROR] Unable to enumerate tracked Markdown files via git ls-files." >&2
    exit 1
fi

invoke_poetry run mdformat --check "${md_files[@]}"
invoke_poetry run pymarkdown --config .pymarkdown.json scan "${md_files[@]}"

echo "==> Step 15: Resolve Shared Coverage Threshold Policy"
COVERAGE_THRESHOLD_VALUE="$(invoke_poetry run python -c \
    "from tests.tooling.threshold_policy import get_coverage_threshold; print(f'{get_coverage_threshold():.2f}')")"
if ! printf '%s' "$COVERAGE_THRESHOLD_VALUE" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
    echo "[ERROR] Coverage threshold is not numeric: '$COVERAGE_THRESHOLD_VALUE'" >&2
    exit 1
fi
echo "Using coverage threshold: $COVERAGE_THRESHOLD_VALUE"

echo "==> Step 16: Run Tests with Coverage"
invoke_poetry run pytest -o addopts= --cov=restore_audio_hybrid --cov=modules --cov-branch \
    --cov-report=xml --cov-report=json --cov-report=term --cov-fail-under="$COVERAGE_THRESHOLD_VALUE" tests/

echo "==> Step 17: Enforce Strict Per-File Coverage (>= $COVERAGE_THRESHOLD_VALUE%)"
invoke_poetry run python tests/tooling/quality_gate.py coverage.json --threshold "$COVERAGE_THRESHOLD_VALUE"

echo "==> Step 18: Generate Coverage Badge and Summary"
invoke_poetry run python tests/tooling/badge_report.py coverage.xml

echo -e "\n=== Local quality pipeline passed successfully! ==="
