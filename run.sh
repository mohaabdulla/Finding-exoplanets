#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-auto}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found in PATH." >&2
  exit 1
fi

if [ "$PIP_EXTRA_ARGS" = "auto" ]; then
  if "$PYTHON_BIN" -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages'; then
    # Debian/Ubuntu system Python needs this flag when PEP 668 is enforced.
    PIP_ARGS=("--break-system-packages")
  else
    PIP_ARGS=()
  fi
else
  # shellcheck disable=SC2206
  PIP_ARGS=($PIP_EXTRA_ARGS)
fi

echo "Installing dependencies from requirements.txt..."
"$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" -r requirements.txt

echo "Running training script (train.py)..."
"$PYTHON_BIN" train.py
