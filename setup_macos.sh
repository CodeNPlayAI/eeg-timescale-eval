#!/usr/bin/env bash
set -euo pipefail

# Create venv/ in this repository and install the pinned packages.
#   bash setup_macos.sh                 # figures and self-test only (no PyTorch)
#   bash setup_macos.sh --experiments   # + PyTorch, MNE, openneuro-py for the full pipeline
# Works on macOS and Linux. Run from any directory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REQ=requirements-figures.txt
if [[ "${1:-}" == "--experiments" ]]; then
  REQ=requirements-experiments.txt
elif [[ -n "${1:-}" ]]; then
  printf 'Usage: bash setup_macos.sh [--experiments]\n' >&2
  exit 2
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  printf 'ERROR: python3.12 is not found in PATH. Install Python 3.12 first.\n' >&2
  exit 1
fi

if [[ ! -d venv ]]; then
  python3.12 -m venv venv
elif [[ ! -f venv/bin/python ]]; then
  printf 'ERROR: existing venv/ is not a valid virtual environment. No changes made.\n' >&2
  exit 1
fi

venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r "$REQ"
venv/bin/python -m pip check
venv/bin/python --version
printf 'Environment ready (%s): %s/venv\n' "$REQ" "$ROOT"
printf 'Next: source venv/bin/activate && python pipeline/run_openneuro.py selftest\n'
