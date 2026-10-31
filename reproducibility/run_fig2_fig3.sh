#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON_BIN:-python3}" "$SCRIPT_DIR/build_fig2_fig3_from_figure_data.py"
