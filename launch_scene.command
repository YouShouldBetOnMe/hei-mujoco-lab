#!/bin/zsh
set -eu
BASE="${0:A:h}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${MUJOCO_PYTHON:-}" ]]; then
  exec "$MUJOCO_PYTHON" "$BASE/simulate.py" --workbench "$@"
fi
PYTHON_BIN="$BASE/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
exec "$PYTHON_BIN" "$BASE/launch.py" "$@"
