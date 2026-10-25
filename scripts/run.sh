#!/usr/bin/env bash
# Runs the ClinicalLingo CLI with the portable Python.  Examples:
#   bash run.sh explain "echogenic" --mode word
#   bash run.sh doctor
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PY="$CL_ROOT/linux/python/bin/python3"
if [ ! -x "$PY" ]; then
  echo "Python is not set up yet. Run:  bash $CL_ROOT/linux/setup.sh" >&2
  exit 1
fi
cd "$CL_REPO"
exec "$PY" "$CL_REPO/cl.py" "$@"
