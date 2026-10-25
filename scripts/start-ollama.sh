#!/usr/bin/env bash
# Starts the portable Ollama server. Leave this terminal open while using ClinicalLingo.
# HOME is redirected so Ollama's identity key is written to <root>/home, not ~/.ollama.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HOME="$CL_ROOT/home"
mkdir -p "$HOME"
if [ ! -x "$CL_ROOT/linux/ollama/bin/ollama" ]; then
  echo "Ollama is not set up yet. Run:  bash $CL_ROOT/linux/setup.sh" >&2
  exit 1
fi
echo "ClinicalLingo: starting Ollama from $CL_ROOT/linux/ollama (models: $OLLAMA_MODELS)"
echo "First model load from a flash drive can take 1-3 minutes; later requests are fast."
exec "$CL_ROOT/linux/ollama/bin/ollama" serve
