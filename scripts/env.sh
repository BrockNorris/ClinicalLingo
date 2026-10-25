#!/usr/bin/env bash
# ClinicalLingo portable environment (Linux). Source this:  source ./env.sh
# Every path is relative to this file, so the mount point can change.
# Expected layout:  <root>/ClinicalLingo (repo)  <root>/linux/python  <root>/linux/ollama  <root>/models
CL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CL_ROOT
export CL_REPO="$CL_ROOT/ClinicalLingo"
export OLLAMA_MODELS="$CL_ROOT/models"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_TMPDIR="$CL_ROOT/tmp"
export PIP_CACHE_DIR="$CL_ROOT/pip-cache"
export PATH="$CL_ROOT/linux/ollama/bin:$CL_ROOT/linux/python/bin:$PATH"
# Ollama bundles its own CUDA runtime libraries; only the NVIDIA driver is needed on the host.
export LD_LIBRARY_PATH="$CL_ROOT/linux/ollama/lib/ollama${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$CL_ROOT/tmp"
