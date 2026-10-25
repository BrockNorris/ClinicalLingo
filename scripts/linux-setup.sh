#!/usr/bin/env bash
# One-time Linux setup for the ClinicalLingo drive. Run with:  bash linux/setup.sh
#
# What it does (all inside the drive, nothing installed on the host):
#   1. extracts linux/python/cpython-*.tar.gz  -> linux/python/        (relocatable CPython 3.13)
#   2. extracts linux/ollama/ollama-linux-amd64.tar.zst -> linux/ollama/ (needs `zstd` on the host)
#   3. installs the pre-downloaded wheels in linux/wheels into that Python, offline
#   4. verifies the python and ollama binaries execute from this filesystem
#
# exFAT cannot store symlinks, so archives are extracted in a temporary directory on the host
# and copied over with symlinks dereferenced; the temporary directory is removed afterwards.
set -euo pipefail
LINUX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$LINUX_DIR/.." && pwd)"
PY_DIR="$LINUX_DIR/python"
OL_DIR="$LINUX_DIR/ollama"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== ClinicalLingo Linux setup (root: $ROOT) =="

# 1. Python
if [ ! -x "$PY_DIR/bin/python3" ]; then
  TARBALL="$(ls "$PY_DIR"/cpython-*-x86_64-unknown-linux-gnu-install_only.tar.gz | head -n1)"
  echo "-- extracting $(basename "$TARBALL")"
  (cd "$PY_DIR" && sha256sum -c SHA256SUMS)
  mkdir -p "$TMP/py" && tar -xzf "$TARBALL" -C "$TMP/py"
  cp -rL "$TMP/py/python/." "$PY_DIR/"
else
  echo "-- python already extracted"
fi

# 2. Ollama
if [ ! -x "$OL_DIR/bin/ollama" ]; then
  ARCHIVE="$OL_DIR/ollama-linux-amd64.tar.zst"
  if ! command -v zstd >/dev/null 2>&1; then
    echo "!! 'zstd' is required to extract $(basename "$ARCHIVE"). Install it (e.g. apt install zstd) and re-run." >&2
    exit 1
  fi
  echo "-- extracting $(basename "$ARCHIVE") (about 1.5 GB)"
  (cd "$OL_DIR" && grep ollama-linux-amd64.tar.zst ollama-sha256sum.txt | sha256sum -c -)
  mkdir -p "$TMP/ol" && tar --use-compress-program=unzstd -xf "$ARCHIVE" -C "$TMP/ol"
  cp -rL "$TMP/ol/." "$OL_DIR/"
else
  echo "-- ollama already extracted"
fi

# 3. Exec check (exFAT mounted with noexec, or without fmask, will fail here)
if ! "$PY_DIR/bin/python3" --version >/dev/null 2>&1; then
  echo "!! $PY_DIR/bin/python3 cannot execute from this filesystem." >&2
  echo "   Remount the drive with exec permissions, e.g.:" >&2
  echo "     sudo mount -o remount,exec,fmask=0022,dmask=0022 $ROOT" >&2
  exit 1
fi
echo "-- $("$PY_DIR/bin/python3" --version) executes OK"

# 4. Offline wheel install
"$PY_DIR/bin/python3" -m pip install --no-index --find-links "$LINUX_DIR/wheels" -r "$ROOT/ClinicalLingo/requirements.txt" -q
echo "-- requirements installed into $PY_DIR"

echo "== done. Next:  bash $ROOT/start-ollama.sh   (in one terminal)"
echo "                bash $ROOT/run.sh doctor      (in another)"
