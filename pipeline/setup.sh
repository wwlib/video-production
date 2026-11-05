#!/usr/bin/env bash
# Setup the Code de Vie post pipeline on macOS (Apple Silicon) or Linux.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
echo "== Code de Vie pipeline setup =="
echo "ROOT=$ROOT"

if ! command -v ffmpeg >/dev/null; then
  if command -v brew >/dev/null; then
    echo "Installing ffmpeg via Homebrew..."
    brew install ffmpeg
  else
    echo "ERROR: ffmpeg not found. Install ffmpeg (Homebrew: brew install ffmpeg) and re-run." >&2
    exit 1
  fi
fi
ffmpeg -version | head -1

PY=python3
if command -v python3.11 >/dev/null; then PY=python3.11; fi
echo "Using $($PY --version)"

if [[ ! -d .venv ]]; then
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt

mkdir -p "${CDV_DATA:-$HOME/video-data/code-de-vie}"/{audio,subs,models,out}
./scripts/fetch_models.sh
# Seed rembg human-seg weight if we shipped one
if [[ -f "${CDV_DATA:-$HOME/video-data/code-de-vie}/models/rembg/u2net_human_seg.onnx" ]]; then
  mkdir -p "$HOME/.u2net"
  cp -n "${CDV_DATA:-$HOME/video-data/code-de-vie}/models/rembg/u2net_human_seg.onnx" "$HOME/.u2net/u2net_human_seg.onnx" || true
fi

echo
echo "Setup OK."
echo "  venv:     $ROOT/.venv"
echo "  data dir: ${CDV_DATA:-$HOME/video-data/code-de-vie}"
echo "Put media into the data dir (see README), then: ./run.sh test"
