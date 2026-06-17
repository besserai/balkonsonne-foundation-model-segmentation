#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
MODEL_DIR="$ROOT_DIR/sam3-bucket"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUCKET_URI="${SAM3_BUCKET_URI:-hf://buckets/moritz-envite-1/sam3-bucket}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: hf CLI not found. Install with: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "${1:-}" == "--skip-sync" ]]; then
  echo "Skipping SAM3 bucket sync (--skip-sync)."
else
  mkdir -p "$MODEL_DIR"
  hf sync "$BUCKET_URI" "$MODEL_DIR"
fi

echo "Setup complete."
echo "Run inference with:"
echo "  $VENV_DIR/bin/python $ROOT_DIR/predict_sky_masks.py --model-dir $MODEL_DIR --test-images-dir $ROOT_DIR/test_images --output-dir $ROOT_DIR/output"
