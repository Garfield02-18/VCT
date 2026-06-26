#!/usr/bin/env bash
set -euo pipefail

QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/home/hx/Qualcomm/QAIRT/2.47.1}"
if [[ ! -f "$QAIRT_SDK_ROOT/bin/envsetup.sh" ]]; then
  echo "ERROR: QAIRT_SDK_ROOT is invalid: $QAIRT_SDK_ROOT" >&2
  exit 1
fi

source "$QAIRT_SDK_ROOT/bin/envsetup.sh"

MODEL="qnn_friendly_policy.onnx"
OUT_DIR="qnn_artifacts"
CPP="$OUT_DIR/qnn_friendly_policy.cpp"
BIN="$OUT_DIR/qnn_friendly_policy.bin"
LIB_OUT="$OUT_DIR/lib"
LIB_NAME="qnn_friendly_policy"

mkdir -p "$OUT_DIR" "$LIB_OUT"

qnn-onnx-converter \
  --input_network "$MODEL" \
  --output_path "$CPP" \
  --input_dim image 1,3,480,640 \
  --input_dim qpos 1,14 \
  --input_dim task_embedding 1,32

qnn-model-lib-generator \
  -c "$CPP" \
  -b "$BIN" \
  -t aarch64-oe-linux-gcc11.2 \
  -l "$LIB_NAME" \
  -o "$LIB_OUT"

find "$OUT_DIR" -maxdepth 4 -type f | sort
