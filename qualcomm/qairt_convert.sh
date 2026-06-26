#!/usr/bin/env bash
# Host-side conversion pipeline ONNX -> DLC -> int8 DLC -> Context-Binary.
# Mirrors Listing 3 in the final report, with the Team14 file layout.

set -euo pipefail

QAIRT_ROOT="${QAIRT_ROOT:-$HOME/qairt/2.42.0.251225}"
PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

ONNX_PATH="${ONNX_PATH:-$PROJECT_ROOT/deploy/onnx/maniflow_handover_1step.onnx}"
DLC_DIR="${DLC_DIR:-$PROJECT_ROOT/deploy/qnn}"
DLC_FP32="$DLC_DIR/maniflow_handover_1step.dlc"
DLC_INT8="$DLC_DIR/maniflow_handover_1step_int8.dlc"
INPUT_LIST="${INPUT_LIST:-$DLC_DIR/maniflow_calib_input_list.txt}"
CTX_DIR="${CTX_DIR:-$DLC_DIR/context_binary}"
CTX_NAME="${CTX_NAME:-maniflow_handover_1step}"

QNN_CONFIG="${QNN_CONFIG:-$PROJECT_ROOT/qualcomm/config_file.json}"

mkdir -p "$DLC_DIR" "$CTX_DIR"

export PATH="$QAIRT_ROOT/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="$QAIRT_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"

echo "[qairt_convert] ONNX -> DLC"
qairt-converter \
    --input_network "$ONNX_PATH" \
    --output_path "$DLC_FP32" \
    -d "rgb" 1,3,224,224 \
    -d "robot_state" 1,14

echo "[qairt_convert] generating raw calibration tensors"
python "$PROJECT_ROOT/qualcomm/tools/export_qnn_raw_inputs.py" \
    --dataset-root "${DATASET_ROOT:-$PROJECT_ROOT/data/handover_240}" \
    --tasks cucumber pepper banana \
    --num-samples "${CALIB_SAMPLES:-96}" \
    --image-size 224 \
    --state-dim 14 \
    --output-dir "$DLC_DIR/calib_raw"

python "$PROJECT_ROOT/qualcomm/tools/make_qnn_input_list.py" \
    --raw-dir "$DLC_DIR/calib_raw" \
    --inputs rgb robot_state \
    --output "$INPUT_LIST"

echo "[qairt_convert] quantising to INT8"
qairt-quantizer \
    --input_dlc "$DLC_FP32" \
    --input_list "$INPUT_LIST" \
    --output_dlc "$DLC_INT8"

echo "[qairt_convert] generating Context-Binary for the AIRbox Q900"
qnn-context-binary-generator \
    --model libQnnModelDlc.so \
    --backend libQnnHtp.so \
    --dlc_path "$DLC_INT8" \
    --output_dir "$CTX_DIR" \
    --binary_file "$CTX_NAME" \
    --config_file "$QNN_CONFIG"

echo "[qairt_convert] done. Context binary: $CTX_DIR/${CTX_NAME}.bin"
