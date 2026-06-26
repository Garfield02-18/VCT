#!/usr/bin/env bash
set -euo pipefail

QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/home/radxa/qairt/2.47.1}"
QAIRT_ARCH="${QAIRT_ARCH:-aarch64-oe-linux-gcc11.2}"
NUM_INFERENCES="${NUM_INFERENCES:-100}"
PERF_PROFILE="${PERF_PROFILE:-burst}"
LOG_LEVEL="${LOG_LEVEL:-error}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$QAIRT_SDK_ROOT/bin/$QAIRT_ARCH"
LIB_DIR="$QAIRT_SDK_ROOT/lib/$QAIRT_ARCH"
HEXAGON_DIR="$QAIRT_SDK_ROOT/lib/hexagon-v73/unsigned"
MODEL_LIB="$ROOT/qnn_artifacts/lib/$QAIRT_ARCH/libqnn_friendly_policy.so"
CONTEXT_DIR="$ROOT/htp_context"
CONTEXT_FILE="$CONTEXT_DIR/qnn_friendly_policy_htp.bin"
OUTPUT_DIR="$ROOT/output_htp_$(date +%Y%m%d_%H%M%S)"

if [[ ! -x "$BIN_DIR/qnn-net-run" ]]; then
  echo "ERROR: missing qnn-net-run: $BIN_DIR/qnn-net-run" >&2
  exit 1
fi
if [[ ! -x "$BIN_DIR/qnn-context-binary-generator" ]]; then
  echo "ERROR: missing qnn-context-binary-generator: $BIN_DIR/qnn-context-binary-generator" >&2
  exit 1
fi
# A prebuilt HTP context is included in this repository, so the compiled model
# library is only needed when rebuilding the context binary.
if [[ ! -s "$CONTEXT_FILE" && ! -f "$MODEL_LIB" ]]; then
  echo "ERROR: missing compiled model library: $MODEL_LIB" >&2
  echo "The model library is only required when $CONTEXT_FILE is absent." >&2
  echo "Run convert_on_x86.sh on the x86 host, then copy qnn_artifacts back here." >&2
  exit 1
fi

export PATH="$BIN_DIR:$PATH"
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$HEXAGON_DIR;$LIB_DIR"

if [[ "${SKIP_MAKE_SAMPLE_INPUTS:-0}" != "1" ]]; then
  python3 "$ROOT/make_sample_inputs.py" >/dev/null
fi

mkdir -p "$CONTEXT_DIR"
if [[ ! -s "$CONTEXT_FILE" ]]; then
  echo "CONTEXT_BUILD_START note=first build may take several seconds"
  qnn-context-binary-generator \
    --backend "$LIB_DIR/libQnnHtp.so" \
    --model "$MODEL_LIB" \
    --binary_file qnn_friendly_policy_htp \
    --output_dir "$CONTEXT_DIR" \
    --log_level warn
fi

if [[ ! -s "$CONTEXT_FILE" ]]; then
  ALT_CONTEXT="$(find "$CONTEXT_DIR" -maxdepth 1 -type f -name '*.bin' -size +0c | sort | head -n 1)"
  if [[ -n "$ALT_CONTEXT" ]]; then
    CONTEXT_FILE="$ALT_CONTEXT"
  else
    echo "ERROR: no context binary found in $CONTEXT_DIR" >&2
    exit 1
  fi
fi

echo "CONTEXT_FILE=$CONTEXT_FILE"
echo "NUM_INFERENCES=$NUM_INFERENCES"
echo "ADSP_LIBRARY_PATH=$ADSP_LIBRARY_PATH"

qnn-net-run \
  --backend "$LIB_DIR/libQnnHtp.so" \
  --retrieve_context "$CONTEXT_FILE" \
  --input_list "$ROOT/sample_inputs/input_list.txt" \
  --output_dir "$OUTPUT_DIR" \
  --use_native_input_files \
  --use_native_output_files \
  --num_inferences "$NUM_INFERENCES" \
  --keep_num_outputs 1 \
  --perf_profile "$PERF_PROFILE" \
  --profiling_level basic \
  --log_level "$LOG_LEVEL"

ACTIONS="$(find "$OUTPUT_DIR" -path '*/Result_0/actions.raw' -type f | head -n 1)"
if [[ -n "$ACTIONS" ]]; then
  python3 - "$ACTIONS" <<'PY'
from pathlib import Path
import array
import sys
p = Path(sys.argv[1])
vals = array.array("f")
vals.frombytes(p.read_bytes())
print(f"ACTION_FILE={p}")
print(f"ACTION_FLOATS={len(vals)}")
if len(vals) == 1400:
    print("ACTION_SHAPE=(1,100,14)")
print("ACTION_FIRST14=[" + ", ".join(f"{v:.6g}" for v in vals[:14]) + "]")
PY
fi

PROFILE="$(find "$OUTPUT_DIR" -maxdepth 1 -name 'qnn-profiling-data_*.log' | sort | head -n 1)"
VIEWER="$BIN_DIR/qnn-profile-viewer"
if [[ -n "$PROFILE" && -x "$VIEWER" ]]; then
  "$VIEWER" --input_log "$PROFILE" 2>/dev/null | awk '
    /NetRun IPS/ { print; next }
    /Execute Stats \(Average\)/ { avg=1; print; next }
    /Execute Stats \(Min\)/ { avg=0 }
    avg && /Graph 0/ { print; next }
    avg && /NetRun:/ { print; next }
    avg && /Backend \(QNN accelerator \(execute\) time\):/ { print; next }
    avg && /Backend \(Accelerator \(execute\) time\):/ { print; next }
  '
fi

echo "OUTPUT_DIR=$OUTPUT_DIR"
