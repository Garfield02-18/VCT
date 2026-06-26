#!/usr/bin/env bash
# Stage runtime artefacts on the AIRbox Q900 and run one-step NPU inference.
# Mirrors Listing 6 in the final report.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

QAIRT_ROOT="${QAIRT_ROOT:-$HOME/qairt/2.42.0.251225}"
QAIRT_AARCH64_LIB="$QAIRT_ROOT/lib/aarch64-ubuntu-gcc9.4"
# NB: DSP_ARCH is the bare number (73), NOT the "v73" form used in
# config_backend.json / config.py.  Qualcomm HTP libraries are named
# libQnnHtpV${DSP_ARCH}Stub.so (e.g. libQnnHtpV73Stub.so).
DSP_ARCH="${DSP_ARCH:-73}"

AIRBOX_SSH="${AIRBOX_SSH:-radxa@192.168.3.42}"
BOARD_DIR="${BOARD_DIR:-/home/radxa/maniflow_q900}"

CTX_BIN="${CTX_BIN:-$PROJECT_ROOT/deploy/qnn/context_binary/maniflow_handover_1step.bin}"
INPUT_LIST_HOST="${INPUT_LIST_HOST:-$PROJECT_ROOT/deploy/qnn/one_step_eval_input_list.txt}"

echo "[run_on_airbox] preparing remote directories"
ssh "$AIRBOX_SSH" "mkdir -p $BOARD_DIR/model $BOARD_DIR/lib $BOARD_DIR/io $BOARD_DIR/qnn_output"

echo "[run_on_airbox] copying context binary + qnn-net-run + HTP libs"
scp "$CTX_BIN"                                  "$AIRBOX_SSH:$BOARD_DIR/model/"
scp "$INPUT_LIST_HOST"                          "$AIRBOX_SSH:$BOARD_DIR/io/"
scp "$QAIRT_ROOT/bin/aarch64-ubuntu-gcc9.4/qnn-net-run" "$AIRBOX_SSH:$BOARD_DIR/"
scp "$QAIRT_AARCH64_LIB/libQnnHtp.so"           "$AIRBOX_SSH:$BOARD_DIR/lib/"
scp "$QAIRT_AARCH64_LIB/libQnnHtpV${DSP_ARCH}Stub.so"  "$AIRBOX_SSH:$BOARD_DIR/lib/"
scp "$QAIRT_AARCH64_LIB/libQnnHtpV${DSP_ARCH}Skel.so"  "$AIRBOX_SSH:$BOARD_DIR/lib/"
scp "$QAIRT_AARCH64_LIB/libQnnHtpNetRunExtensions.so"  "$AIRBOX_SSH:$BOARD_DIR/lib/"

echo "[run_on_airbox] running one-step NPU inference on the AIRbox Q900"
ssh "$AIRBOX_SSH" "cd $BOARD_DIR && \
    export LD_LIBRARY_PATH=$BOARD_DIR/lib:\$LD_LIBRARY_PATH && \
    ./qnn-net-run \
        --backend ./lib/libQnnHtp.so \
        --retrieve_context ./model/$(basename "$CTX_BIN") \
        --input_list ./io/$(basename "$INPUT_LIST_HOST") \
        --output_dir ./qnn_output"

echo "[run_on_airbox] done. Outputs are in $AIRBOX_SSH:$BOARD_DIR/qnn_output"
