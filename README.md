# VCT: Team14 Control-Link Test on Qualcomm-powered Radxa AIRbox

This repository packages the Team14 final-project code with a smoke-test
path focused on the Qualcomm-powered Radxa AIRbox. It is designed to check
whether the control and inference link can run on a Qualcomm-powered Radxa
AIRbox with a real Qualcomm QNN/HTP backend. Because the original Team14
project does not include the trained checkpoint or Team14's original QNN
deployment artifacts, this repository uses a small ACT-style QNN-friendly
action policy as a substitute backend.

The substitute model is located under `assets/qnn_friendly_policy_bundle/`. It
is used only to validate the software path from observation preprocessing, QNN
inference, and action output to safety checking. It does not represent the task
performance of Team14's original ManiFlow policy.

## What This Repository Can Reproduce

| Target | Status | Notes |
| --- | --- | --- |
| Team14 Python control-link smoke test | Supported | Runs without real model assets by generating temporary fake checkpoint data. |
| Qualcomm-powered Radxa AIRbox QNN/HTP backend smoke test | Supported after installing QAIRT | Uses the bundled VCT/ACT QNN model bundle. |
| Team14 original trained-policy inference | Not included | Team14 checkpoint and normalizer are missing. |
| Team14 original QNN deployment | Not included | Team14 ONNX/DLC/context-binary artifacts are missing. |
| Original task success-rate evaluation | Not included | Demo data, real robot runtime, and trained policy are missing. |

For a stricter reproduction checklist, see [`REPRODUCE.md`](REPRODUCE.md).

## Role of the Qualcomm-powered Radxa AIRbox

In this test, the Qualcomm-powered Radxa AIRbox is used as the edge inference
node and control-link node. The smoke-test data flow is:

```text
synthetic Team14 observation
-> Team14 PolicyRunner
-> VCT/ACT QNN adapter
-> Qualcomm qnn-net-run
-> HTP context binary on Qualcomm-powered Radxa AIRbox NPU/HTP
-> action tensor
-> Team14 postprocess and SafetyMonitor
```

The QNN target configuration used in local testing matches the
Qualcomm-powered Radxa AIRbox Q900/QCS9075 HTP deployment path:

| Item | Value used in this repository |
| --- | --- |
| Target device | Qualcomm-powered Radxa AIRbox Q900 |
| QNN backend | `libQnnHtp.so` |
| DSP architecture | `v73` |
| SoC ID in config | `77` |
| Default QAIRT path | `/home/radxa/qairt/2.47.1` |
| Default QAIRT architecture directory | `aarch64-oe-linux-gcc11.2` |
| HTP context | `assets/qnn_friendly_policy_bundle/htp_context/qnn_friendly_policy_htp.bin` |

This repository does not include Qualcomm QAIRT runtime files. The target
Qualcomm-powered Radxa AIRbox must already have QAIRT installed and must be able
to run `qnn-net-run` with the HTP backend.

## Extra Environment Required

### Required for the QNN/HTP Smoke Test on Qualcomm-powered Radxa AIRbox

- Qualcomm-powered Radxa AIRbox Q900, or a compatible Qualcomm HTP target device.
- Qualcomm QAIRT installed on the Qualcomm-powered Radxa AIRbox.
- QAIRT files available in the following layout, or specified with
  `--qairt-root` when running the script:

```text
/path/to/qairt/2.47.1/
|-- bin/aarch64-oe-linux-gcc11.2/qnn-net-run
|-- lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so
`-- lib/hexagon-v73/unsigned/
```

- Python 3 with `numpy` installed.
- Access to the HTP/NPU device nodes on the Qualcomm-powered Radxa AIRbox.
  A useful check is:

```bash
ls -l /dev/fastrpc-* /dev/dma_heap/system /dev/dma_heap/qcom,cma-secure-cdsp 2>/dev/null
```

### Required for the Code-Only Smoke Test

- Python 3.
- `numpy`.
- `torch`.

The local Qualcomm-powered Radxa AIRbox test used a conda environment named
`llm`, but the environment name is not required. The scripts can run as
long as the required Python packages are available.

### Not Provided by This Repository

- Qualcomm QAIRT SDK/runtime.
- Team14 trained checkpoint files such as `*.pt`, `*.pth`, or `*.ckpt`.
- Team14 original `normalizer.json` files.
- Team14 original ONNX/DLC/context-binary deployment artifacts.
- Demo or evaluation datasets such as `*.zarr`.
- Real robot drivers, real arm controllers, or upper-computer runtime outside
  this codebase.

## Quick Start on Qualcomm-powered Radxa AIRbox

Clone the repository and enter it:

```bash
git clone https://github.com/Garfield02-18/VCT.git
cd VCT
```

Run the Team14 code-only smoke test:

```bash
PYTHONPATH=. python3 scripts/smoke_test_no_assets.py
```

Run the Qualcomm-powered Radxa AIRbox QNN/HTP smoke test with the default
QAIRT path:

```bash
PYTHONPATH=. python3 scripts/smoke_test_act_qnn_backend.py
```

If QAIRT is not installed at the default path, specify it explicitly:

```bash
PYTHONPATH=. python3 scripts/smoke_test_act_qnn_backend.py \
  --qairt-root /path/to/qairt/2.47.1
```

Expected output format:

```text
ACTION_SHAPE=(1, 14)
LATENCY_MS=<measured value>
SAFETY_OK=True REASON=ok
[done] Team14 PolicyRunner successfully used ACT/QNN backend
```

## Bundled VCT/ACT QNN Model

This repository includes a substitute model bundle so the Qualcomm-powered
Radxa AIRbox QNN inference path can be tested even when Team14's original QNN
deployment artifacts are missing.

```text
assets/qnn_friendly_policy_bundle/
|-- model_io.json
|-- qnn_friendly_policy.onnx
|-- qnn_friendly_policy.onnx.data
|-- htp_context/
|   `-- qnn_friendly_policy_htp.bin
|-- qnn_artifacts/
|   |-- qnn_friendly_policy.cpp
|   |-- qnn_friendly_policy.bin
|   |-- qnn_friendly_policy_net.json
|   `-- lib/aarch64-ubuntu-gcc9.4/libqnn_friendly_policy.so
`-- sample_inputs/
    |-- image.raw
    |-- qpos.raw
    |-- task_embedding.raw
    `-- input_list.txt
```

Model inputs and outputs:

| Tensor | Shape | dtype |
| --- | --- | --- |
| `image` | `1 x 3 x 480 x 640` | float32 |
| `qpos` | `1 x 14` | float32 |
| `task_embedding` | `1 x 32` | float32 |
| `actions` | `1 x 100 x 14` | float32 |

Adapter behavior:

| Team14 Interface | VCT/ACT Bundle Interface |
| --- | --- |
| `rgb`, `1 x 3 x 224 x 224` | resized to `image`, `1 x 3 x 480 x 640` |
| `robot_state`, `1 x 14` | passed as `qpos`, `1 x 14` |
| `action_chunk`, `1 x 16 x 14` | first 16 steps are cropped from `actions`, `1 x 100 x 14` |

Reference results from the local Qualcomm-powered Radxa AIRbox test:

| Metric | Result |
| --- | --- |
| ONNX parameter count | 478,365, about 0.478M |
| External input/output dtype | float32 |
| Pure QNN/HTP NetRun IPS | 136.7716 inf/sec |
| Pure QNN/HTP average NetRun latency | 5.842 ms |
| Team14 adapter end-to-end average latency | 256.65 ms |
| Team14 adapter end-to-end FPS | 3.90 FPS |

The end-to-end FPS is lower than pure NetRun IPS because the adapter test also
includes Python input generation, process startup, `qnn-net-run` invocation,
output parsing, and Team14 wrapper overhead.

## Repository Layout

```text
VCT/
|-- assets/
|   `-- qnn_friendly_policy_bundle/     # bundled substitute VCT/ACT QNN model
|-- qualcomm/                           # Team14 Qualcomm conversion helpers
|-- reports/                            # local Qualcomm-powered Radxa AIRbox test report
|-- scripts/
|   |-- smoke_test_no_assets.py         # code-only smoke test
|   `-- smoke_test_act_qnn_backend.py   # Qualcomm-powered Radxa AIRbox QNN/HTP smoke test
|-- src/
|   |-- bridge/                         # Qualcomm-powered Radxa AIRbox/upper-computer UDP bridge code
|   |-- common/                         # shared shapes and deployment config
|   |-- inference/                      # PolicyRunner and postprocess logic
|   |-- models/                         # Team14 ManiFlow/FM model definitions
|   |-- robot/                          # dummy robot interface and safety checks
|   `-- training/                       # original Team14 training/evaluation code
|-- README.md
|-- REPRODUCE.md
`-- UPLOAD_NOTES.md
```

## Original Team14 Pipeline

The original Team14 project targets few-step bimanual handover using the
ManiFlow policy. The full workflow requires demo data, trained checkpoints,
normalizer files, and Team14's own converted QNN artifacts.

Original training command shape:

```bash
python -m src.training.train_maniflow \
  --task cucumber \
  --data-root ./data/handover_240 \
  --output-dir ./checkpoints/cucumber \
  --policy maniflow
```

Original deployment command shape:

```bash
python qualcomm/export_onnx.py \
  --checkpoint ./checkpoints/cucumber/maniflow_cucumber_best.pt \
  --onnx-path ./deploy/onnx/maniflow_handover_1step.onnx

bash qualcomm/qairt_convert.sh
bash qualcomm/run_on_airbox.sh
```

These commands are kept in the repository for code inspection and future work.
They cannot be fully run from this repository alone because the required Team14
data and trained model assets are not included.

## Notes

- `src/robot/robot_interface.py` includes `DummyEnv`, so the evaluation path can
  be tested without real robot hardware when using simulation mode.
- `src/robot/safety_checks.py` contains action range, action delta, and data
  freshness checks.
- The bundled VCT/ACT model is only for integration testing. It should not be
  interpreted as a Team14 ManiFlow experimental result.
