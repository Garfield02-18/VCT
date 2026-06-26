# Airbox Reproduction Notes

This repository is intended to reproduce the Team14 control-link smoke test on
Radxa AIRbox. It is not a full reproduction package for Team14's original
ManiFlow experiment results.

## What Is Included

- Team14 source code under `src/`.
- Qualcomm conversion helper scripts under `qualcomm/`.
- `scripts/smoke_test_no_assets.py`, which exercises the Team14 code path
  without real checkpoints or QNN deployment artifacts.
- `scripts/smoke_test_act_qnn_backend.py`, which adapts a bundled ACT/QNN HTP
  artifact to Team14's `PolicyRunner` interface for control-link validation.
- `assets/qnn_friendly_policy_bundle/`, a small substitute ACT/QNN model bundle
  containing ONNX/QNN artifacts, HTP context binary, and sample input tensors.
- `reports/Team14_ACT_QNN_Control_Link_Test_Report.md`, which records the local
  Airbox test result, latency, FPS, model size, and precision information.

## What Is Not Included

The following files are not included in this Git repository:

- Team14 trained checkpoints, such as `*.pt`.
- Team14 original QNN deployment artifacts for the ManiFlow policy.
- Demonstration or evaluation datasets, such as `*.zarr`.
- Qualcomm QAIRT SDK/runtime files.
- Robot hardware drivers or upper-computer runtime outside this project.

Because QAIRT is not included, another Airbox can directly reproduce the
code-only smoke test after installing Python dependencies. The ACT/QNN backend
smoke test is reproducible after installing QAIRT and using the bundled model
artifact under `assets/qnn_friendly_policy_bundle/`.

## Test Level 1: Code-Only Smoke Test

This test does not need Team14 checkpoints or QNN files. It creates temporary
fake checkpoint and normalizer files, then checks whether the Team14 model,
postprocess, policy runner, safety check, and simulated evaluation path can run.

Example command:

```bash
cd /path/to/VCT
source /home/radxa/miniconda3/etc/profile.d/conda.sh
conda activate llm
PYTHONPATH=. python3 scripts/smoke_test_no_assets.py
```

Minimum Python dependencies:

- `numpy`
- `torch`

Expected result:

- The script exits without exception.
- It prints a final success message for the Team14 code path.

## Test Level 2: ACT/QNN Backend Smoke Test

This test uses the bundled ACT-style QNN HTP context binary to replace the
missing Team14 QNN deployment artifact. It is only for validating whether the
Team14 control link can call a real QNN backend and receive an action tensor.
It does not reproduce Team14's original task accuracy or success rate.

The bundled ACT/QNN artifact follows this IO contract:

| Tensor | Shape | dtype |
| --- | --- | --- |
| `image` | `1 x 3 x 480 x 640` | float32 |
| `qpos` | `1 x 14` | float32 |
| `task_embedding` | `1 x 32` | float32 |
| `actions` | `1 x 100 x 14` | float32 |

The adapter maps it to Team14's expected contract:

| Team14 Tensor | Shape | Adapter Behavior |
| --- | --- | --- |
| `rgb` | `1 x 3 x 224 x 224` | resized to `1 x 3 x 480 x 640` |
| `robot_state` | `1 x 14` | passed as `qpos` |
| `action_chunk` | `1 x 16 x 14` | first 16 steps are cropped from ACT `actions` |

Bundled model files:

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
|   `-- qnn_friendly_policy_net.json
`-- sample_inputs/
    |-- image.raw
    |-- qpos.raw
    |-- task_embedding.raw
    `-- input_list.txt
```

Required QAIRT layout on the Airbox:

```text
/path/to/qairt/2.47.1/
|-- bin/aarch64-oe-linux-gcc11.2/qnn-net-run
|-- lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so
`-- lib/hexagon-v73/unsigned/
```

Example command when QAIRT is installed at the default local path:

```bash
cd /path/to/VCT
PYTHONPATH=. python3 scripts/smoke_test_act_qnn_backend.py
```

Example command with an explicit QAIRT path:

```bash
cd /path/to/VCT
PYTHONPATH=. python3 scripts/smoke_test_act_qnn_backend.py \
  --qairt-root /path/to/qairt/2.47.1
```

Expected result:

```text
ACTION_SHAPE=(1, 14)
LATENCY_MS=<measured value>
SAFETY_OK=True REASON=ok
[done] Team14 PolicyRunner successfully used ACT/QNN backend
```

## Local Reference Result

The local test recorded in
`reports/Team14_ACT_QNN_Control_Link_Test_Report.md` used the bundled ACT/QNN
HTP context binary with approximately 0.478M ONNX parameters and float32
external IO.

Measured reference results:

| Test Item | Result |
| --- | --- |
| Pure QNN/HTP NetRun IPS | 136.7716 inf/sec |
| Pure QNN/HTP average NetRun latency | 5.842 ms |
| Team14 adapter end-to-end average latency | 256.65 ms |
| Team14 adapter end-to-end FPS | 3.90 FPS |

The adapter FPS is much lower than pure NetRun IPS because it includes Python
process launch, input file generation, qnn-net-run invocation, output parsing,
and Team14 policy/safety wrapper overhead.

## Reproducibility Summary

Clone-only reproduction is possible for:

- Importing and checking the Team14 code structure.
- Running `scripts/smoke_test_no_assets.py` after installing Python
  dependencies.

Clone plus QAIRT reproduction is possible for:

- Running `scripts/smoke_test_act_qnn_backend.py` with the bundled substitute
  ACT/QNN model.

Clone-only reproduction is not possible for:

- Team14 original trained-policy inference.
- Team14 original QNN deployment for the ManiFlow policy.
- Original Team14 task success-rate evaluation.

To reproduce the ACT/QNN backend smoke test on another Airbox, install the
matching QAIRT runtime on that device. No separate VCT/ACT model bundle copy is
needed because this repository now includes the small substitute bundle under
`assets/`.
