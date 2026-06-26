# QNN-Friendly Action Policy Bundle

This small bundle is included so `scripts/smoke_test_act_qnn_backend.py` can run
on a Radxa AIRbox with a real QNN/HTP context binary, even though Team14's
original QNN deployment artifact is unavailable.

This is a substitute backend for control-link validation only. It is not a
Team14 trained ManiFlow checkpoint and should not be used to report Team14 task
accuracy or success rate.

## Model IO

```text
image [1,3,480,640] float32
qpos [1,14] float32
task_embedding [1,32] float32
-> actions [1,100,14] float32
```

## Included Files

```text
qnn_friendly_policy.onnx
qnn_friendly_policy.onnx.data
model_io.json
htp_context/qnn_friendly_policy_htp.bin
qnn_artifacts/qnn_friendly_policy.cpp
qnn_artifacts/qnn_friendly_policy.bin
qnn_artifacts/qnn_friendly_policy_net.json
sample_inputs/*.raw
```

Qualcomm QAIRT SDK/runtime files are not included. Install QAIRT separately on
the Airbox and pass its path with `--qairt-root` if it is not located at
`/home/radxa/qairt/2.47.1`.

## Team14 Adapter Smoke Test

From the repository root:

```bash
PYTHONPATH=. python3 scripts/smoke_test_act_qnn_backend.py \
  --qairt-root /path/to/qairt/2.47.1
```

The adapter resizes Team14's synthetic `3 x 224 x 224` RGB tensor to the
`3 x 480 x 640` image tensor expected by this model, passes the Team14
14-dimensional state as `qpos`, and crops the returned `1 x 100 x 14` action
sequence to Team14's `16 x 14` action chunk contract.

## Direct QNN NetRun Test

The included `run_on_airbox_htp.sh` can also run this bundle directly with
`qnn-net-run` after QAIRT is installed:

```bash
cd assets/qnn_friendly_policy_bundle
QAIRT_SDK_ROOT=/path/to/qairt/2.47.1 ./run_on_airbox_htp.sh
```

Generated output directories such as `output_htp_*` are intentionally ignored by
Git.
