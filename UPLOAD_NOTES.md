# Upload Notes

This repository contains the Team14 code snapshot plus local Airbox smoke-test
additions for control-link validation.

Included additions:

- `scripts/smoke_test_no_assets.py`: runs Team14 code-path checks without real
  checkpoints or QNN artifacts.
- `scripts/smoke_test_act_qnn_backend.py`: adapts the bundled ACT/QNN HTP
  artifact as a Team14 `PolicyRunner` backend.
- `assets/qnn_friendly_policy_bundle/`: small substitute ACT/QNN model bundle
  used by the backend smoke test.
- `reports/Team14_ACT_QNN_Control_Link_Test_Report.md`: test report with QNN
  latency/FPS and model-size data.
- `src/robot/safety_checks.py`: Python 3.12 dataclass default fix.

Excluded generated or external artifacts:

- temporary checkpoints
- `tmp/` and `results/`
- `__pycache__/`
- demo/evaluation data such as `.zarr`
- Team14 trained checkpoint files such as `.pt`, `.pth`, `.ckpt`
- Team14 original QNN deployment artifacts
- Qualcomm QAIRT SDK/runtime files
- QNN runtime outputs such as `output_htp_*` and profiling logs
