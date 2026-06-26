#!/usr/bin/env python3
"""Run Team14 code-path smoke tests without real checkpoints or QNN artefacts.

This script is intentionally not a performance or task-success test. It creates
small temporary assets and synthetic observations so the AIRbox can verify that
imports, tensor shapes, policy runner glue, packet encoding, safety checks, and
simulation entry points are usable even when the real demo data/checkpoints/QNN
context binary are unavailable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bridge.board_client import encode_packet
from src.common.config import CONFIG
from src.inference.postprocess import (
    pack_inputs_for_qnn,
    preprocess_rgb,
    split_packet,
    unpack_qnn_action,
)
from src.inference.run_policy import PolicyRunner, TorchBackend, _Backend
from src.models import HandoverManiFlowPolicy
from src.robot.safety_checks import SafetyMonitor


class FakeQnnBackend(_Backend):
    """Shape-compatible stand-in for QnnContextBinaryBackend."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, rgb_chw: np.ndarray, state_14: np.ndarray) -> np.ndarray:
        packed = pack_inputs_for_qnn(rgb_chw, state_14)
        assert packed["rgb"].shape == CONFIG.rgb_shape
        assert packed["robot_state"].shape == CONFIG.state_shape
        # Keep actions near the current state so the default safety monitor can
        # accept the first command during the smoke test.
        base = state_14.astype(np.float32)[None, :]
        noise = self.rng.normal(0.0, 0.02, size=(CONFIG.shape.action_horizon, CONFIG.shape.action_dim))
        return np.repeat(base, CONFIG.shape.action_horizon, axis=0) + noise.astype(np.float32)


def make_smoke_assets(asset_dir: Path) -> tuple[Path, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = asset_dir / "maniflow_smoke_best.pt"
    normalizer_path = asset_dir / "normalizer.json"

    policy = HandoverManiFlowPolicy(
        image_size=CONFIG.shape.image_size,
        state_dim=CONFIG.shape.state_dim,
        action_dim=CONFIG.shape.action_dim,
        action_horizon=CONFIG.shape.action_horizon,
        inference_steps=CONFIG.train.inference_steps,
    )
    normalizer = {
        "state": {
            "low": [-2.5] * CONFIG.shape.state_dim,
            "high": [2.5] * CONFIG.shape.state_dim,
        },
        "action": {
            "low": [-2.5] * CONFIG.shape.action_dim,
            "high": [2.5] * CONFIG.shape.action_dim,
        },
    }
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": {},
            "epoch": 0,
            "normalizer": normalizer,
            "policy_class": policy.__class__.__name__,
            "config": {
                "shape": CONFIG.shape.__dict__,
                "train": CONFIG.train.__dict__,
            },
        },
        checkpoint_path,
    )
    normalizer_path.write_text(json.dumps(normalizer, indent=2))
    return checkpoint_path, normalizer_path


def run_model_forward() -> None:
    policy = HandoverManiFlowPolicy().eval()
    rgb = torch.randn(*CONFIG.rgb_shape)
    state = torch.randn(*CONFIG.state_shape)
    with torch.no_grad():
        action = policy(rgb, state)
    assert tuple(action.shape) == CONFIG.action_shape
    assert torch.isfinite(action).all()
    print(f"[ok] model forward -> {tuple(action.shape)}")


def run_postprocess_and_packet() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb = preprocess_rgb(frame)
    state = np.zeros(CONFIG.shape.state_dim, dtype=np.float32)
    packed = pack_inputs_for_qnn(rgb, state)
    action_chunk = unpack_qnn_action(np.zeros(CONFIG.shape.action_horizon * CONFIG.shape.action_dim, dtype=np.float32))
    pkt = encode_packet(action_chunk[0], time.time())
    ts, action = split_packet(pkt)
    ok, reason = SafetyMonitor().check(action[None, :], state, action_timestamp=ts)
    assert packed["rgb"].shape == CONFIG.rgb_shape
    assert packed["robot_state"].shape == CONFIG.state_shape
    assert action.shape == (CONFIG.shape.action_dim,)
    assert ok, reason
    print("[ok] preprocess/qnn-pack/packet/safety")


def run_torch_runner(checkpoint_path: Path, normalizer_path: Path) -> None:
    runner = PolicyRunner(TorchBackend(str(checkpoint_path), device="cpu"), str(normalizer_path))
    rgb = np.random.rand(3, CONFIG.shape.image_size, CONFIG.shape.image_size).astype(np.float32)
    state = np.zeros(CONFIG.shape.state_dim, dtype=np.float32)
    action = runner.step(rgb, state)
    assert action.shape == (CONFIG.deploy.receding_horizon_first_n, CONFIG.shape.action_dim)
    assert np.isfinite(action).all()
    print(f"[ok] torch PolicyRunner -> {action.shape}, {runner.stats.last_inference_ms:.2f} ms")


def run_fake_qnn_runner(normalizer_path: Path) -> None:
    runner = PolicyRunner(FakeQnnBackend(), str(normalizer_path))
    rgb = np.random.rand(3, CONFIG.shape.image_size, CONFIG.shape.image_size).astype(np.float32)
    state = np.zeros(CONFIG.shape.state_dim, dtype=np.float32)
    action = runner.step(rgb, state)
    ok, reason = SafetyMonitor().check(action, state)
    assert action.shape == (CONFIG.deploy.receding_horizon_first_n, CONFIG.shape.action_dim)
    assert ok, reason
    print(f"[ok] fake-qnn PolicyRunner -> {action.shape}")


def run_simulated_eval(checkpoint_path: Path, output_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.training.evaluate",
        "--checkpoint",
        str(checkpoint_path),
        "--task",
        "cucumber",
        "--rollouts",
        "1",
        "--simulate",
        "--device",
        "cpu",
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env={**dict(), **__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)}, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    assert output_path.exists()
    print("[ok] simulated evaluate entrypoint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-dir",
        default=str(PROJECT_ROOT / "tmp" / "smoke_no_assets"),
        help="Directory for generated temporary checkpoint/normalizer/output.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip the simulated evaluate subprocess.")
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    checkpoint_path, normalizer_path = make_smoke_assets(asset_dir)
    print(f"[info] smoke checkpoint: {checkpoint_path}")
    print(f"[info] smoke normalizer : {normalizer_path}")

    run_model_forward()
    run_postprocess_and_packet()
    run_torch_runner(checkpoint_path, normalizer_path)
    run_fake_qnn_runner(normalizer_path)
    if not args.skip_eval:
        run_simulated_eval(checkpoint_path, asset_dir / "smoke_eval.json")

    print("[done] code-path smoke test passed without real checkpoint or QNN artefacts")


if __name__ == "__main__":
    main()
