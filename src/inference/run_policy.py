"""Board-side / workstation-side single-frame inference driver.

This module implements the ``PolicyRunner`` used by both:
    * the offline sanity-check tool that compares PyTorch and ONNX outputs;
    * the AIRbox Q900 ``board_client`` that calls qnn-net-run for every
      observation and forwards the newest action.

Two backends are supported:

    1. ``TorchBackend`` - loads a PyTorch checkpoint and runs the standard
       1-step ManiFlow integrator on CPU/GPU.
    2. ``QnnContextBinaryBackend`` - subprocess wrapper around qnn-net-run.
       The Q900 NPU consumes the int8 context binary produced by
       ``qualcomm/qairt_convert.sh`` and writes the action tensor to a raw
       file that we read back into a numpy array.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.common.config import CONFIG
from src.inference.postprocess import (
    denormalize_action_chunk,
    pack_inputs_for_qnn,
    select_executable_steps,
    unpack_qnn_action,
)
from src.training.dataset import MinMaxNormalizer


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class _Backend:
    def __call__(self, rgb_chw: np.ndarray, state_14: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---- PyTorch backend -------------------------------------------------------


class TorchBackend(_Backend):
    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        import torch
        from src.models import HandoverFlowMatchingBaseline, HandoverManiFlowPolicy

        payload = torch.load(checkpoint_path, map_location=device)
        cls = payload.get("policy_class", "HandoverManiFlowPolicy")
        common = dict(
            image_size=CONFIG.shape.image_size,
            state_dim=CONFIG.shape.state_dim,
            action_dim=CONFIG.shape.action_dim,
            action_horizon=CONFIG.shape.action_horizon,
            inference_steps=CONFIG.train.inference_steps,
        )
        if cls == "HandoverFlowMatchingBaseline":
            policy = HandoverFlowMatchingBaseline(**common)
        else:
            policy = HandoverManiFlowPolicy(use_consistency_training=True, **common)
        policy.load_state_dict(payload["model"])
        policy.eval().to(device)

        self._policy = policy
        self._torch = torch
        self._device = device

    def __call__(self, rgb_chw: np.ndarray, state_14: np.ndarray) -> np.ndarray:
        torch = self._torch
        rgb = torch.from_numpy(rgb_chw).unsqueeze(0).to(self._device)
        state = torch.from_numpy(state_14).unsqueeze(0).to(self._device)
        with torch.no_grad():
            action = self._policy(rgb, state).squeeze(0).cpu().numpy()
        return action


# ---- AIRbox Q900 backend ---------------------------------------------------


class QnnContextBinaryBackend(_Backend):
    """Wrap the AIRbox-side ``qnn-net-run`` invocation.

    The constructor records where the context binary, qnn-net-run binary, and
    HTP libraries live on the device.  Each ``__call__`` writes the current
    rgb+state into a temporary directory, runs qnn-net-run with the input
    list pointing at those files, and reads the resulting action tensor.

    On the AIRbox itself (``board_client.py``) we typically reuse this class
    in-process; on the workstation we use it for export sanity checks via
    ``ssh radxa@... 'qnn-net-run ...'`` (handled by ``qairt_convert.sh``).
    """

    def __init__(
        self,
        context_binary: str,
        qnn_net_run: str,
        backend_lib: str,
        lib_dir: str,
        work_dir: Optional[str] = None,
    ) -> None:
        self.context_binary = context_binary
        self.qnn_net_run = qnn_net_run
        self.backend_lib = backend_lib
        self.lib_dir = lib_dir
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="qnn_run_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _write_inputs(self, rgb_chw: np.ndarray, state_14: np.ndarray) -> str:
        packed = pack_inputs_for_qnn(rgb_chw, state_14)
        rgb_path = self.work_dir / "rgb.raw"
        state_path = self.work_dir / "robot_state.raw"
        packed["rgb"].tofile(rgb_path)
        packed["robot_state"].tofile(state_path)

        list_path = self.work_dir / "input_list.txt"
        list_path.write_text(f"rgb:={rgb_path} robot_state:={state_path}\n")
        return str(list_path)

    def _read_output(self) -> np.ndarray:
        out = self.work_dir / "qnn_output" / "Result_0" / "action_chunk.raw"
        if not out.exists():
            raise FileNotFoundError(f"qnn-net-run did not produce {out}")
        size = CONFIG.shape.action_horizon * CONFIG.shape.action_dim
        return np.fromfile(out, dtype=np.float32, count=size)

    def __call__(self, rgb_chw: np.ndarray, state_14: np.ndarray) -> np.ndarray:
        input_list = self._write_inputs(rgb_chw, state_14)
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        cmd = [
            self.qnn_net_run,
            "--backend", self.backend_lib,
            "--retrieve_context", self.context_binary,
            "--input_list", input_list,
            "--output_dir", str(self.work_dir / "qnn_output"),
        ]
        subprocess.run(cmd, check=True, env=env)
        return unpack_qnn_action(self._read_output())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class RunnerStats:
    n_calls: int = 0
    last_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0


class PolicyRunner:
    """Combine a backend with the project-side normaliser and chunk policy."""

    def __init__(
        self,
        backend: _Backend,
        normalizer_path: str,
        first_n_executable: int = CONFIG.deploy.receding_horizon_first_n,
    ) -> None:
        with open(normalizer_path, "r") as f:
            data = json.load(f)
        self.state_norm = MinMaxNormalizer.from_dict(data["state"])
        self.action_norm = MinMaxNormalizer.from_dict(data["action"])
        self.backend = backend
        self.first_n = first_n_executable
        self.stats = RunnerStats()

    def step(self, rgb_chw: np.ndarray, state_14_raw: np.ndarray) -> np.ndarray:
        """Return the executable joint commands for the upper computer."""
        state_norm = self.state_norm.normalize(state_14_raw.astype(np.float32))
        t0 = time.perf_counter()
        action_norm_chunk = self.backend(rgb_chw, state_norm)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        action_chunk = denormalize_action_chunk(action_norm_chunk, self.action_norm)
        self.stats.n_calls += 1
        self.stats.last_inference_ms = elapsed_ms
        self.stats.avg_inference_ms = (
            (self.stats.avg_inference_ms * (self.stats.n_calls - 1) + elapsed_ms)
            / self.stats.n_calls
        )
        return select_executable_steps(action_chunk, self.first_n)


# ---------------------------------------------------------------------------
# CLI: a small "run one frame" sanity check
# ---------------------------------------------------------------------------


def _build_backend_from_args(args: argparse.Namespace) -> _Backend:
    if args.backend == "torch":
        return TorchBackend(args.torch_checkpoint, device=args.device)
    if args.backend == "qnn":
        return QnnContextBinaryBackend(
            context_binary=args.qnn_context_binary,
            qnn_net_run=args.qnn_net_run,
            backend_lib=args.qnn_backend_lib,
            lib_dir=args.qnn_lib_dir,
        )
    raise ValueError(f"Unknown backend {args.backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "qnn"], required=True)
    parser.add_argument("--normalizer", required=True,
                        help="Path to normalizer.json saved next to the checkpoint.")
    parser.add_argument("--torch-checkpoint", help="Required when backend=torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qnn-context-binary")
    parser.add_argument("--qnn-net-run")
    parser.add_argument("--qnn-backend-lib")
    parser.add_argument("--qnn-lib-dir")
    args = parser.parse_args()

    backend = _build_backend_from_args(args)
    runner = PolicyRunner(backend, args.normalizer)

    # Synthetic frame for the smoke test.
    rgb = np.random.rand(3, CONFIG.shape.image_size, CONFIG.shape.image_size).astype(np.float32)
    state = np.random.randn(CONFIG.shape.state_dim).astype(np.float32)

    action = runner.step(rgb, state)
    print(f"action shape       : {action.shape}")
    print(f"latency (last call): {runner.stats.last_inference_ms:.2f} ms")
    print(f"action[0]          : {action[0]}")


if __name__ == "__main__":
    main()
