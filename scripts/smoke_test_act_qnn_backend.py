#!/usr/bin/env python3
"""Use the bundled ACT/QNN artifact as a Team14 PolicyRunner backend.

This is an integration smoke test, not a Team14 result reproduction. It adapts
an ACT-style QNN policy with IO

    image [1,3,480,640], qpos [1,14], task_embedding [1,32]
    -> actions [1,100,14]

to Team14's runner contract

    rgb_chw [3,224,224], state [14] -> action_chunk [16,14]

so the Team14 control/normalization/selection path can be exercised with a real
QNN/HTP context binary even when Team14's own QNN deployment artifacts are not
available.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import CONFIG
from src.inference.run_policy import PolicyRunner, _Backend
from src.robot.safety_checks import SafetyMonitor


DEFAULT_BUNDLE = PROJECT_ROOT / "assets" / "qnn_friendly_policy_bundle"
DEFAULT_QAIRT = Path("/home/radxa/qairt/2.47.1")
DEFAULT_ARCH = "aarch64-oe-linux-gcc11.2"


class ActQnnBackend(_Backend):
    def __init__(
        self,
        bundle_dir: Path,
        qairt_root: Path,
        arch: str,
        task_embedding_path: Path | None = None,
        work_dir: Path | None = None,
        use_sample_image: bool = False,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.qairt_root = qairt_root
        self.arch = arch
        self.task_embedding_path = task_embedding_path or bundle_dir / "sample_inputs" / "task_embedding.raw"
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="team14_act_qnn_"))
        self.use_sample_image = use_sample_image

        self.bin_dir = qairt_root / "bin" / arch
        self.lib_dir = qairt_root / "lib" / arch
        self.hexagon_dir = qairt_root / "lib" / "hexagon-v73" / "unsigned"
        self.qnn_net_run = self.bin_dir / "qnn-net-run"
        self.backend_lib = self.lib_dir / "libQnnHtp.so"
        self.context_file = bundle_dir / "htp_context" / "qnn_friendly_policy_htp.bin"
        self.sample_image = bundle_dir / "sample_inputs" / "image.raw"

        self._validate_paths()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _validate_paths(self) -> None:
        required = [
            self.qnn_net_run,
            self.backend_lib,
            self.context_file,
            self.task_embedding_path,
        ]
        if self.use_sample_image:
            required.append(self.sample_image)
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("missing required ACT/QNN files:\n" + "\n".join(missing))

    @staticmethod
    def _resize_224_to_480x640(rgb_chw: np.ndarray) -> np.ndarray:
        """Nearest-neighbor resize from Team14 CHW 224x224 to ACT CHW 480x640."""
        if rgb_chw.shape != (3, CONFIG.shape.image_size, CONFIG.shape.image_size):
            raise ValueError(f"bad Team14 rgb shape {rgb_chw.shape}")
        y_idx = np.linspace(0, CONFIG.shape.image_size - 1, 480).round().astype(np.int64)
        x_idx = np.linspace(0, CONFIG.shape.image_size - 1, 640).round().astype(np.int64)
        return rgb_chw[:, y_idx][:, :, x_idx].astype(np.float32, copy=False)

    def _write_inputs(self, rgb_chw: np.ndarray, state_14: np.ndarray) -> Path:
        if state_14.shape != (CONFIG.shape.state_dim,):
            raise ValueError(f"bad state shape {state_14.shape}")

        image_path = self.work_dir / "image.raw"
        qpos_path = self.work_dir / "qpos.raw"
        task_path = self.task_embedding_path
        input_list = self.work_dir / "input_list.txt"

        if self.use_sample_image:
            image_path = self.sample_image
        else:
            image = self._resize_224_to_480x640(rgb_chw)
            image.reshape(1, 3, 480, 640).astype(np.float32).tofile(image_path)

        state_14.reshape(1, CONFIG.shape.state_dim).astype(np.float32).tofile(qpos_path)
        input_list.write_text(f"image:={image_path} qpos:={qpos_path} task_embedding:={task_path}\n")
        return input_list

    def __call__(self, rgb_chw: np.ndarray, state_14: np.ndarray) -> np.ndarray:
        input_list = self._write_inputs(rgb_chw, state_14.astype(np.float32))
        output_dir = self.work_dir / f"qnn_output_{int(time.time() * 1000)}"
        output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{self.lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        # HTP skel lookup expects semicolon-separated ADSP paths on this platform.
        env["ADSP_LIBRARY_PATH"] = f"{self.hexagon_dir};{self.lib_dir}"

        cmd = [
            str(self.qnn_net_run),
            "--backend", str(self.backend_lib),
            "--retrieve_context", str(self.context_file),
            "--input_list", str(input_list),
            "--output_dir", str(output_dir),
            "--use_native_input_files",
            "--use_native_output_files",
            "--num_inferences", "1",
            "--keep_num_outputs", "1",
            "--perf_profile", "burst",
            "--log_level", "error",
        ]
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        action_files = sorted(output_dir.glob("**/actions.raw"))
        if not action_files:
            raise FileNotFoundError(f"qnn-net-run produced no actions.raw under {output_dir}")
        raw = np.fromfile(action_files[0], dtype=np.float32)
        if raw.size != 1 * 100 * CONFIG.shape.action_dim:
            raise ValueError(f"unexpected ACT action float count {raw.size}")
        actions_100 = raw.reshape(100, CONFIG.shape.action_dim)
        return actions_100[: CONFIG.shape.action_horizon]


def make_identity_normalizer(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.write_text(json.dumps(normalizer, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--qairt-root", type=Path, default=DEFAULT_QAIRT)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--task-embedding", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "tmp" / "act_qnn_team14")
    parser.add_argument("--use-sample-image", action="store_true", help="Use ACT bundle sample image instead of resizing Team14 synthetic RGB.")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    normalizer_path = make_identity_normalizer(args.work_dir / "team14_identity_normalizer.json")

    backend = ActQnnBackend(
        bundle_dir=args.bundle_dir,
        qairt_root=args.qairt_root,
        arch=args.arch,
        task_embedding_path=args.task_embedding,
        work_dir=args.work_dir / "qnn_run",
        use_sample_image=args.use_sample_image,
    )
    runner = PolicyRunner(backend, str(normalizer_path), first_n_executable=CONFIG.deploy.receding_horizon_first_n)

    # Team14-shaped synthetic observation. The backend adapts it to ACT/QNN IO.
    rgb = np.random.rand(3, CONFIG.shape.image_size, CONFIG.shape.image_size).astype(np.float32)
    state = np.zeros(CONFIG.shape.state_dim, dtype=np.float32)

    action = runner.step(rgb, state)
    ok, reason = SafetyMonitor().check(action, state)
    print(f"ACTION_SHAPE={action.shape}")
    print(f"LATENCY_MS={runner.stats.last_inference_ms:.2f}")
    print("ACTION_FIRST14=[" + ", ".join(f"{v:.6g}" for v in action[0]) + "]")
    print(f"SAFETY_OK={ok} REASON={reason}")
    print(f"WORK_DIR={args.work_dir}")

    if not np.isfinite(action).all():
        raise SystemExit("non-finite action output")
    if action.shape != (CONFIG.deploy.receding_horizon_first_n, CONFIG.shape.action_dim):
        raise SystemExit(f"bad final action shape {action.shape}")
    print("[done] Team14 PolicyRunner successfully used ACT/QNN backend")


if __name__ == "__main__":
    main()
