"""AIRbox Q900 side of the bridge.

This is the script that runs on the edge device.  It is the production-ready
version of Listing 7 in the report:

    1. Read the latest synchronised observation (RGB + 14-d robot state).
    2. Run one-step inference through the QNN context binary.
    3. Take the first action of the predicted chunk and ship it to the upper
       computer over UDP, with a timestamp so the upper computer can drop
       stale packets.
    4. Loop forever, *not* queuing predictions: the upper computer always
       gets the newest action, never an old one.

The matching server lives in ``upper_computer_server.py``.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.common.config import CONFIG
from src.inference.postprocess import preprocess_rgb
from src.inference.run_policy import PolicyRunner, QnnContextBinaryBackend, TorchBackend


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def encode_packet(action_14d: np.ndarray, timestamp: float) -> bytes:
    """Wire format: [timestamp:f64][action_14:f32]. Mirrors Listing 7."""
    if action_14d.shape != (CONFIG.shape.action_dim,):
        raise ValueError(f"expected ({CONFIG.shape.action_dim},), got {action_14d.shape}")
    return struct.pack("d", timestamp) + action_14d.astype(np.float32).tobytes()


# ---------------------------------------------------------------------------
# Observation source
# ---------------------------------------------------------------------------


@dataclass
class _Frame:
    rgb: np.ndarray   # CHW float32 in [0,1]
    state: np.ndarray # 14, float32


class ObservationSource:
    """Read camera + arm state.  Backed by either real hardware or a stub.

    On the AIRbox we usually receive observations *from* the upper computer
    via a small TCP/RealSense bridge -- this class hides that detail so the
    main loop reads cleanly.
    """

    def __init__(self, simulate: bool = False) -> None:
        self.simulate = simulate
        if not simulate:
            # In production, initialise camera grab + state subscription here.
            self._rng = None
        else:
            self._rng = np.random.default_rng(0)

    def latest(self) -> _Frame:
        if self.simulate:
            assert self._rng is not None
            rgb = self._rng.uniform(0, 255, size=(224, 224, 3)).astype(np.uint8)
            return _Frame(
                rgb=preprocess_rgb(rgb),
                state=self._rng.normal(size=CONFIG.shape.state_dim).astype(np.float32),
            )
        raise NotImplementedError("Wire to RealSense + arm-state subscriber on the AIRbox.")


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------


class ActionSender:
    def __init__(self, ip: str, port: int) -> None:
        self.addr = (ip, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, action_14d: np.ndarray) -> None:
        pkt = encode_packet(action_14d, time.time())
        self._sock.sendto(pkt, self.addr)

    def close(self) -> None:
        self._sock.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _build_runner(args: argparse.Namespace) -> PolicyRunner:
    if args.backend == "qnn":
        backend = QnnContextBinaryBackend(
            context_binary=args.qnn_context_binary,
            qnn_net_run=args.qnn_net_run,
            backend_lib=args.qnn_backend_lib,
            lib_dir=args.qnn_lib_dir,
        )
    elif args.backend == "torch":
        backend = TorchBackend(args.torch_checkpoint, device=args.device)
    else:
        raise ValueError(args.backend)
    return PolicyRunner(backend, args.normalizer,
                        first_n_executable=CONFIG.deploy.receding_horizon_first_n)


def run_loop(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    sender = ActionSender(args.upper_ip, args.upper_port)
    obs_source = ObservationSource(simulate=args.simulate)

    print(f"[board_client] sending to {args.upper_ip}:{args.upper_port}")
    print(f"[board_client] backend={args.backend}")
    try:
        while True:
            frame = obs_source.latest()
            executable = runner.step(frame.rgb, frame.state)
            # The bridge always forwards the newest single action, even if the
            # policy produced a chunk of length 16: stale packets must not be
            # allowed to drive the robot (see report, "Communication with the
            # Upper Computer").
            sender.send(executable[0])

            if runner.stats.n_calls % 50 == 0:
                print(
                    f"[board_client] step={runner.stats.n_calls} "
                    f"latency_last={runner.stats.last_inference_ms:.1f}ms "
                    f"latency_avg={runner.stats.avg_inference_ms:.1f}ms"
                )
    except KeyboardInterrupt:
        print("\n[board_client] stopping on Ctrl-C")
    finally:
        sender.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qnn", "torch"], default="qnn")
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--upper-ip", default=CONFIG.deploy.upper_computer_ip)
    parser.add_argument("--upper-port", type=int, default=CONFIG.deploy.upper_computer_port)
    parser.add_argument("--simulate", action="store_true",
                        help="Use random observations instead of the real camera.")
    # qnn backend
    parser.add_argument("--qnn-context-binary",
                        default="/home/radxa/maniflow_q900/model/maniflow_handover_1step.bin")
    parser.add_argument("--qnn-net-run",
                        default="/home/radxa/maniflow_q900/qnn-net-run")
    parser.add_argument("--qnn-backend-lib",
                        default="/home/radxa/maniflow_q900/lib/libQnnHtp.so")
    parser.add_argument("--qnn-lib-dir",
                        default="/home/radxa/maniflow_q900/lib")
    # torch fallback (only used during sanity checks on the workstation)
    parser.add_argument("--torch-checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.backend == "torch" and not args.torch_checkpoint:
        print("--torch-checkpoint is required when backend=torch", file=sys.stderr)
        sys.exit(2)

    run_loop(args)


if __name__ == "__main__":
    main()
