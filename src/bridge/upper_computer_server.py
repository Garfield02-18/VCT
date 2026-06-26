"""Upper-computer side of the bridge.

The AIRbox sends action packets over UDP; this server runs on the workstation
that drives the arms and is responsible for:

    * dropping stale packets (older than ``freshness_window_s``);
    * dropping packets that fail the safety check;
    * forwarding the newest valid action to the robot interface.

The "newest packet wins" policy is the one described in the report's
"Communication with the Upper Computer" section: if the AIRbox produces a
new chunk after the scene has changed, old packets must not accumulate in a
queue and drive the robot from outdated observations.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from src.common.config import CONFIG
from src.inference.postprocess import split_packet
from src.robot.robot_interface import RobotInterface
from src.robot.safety_checks import SafetyMonitor


@dataclass
class ServerStats:
    received: int = 0
    accepted: int = 0
    dropped_stale: int = 0
    dropped_unsafe: int = 0
    last_action_ts: float = 0.0
    history: Deque[Tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=64))


class UpperComputerServer:
    def __init__(
        self,
        bind_ip: str,
        bind_port: int,
        robot: RobotInterface,
        safety: Optional[SafetyMonitor] = None,
        freshness_window_s: float = CONFIG.deploy.action_freshness_window_s,
    ) -> None:
        self.bind = (bind_ip, bind_port)
        self.robot = robot
        self.safety = safety or SafetyMonitor()
        self.freshness_window_s = freshness_window_s
        self.stats = ServerStats()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self.bind)
        self._sock.settimeout(0.1)
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _consume_one(self) -> Optional[Tuple[float, np.ndarray]]:
        try:
            data, _ = self._sock.recvfrom(4096)
        except socket.timeout:
            return None
        try:
            return split_packet(data)
        except Exception as e:                       # corrupted packet
            print(f"[server] dropping malformed packet: {e}", file=sys.stderr)
            return None

    def _newest_in_buffer(self) -> Optional[Tuple[float, np.ndarray]]:
        """Drain the OS receive buffer and return only the newest packet."""
        latest: Optional[Tuple[float, np.ndarray]] = None
        while True:
            pkt = self._consume_one()
            if pkt is None:
                break
            self.stats.received += 1
            if latest is None or pkt[0] > latest[0]:
                if latest is not None:
                    self.stats.dropped_stale += 1   # discarded by "newest wins"
                latest = pkt
            else:
                self.stats.dropped_stale += 1
        return latest

    # ------------------------------------------------------------------
    def serve_forever(self) -> None:
        print(f"[server] listening on {self.bind[0]}:{self.bind[1]}")
        while not self._stop.is_set():
            pkt = self._newest_in_buffer()
            if pkt is None:
                continue
            ts, action = pkt
            age = time.time() - ts
            if age > self.freshness_window_s:
                self.stats.dropped_stale += 1
                continue

            obs = self.robot.read_observation()
            ok, reason = self.safety.check(action[None, :], obs.state, action_timestamp=ts)
            if not ok:
                self.stats.dropped_unsafe += 1
                print(f"[server] dropping unsafe action: {reason}", file=sys.stderr)
                continue

            self.robot.step(action[None, :])
            self.stats.accepted += 1
            self.stats.last_action_ts = ts
            self.stats.history.append((ts, action.copy()))

    def close(self) -> None:
        self._sock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=CONFIG.deploy.upper_computer_port)
    parser.add_argument("--simulate", action="store_true",
                        help="Use the in-process DummyEnv robot backend.")
    args = parser.parse_args()

    robot = RobotInterface(simulate=args.simulate)
    server = UpperComputerServer(args.bind_ip, args.bind_port, robot)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopping on Ctrl-C")
        s = server.stats
        print(
            f"[server] stats: received={s.received} accepted={s.accepted} "
            f"dropped_stale={s.dropped_stale} dropped_unsafe={s.dropped_unsafe}"
        )
    finally:
        server.close()
        robot.shutdown()


if __name__ == "__main__":
    main()
