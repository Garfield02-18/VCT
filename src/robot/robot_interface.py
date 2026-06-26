"""Thin abstraction over the bimanual handover platform.

The real platform exposes:
    * A static front-facing RGB camera (Intel RealSense D-series).
    * Two 7-DoF arms whose joint state is concatenated into the 14-dim
      ``robot_state`` vector consumed by the policy.
    * A high-level "execute action chunk" interface on the upper computer.

This file deliberately abstracts those three things into one class so the
rest of the project (evaluation script, deployment bridge) does not need to
know whether the rollout is happening on the physical robot or in the small
in-process ``DummyEnv`` we use for offline tests.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.common.config import CONFIG


@dataclass
class Observation:
    rgb: np.ndarray         # [3, 224, 224], float32 in [0, 1]
    state: np.ndarray       # [14], float32
    timestamp: float


class _RobotBackend(abc.ABC):
    """Common interface for real and simulated robot backends."""

    @abc.abstractmethod
    def read_rgb(self) -> np.ndarray: ...

    @abc.abstractmethod
    def read_state(self) -> np.ndarray: ...

    @abc.abstractmethod
    def execute_action_chunk(self, action_chunk: np.ndarray) -> Tuple[bool, str]: ...

    @abc.abstractmethod
    def reset_for_task(self, task: str) -> None: ...


class _RealRobotBackend(_RobotBackend):
    """Adapter for the real dual-xArm7 + Inspire-Hand platform.

    The actual driver code lives on the upper computer.  Here we only define
    the surface the rest of the codebase relies on - actual SDK imports are
    left as ``...`` placeholders so the file can also be imported on the
    AIRbox without the SDK installed.
    """

    def __init__(self) -> None:
        # In production these would import xarm.wrapper, realsense2, etc.
        self._rs_camera = None        # initialise RealSense pipeline here
        self._left_arm = None         # initialise left xArm controller
        self._right_arm = None        # initialise right xArm controller
        self._connected = False

    def connect(self) -> None:
        # connect to camera + arms, calibrate, home both arms.
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def read_rgb(self) -> np.ndarray:
        # Capture a frame from the calibrated camera, resize to 224 and
        # convert to CHW float32 in [0, 1].
        raise NotImplementedError("Wire to RealSense pipeline on the upper computer.")

    def read_state(self) -> np.ndarray:
        # Concatenate left/right arm joint angles.
        raise NotImplementedError("Wire to xArm SDK on the upper computer.")

    def execute_action_chunk(self, action_chunk: np.ndarray) -> Tuple[bool, str]:
        # Dispatch the predicted action to the arms and report whether the
        # current handover stage finished after this chunk.
        raise NotImplementedError("Wire to upper-computer command dispatcher.")

    def reset_for_task(self, task: str) -> None:
        # Move the arms to a task-compatible initial configuration.
        raise NotImplementedError("Wire to upper-computer reset routine.")


class _DummyEnv(_RobotBackend):
    """Minimal in-process stand-in used for offline tests.

    It is not meant to model handover physics; it just lets us exercise the
    evaluation control flow (observation -> policy -> action -> step) without
    any robot dependency.
    """

    def __init__(self) -> None:
        self.t = 0
        self.rng = np.random.default_rng(0)
        self.task = "cucumber"

    def reset_for_task(self, task: str) -> None:
        self.t = 0
        self.task = task

    def read_rgb(self) -> np.ndarray:
        rgb = self.rng.uniform(0.0, 1.0, size=(3, CONFIG.shape.image_size, CONFIG.shape.image_size))
        return rgb.astype(np.float32)

    def read_state(self) -> np.ndarray:
        return self.rng.normal(size=CONFIG.shape.state_dim).astype(np.float32)

    def execute_action_chunk(self, action_chunk: np.ndarray) -> Tuple[bool, str]:
        self.t += action_chunk.shape[0]
        # Pretend the rollout finishes after ~60 control steps.
        if self.t < 60:
            return False, "in_progress"
        # Coin-flip success so summary numbers are non-trivial in dry runs.
        succeed = self.rng.random() < 0.7
        return True, ("release_done" if succeed else "transfer")


class RobotInterface:
    """Public, stable wrapper used by the rest of the codebase."""

    def __init__(self, simulate: bool = False) -> None:
        self.simulate = simulate
        self._backend = _DummyEnv() if simulate else _RealRobotBackend()
        if not simulate:
            self._backend.connect()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def read_observation(self) -> Observation:
        rgb = self._backend.read_rgb()
        state = self._backend.read_state()
        return Observation(rgb=rgb, state=state, timestamp=time.time())

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def step(self, action_chunk: np.ndarray) -> Tuple[bool, Optional[str]]:
        return self._backend.execute_action_chunk(action_chunk)

    def reset_for_task(self, task: str) -> None:
        self._backend.reset_for_task(task)

    def shutdown(self) -> None:
        if not self.simulate:
            self._backend.disconnect()
