"""Lightweight safety layer between the policy output and the robot driver.

The report makes the case that the upper computer (not the inference device)
should keep the final say over what is dispatched to the arms.  This module
implements the cheap, always-on checks that the upper computer applies before
forwarding an action chunk:

    * Joint command stays within the configured operational range.
    * Per-step joint deltas stay below a velocity ceiling.
    * The action packet is reasonably fresh.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from src.common.config import CONFIG


# Per-joint operational range for the dual xArm7 platform (radians).  The
# real values come from the manufacturer datasheet; the conservative numbers
# below are used as a safe default for rollouts.
_JOINT_LOW = np.array([-2.5] * 7 + [-2.5] * 7, dtype=np.float32)
_JOINT_HIGH = np.array([2.5] * 7 + [2.5] * 7, dtype=np.float32)
_MAX_STEP_DELTA = np.array([0.20] * 14, dtype=np.float32)   # rad / control step


@dataclass
class SafetyConfig:
    joint_low: np.ndarray = field(default_factory=lambda: _JOINT_LOW.copy())
    joint_high: np.ndarray = field(default_factory=lambda: _JOINT_HIGH.copy())
    max_step_delta: np.ndarray = field(default_factory=lambda: _MAX_STEP_DELTA.copy())
    freshness_window_s: float = CONFIG.deploy.action_freshness_window_s


class SafetyMonitor:
    def __init__(self, cfg: Optional[SafetyConfig] = None) -> None:
        self.cfg = cfg or SafetyConfig()

    def check(
        self,
        action_chunk: np.ndarray,
        current_state: np.ndarray,
        action_timestamp: Optional[float] = None,
    ) -> Tuple[bool, str]:
        if action_chunk.ndim != 2 or action_chunk.shape[1] != 14:
            return False, f"unexpected action shape {action_chunk.shape}"

        # 1. Range check.
        below = (action_chunk < self.cfg.joint_low).any(axis=1)
        above = (action_chunk > self.cfg.joint_high).any(axis=1)
        if below.any() or above.any():
            return False, "action out of joint range"

        # 2. Per-step delta check.  Walk through the chunk starting from the
        # current measured state.
        prev = current_state.astype(np.float32)
        for i in range(action_chunk.shape[0]):
            delta = np.abs(action_chunk[i] - prev)
            if (delta > self.cfg.max_step_delta).any():
                return False, f"action delta exceeds limit at step {i}"
            prev = action_chunk[i]

        # 3. Freshness.  Stale packets are dropped by the bridge already, but
        # we double-check here so the upper computer never executes a stale
        # plan even if the bridge is misconfigured.
        if action_timestamp is not None:
            age = time.time() - action_timestamp
            if age > self.cfg.freshness_window_s:
                return False, f"stale action packet (age={age:.3f}s)"

        return True, "ok"
