"""Teleoperation dataset for the bimanual handover reproduction.

Each task (cucumber / pepper / banana) has its own folder under
``data_root/<task>/`` containing 80 demonstrations.  A demonstration is a
single zarr file (or, for raw recordings, a directory of synchronised images +
state/action npy files) covering a complete handover episode.  The dataset
slides a fixed-length window of length ``H = action_horizon`` across the
episode and produces the same ``(rgb, robot_state, action_chunk)`` tuple that
the policy and the ONNX export both consume.

This mirrors ``maniflow/dataset/robotwin_image_dataset.py`` from the official
Maniflow code, simplified for our single-task / single-camera setting.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common.config import CONFIG, HANDOVER_TASKS


# ---------------------------------------------------------------------------
# Episode loader
# ---------------------------------------------------------------------------


@dataclass
class HandoverEpisode:
    rgb: np.ndarray             # [T, 3, 224, 224], float32, [0,1]
    state: np.ndarray           # [T, 14], float32, normalised
    action: np.ndarray          # [T, 14], float32, normalised
    task: str

    def __len__(self) -> int:
        return self.rgb.shape[0]


def _load_zarr_episode(path: str, task: str) -> HandoverEpisode:
    """Load a single recorded handover episode.

    The training side stores demonstrations as zarr arrays with three
    top-level keys: ``rgb``, ``state``, ``action``.  We deliberately keep this
    loader thin: synchronisation, normalisation and chunking happen here so
    the rest of the training code does not need to know the on-disk layout.
    """
    import zarr  # type: ignore

    z = zarr.open(path, mode="r")
    rgb = np.asarray(z["rgb"][:], dtype=np.float32) / 255.0
    if rgb.ndim == 4 and rgb.shape[-1] == 3:           # T,H,W,C -> T,C,H,W
        rgb = np.transpose(rgb, (0, 3, 1, 2))
    state = np.asarray(z["state"][:], dtype=np.float32)
    action = np.asarray(z["action"][:], dtype=np.float32)
    return HandoverEpisode(rgb=rgb, state=state, action=action, task=task)


def _list_episodes(data_root: str, task: str) -> List[str]:
    pattern = os.path.join(data_root, task, "*.zarr")
    files = sorted(glob.glob(pattern))
    if not files:
        # Allow the directory layout used by the recording tool: each demo
        # is a subfolder rather than a single .zarr.
        files = sorted(
            d for d in glob.glob(os.path.join(data_root, task, "*"))
            if os.path.isdir(d)
        )
    return files


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


class MinMaxNormalizer:
    """Per-dimension min/max scaler stored alongside the checkpoint."""

    def __init__(self, low: np.ndarray, high: np.ndarray, eps: float = 1e-6) -> None:
        self.low = low.astype(np.float32)
        self.high = high.astype(np.float32)
        self.scale = np.maximum(self.high - self.low, eps)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.low) / self.scale * 2.0 - 1.0

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x + 1.0) / 2.0 * self.scale + self.low

    def to_dict(self) -> Dict[str, list]:
        return {"low": self.low.tolist(), "high": self.high.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Sequence[float]]) -> "MinMaxNormalizer":
        return cls(np.asarray(d["low"], dtype=np.float32),
                   np.asarray(d["high"], dtype=np.float32))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HandoverChunkDataset(Dataset):
    """Sliding-window dataset over one handover task."""

    def __init__(
        self,
        data_root: str,
        task: str,
        action_horizon: int = CONFIG.shape.action_horizon,
        max_episodes: Optional[int] = None,
        episode_paths: Optional[List[str]] = None,
        state_norm: Optional[MinMaxNormalizer] = None,
        action_norm: Optional[MinMaxNormalizer] = None,
    ) -> None:
        if task not in HANDOVER_TASKS:
            raise ValueError(f"Unknown task '{task}'. Expected one of {HANDOVER_TASKS}.")
        self.task = task
        self.action_horizon = action_horizon

        if episode_paths is None:
            episode_paths = _list_episodes(data_root, task)
        if max_episodes is not None:
            episode_paths = episode_paths[:max_episodes]
        if not episode_paths:
            raise FileNotFoundError(
                f"No demonstrations found for task '{task}' under {data_root}."
            )

        self.episodes: List[HandoverEpisode] = [
            _load_zarr_episode(p, task) for p in episode_paths
        ]

        # Build list of (episode_idx, start_idx) so that __len__/__getitem__
        # iterate over fixed-length action chunks.
        self.windows: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            last = max(0, len(ep) - action_horizon)
            for start in range(last + 1):
                self.windows.append((ep_idx, start))

        if state_norm is not None and action_norm is not None:
            self.state_norm = state_norm
            self.action_norm = action_norm
        else:
            self.state_norm, self.action_norm = self._fit_normalizers()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _fit_normalizers(self) -> Tuple[MinMaxNormalizer, MinMaxNormalizer]:
        all_state = np.concatenate([ep.state for ep in self.episodes], axis=0)
        all_action = np.concatenate([ep.action for ep in self.episodes], axis=0)
        return (
            MinMaxNormalizer(all_state.min(axis=0), all_state.max(axis=0)),
            MinMaxNormalizer(all_action.min(axis=0), all_action.max(axis=0)),
        )

    # ------------------------------------------------------------------
    # Torch protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start = self.windows[idx]
        ep = self.episodes[ep_idx]
        end = start + self.action_horizon

        rgb_t = ep.rgb[start]                                    # current frame
        state_t = self.state_norm.normalize(ep.state[start])
        action_chunk = self.action_norm.normalize(ep.action[start:end])

        # Pad if the episode ends inside this window.
        if action_chunk.shape[0] < self.action_horizon:
            pad = self.action_horizon - action_chunk.shape[0]
            action_chunk = np.concatenate(
                [action_chunk, np.repeat(action_chunk[-1:], pad, axis=0)], axis=0
            )

        return {
            "rgb": torch.from_numpy(rgb_t),
            "robot_state": torch.from_numpy(state_t),
            "action": torch.from_numpy(action_chunk.astype(np.float32)),
            "task": self.task,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def normalizer_state(self) -> Dict[str, Dict[str, list]]:
        return {
            "state": self.state_norm.to_dict(),
            "action": self.action_norm.to_dict(),
        }


def build_handover_datasets(
    data_root: str,
    tasks: Sequence[str] = HANDOVER_TASKS,
    action_horizon: int = CONFIG.shape.action_horizon,
) -> Dict[str, HandoverChunkDataset]:
    """Build one dataset per task. We train task-specific models per the report."""
    return {
        task: HandoverChunkDataset(data_root, task, action_horizon=action_horizon)
        for task in tasks
    }
