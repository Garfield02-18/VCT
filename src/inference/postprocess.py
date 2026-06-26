"""Pre/post-processing helpers shared between PyTorch eval and AIRbox runtime.

Keeping these helpers in one module is what allows the report's "export sanity
check" to work: the same code path normalises observations and de-normalises
predicted actions whether we are inside the training script, comparing PyTorch
and ONNX outputs, or running on the AIRbox NPU.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.common.config import CONFIG
from src.training.dataset import MinMaxNormalizer


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def preprocess_rgb(frame_hwc_uint8: np.ndarray, target_size: int = CONFIG.shape.image_size) -> np.ndarray:
    """Convert a HWC uint8 RGB frame to the CHW float32 [0,1] tensor the policy expects."""
    if frame_hwc_uint8.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB input from the camera driver.")
    if frame_hwc_uint8.ndim != 3 or frame_hwc_uint8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 input, got {frame_hwc_uint8.shape}.")
    h, w, _ = frame_hwc_uint8.shape
    if (h, w) != (target_size, target_size):
        # Centre-crop to a square then resample.  We do not depend on OpenCV
        # here so the same code can run on the AIRbox without extra wheels.
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        frame_hwc_uint8 = frame_hwc_uint8[top: top + side, left: left + side]
        frame_hwc_uint8 = _bilinear_resize_uint8(frame_hwc_uint8, target_size)

    rgb = frame_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))         # HWC -> CHW


def _bilinear_resize_uint8(img: np.ndarray, target: int) -> np.ndarray:
    """Tiny bilinear resampler so this module stays dependency-free."""
    h, w, _ = img.shape
    ys = np.linspace(0, h - 1, target)
    xs = np.linspace(0, w - 1, target)
    y0 = np.floor(ys).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    Ia = img[y0[:, None], x0[None, :]]
    Ib = img[y0[:, None], x1[None, :]]
    Ic = img[y1[:, None], x0[None, :]]
    Id = img[y1[:, None], x1[None, :]]

    top = Ia + wx * (Ib - Ia)
    bot = Ic + wx * (Id - Ic)
    out = top + wy * (bot - top)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# State / action normalisation
# ---------------------------------------------------------------------------


def normalize_state(state: np.ndarray, normalizer: MinMaxNormalizer) -> np.ndarray:
    return normalizer.normalize(state.astype(np.float32))


def denormalize_action_chunk(action_chunk_norm: np.ndarray,
                             normalizer: MinMaxNormalizer) -> np.ndarray:
    """Convert the policy output (normalised) back to physical joint commands."""
    if action_chunk_norm.shape[-1] != CONFIG.shape.action_dim:
        raise ValueError(f"Bad action dim: {action_chunk_norm.shape}")
    return normalizer.denormalize(action_chunk_norm.astype(np.float32))


# ---------------------------------------------------------------------------
# Receding horizon helper
# ---------------------------------------------------------------------------


def select_executable_steps(action_chunk: np.ndarray,
                            n_first: int = CONFIG.deploy.receding_horizon_first_n,
                            ) -> np.ndarray:
    """Return only the first ``n_first`` actions of the predicted chunk.

    The report's bridge code (Listing 7) always forwards the newest action,
    not the entire chunk: the receiving side should request a fresh
    prediction once those steps are consumed.
    """
    if n_first < 1 or n_first > action_chunk.shape[0]:
        raise ValueError(f"Bad n_first {n_first} for chunk {action_chunk.shape}")
    return action_chunk[:n_first]


# ---------------------------------------------------------------------------
# I/O glue used by the AIRbox-side runtime
# ---------------------------------------------------------------------------


def pack_inputs_for_qnn(rgb_chw: np.ndarray, state_14: np.ndarray) -> Dict[str, np.ndarray]:
    """Reshape numpy arrays so they match the .raw layout used by qnn-net-run."""
    if rgb_chw.shape != (3, CONFIG.shape.image_size, CONFIG.shape.image_size):
        raise ValueError(f"bad rgb shape {rgb_chw.shape}")
    if state_14.shape != (CONFIG.shape.state_dim,):
        raise ValueError(f"bad state shape {state_14.shape}")
    return {
        "rgb": rgb_chw.astype(np.float32, copy=False).reshape(*CONFIG.rgb_shape),
        "robot_state": state_14.astype(np.float32, copy=False).reshape(*CONFIG.state_shape),
    }


def unpack_qnn_action(out: np.ndarray) -> np.ndarray:
    """Reshape qnn-net-run output bytes into the [16, 14] action chunk."""
    return out.reshape(CONFIG.shape.action_horizon, CONFIG.shape.action_dim)


def split_packet(buf: bytes) -> Tuple[float, np.ndarray]:
    """Inverse of the bridge wire format: timestamp (8 bytes) + 14 float32."""
    import struct
    ts = struct.unpack("d", buf[:8])[0]
    arr = np.frombuffer(buf[8:8 + 4 * CONFIG.shape.action_dim], dtype=np.float32)
    return ts, arr.copy()
