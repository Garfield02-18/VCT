"""Export a trained handover policy from PyTorch to ONNX (Listing 1 of report).

This is the workstation-side export wrapper.  It loads the best checkpoint
produced by ``src/training/train_maniflow.py``, wraps the model so that
``forward(rgb, robot_state) -> action_chunk`` is fully deterministic from the
external inputs (no Python-side noise sampling), and writes the ONNX file
that ``qairt_convert.sh`` later turns into a Q900 context binary.

Tensor contract (verbatim from the report appendix):
    rgb         : 1 x 3 x 224 x 224, float32
    robot_state : 1 x 14, float32
    action_chunk: 1 x 16 x 14, float32
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from src.common.config import CONFIG
from src.models import HandoverFlowMatchingBaseline, HandoverManiFlowPolicy


def _build_policy(payload: dict) -> torch.nn.Module:
    cls = payload.get("policy_class", "HandoverManiFlowPolicy")
    common = dict(
        image_size=CONFIG.shape.image_size,
        state_dim=CONFIG.shape.state_dim,
        action_dim=CONFIG.shape.action_dim,
        action_horizon=CONFIG.shape.action_horizon,
        inference_steps=CONFIG.train.inference_steps,
    )
    if cls == "HandoverFlowMatchingBaseline":
        return HandoverFlowMatchingBaseline(**common)
    return HandoverManiFlowPolicy(use_consistency_training=True, **common)


def _verify_export(onnx_path: str, policy: torch.nn.Module) -> None:
    """Run a dummy forward pass through onnxruntime and compare to PyTorch.

    The report's "Export sanity checks" paragraph stresses that silent
    mismatches in tensor ordering or scale can look like a policy failure
    even when the model is correct, so we always run this check after
    exporting.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export_onnx] onnxruntime not available, skipping verify step.")
        return

    rgb = np.random.rand(*CONFIG.rgb_shape).astype(np.float32)
    state = np.random.randn(*CONFIG.state_shape).astype(np.float32)

    with torch.no_grad():
        ref = policy(torch.from_numpy(rgb), torch.from_numpy(state)).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(["action_chunk"], {"rgb": rgb, "robot_state": state})[0]

    diff = np.abs(out - ref).max()
    print(f"[export_onnx] max |onnx - torch| = {diff:.3e}")
    if diff > 1e-3:
        print("[export_onnx] WARNING: large mismatch between PyTorch and ONNX outputs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/maniflow_handover_best.pt")
    parser.add_argument("--onnx-path", default="deploy/onnx/maniflow_handover_1step.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    Path(os.path.dirname(args.onnx_path)).mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu")
    policy = _build_policy(payload)
    policy.load_state_dict(payload["model"])
    policy.eval()

    rgb = torch.randn(*CONFIG.rgb_shape, dtype=torch.float32)
    robot_state = torch.randn(*CONFIG.state_shape, dtype=torch.float32)

    torch.onnx.export(
        policy,
        (rgb, robot_state),
        args.onnx_path,
        input_names=["rgb", "robot_state"],
        output_names=["action_chunk"],
        dynamic_axes=None,                 # fixed shapes for QAIRT conversion
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[export_onnx] wrote {args.onnx_path}")
    print(
        f"[export_onnx] inputs: rgb {CONFIG.rgb_shape}, "
        f"robot_state {CONFIG.state_shape}; "
        f"output: action_chunk {CONFIG.action_shape}"
    )

    if not args.no_verify:
        _verify_export(args.onnx_path, policy)


if __name__ == "__main__":
    main()
