"""Generate raw calibration tensors for QAIRT INT8 quantisation.

Reproduces the helper invoked by Listing 2 of the report.  We sample a small
balanced subset of the 240 demonstrations across the three handover tasks,
preprocess each frame the same way the policy expects, and dump the binary
buffers that ``qairt-quantizer`` consumes for percentile-based calibration.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np

from src.common.config import CONFIG
from src.training.dataset import HandoverChunkDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(CONFIG.tasks))
    parser.add_argument("--num-samples", type=int, default=96,
                        help="Total number of calibration frames across tasks.")
    parser.add_argument("--image-size", type=int, default=CONFIG.shape.image_size)
    parser.add_argument("--state-dim", type=int, default=CONFIG.shape.state_dim)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task = args.num_samples // len(args.tasks)
    counter = 0
    for task in args.tasks:
        ds = HandoverChunkDataset(args.dataset_root, task,
                                  action_horizon=CONFIG.shape.action_horizon)
        idxs = rng.choice(len(ds), size=min(per_task, len(ds)), replace=False)
        for j in idxs:
            sample = ds[int(j)]
            rgb = sample["rgb"].numpy().astype(np.float32)
            state = sample["robot_state"].numpy().astype(np.float32)
            assert rgb.shape == (3, args.image_size, args.image_size), rgb.shape
            assert state.shape == (args.state_dim,), state.shape

            rgb.tofile(out_dir / f"rgb_{counter:04d}.raw")
            state.tofile(out_dir / f"robot_state_{counter:04d}.raw")
            counter += 1

    print(f"[export_qnn_raw_inputs] wrote {counter} calibration samples to {out_dir}")


if __name__ == "__main__":
    main()
