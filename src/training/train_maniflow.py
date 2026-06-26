"""Train a ManiFlow handover policy on one of the three tasks.

This is the workstation-side entry point.  It implements the same overall
recipe documented in the final report:

    1. Build a task-specific dataset of 80 teleoperated demonstrations.
    2. Optimise a DiT-X style flow model with the joint flow + consistency
       objective (or, for the baseline, flow only).
    3. Save a checkpoint that the export script (``qualcomm/export_onnx.py``)
       can later load and convert to ONNX.

Run example::

    python -m src.training.train_maniflow \\
        --task cucumber \\
        --data-root ./data/handover_240 \\
        --output-dir ./checkpoints/cucumber \\
        --policy maniflow
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.config import CONFIG
from src.models import HandoverFlowMatchingBaseline, HandoverManiFlowPolicy
from src.training.dataset import HandoverChunkDataset, _list_episodes


def _build_policy(name: str) -> torch.nn.Module:
    shape = CONFIG.shape
    common = dict(
        image_size=shape.image_size,
        state_dim=shape.state_dim,
        action_dim=shape.action_dim,
        action_horizon=shape.action_horizon,
        inference_steps=CONFIG.train.inference_steps,
    )
    if name == "maniflow":
        return HandoverManiFlowPolicy(use_consistency_training=True, **common)
    if name == "fm_baseline":
        return HandoverFlowMatchingBaseline(**common)
    raise ValueError(f"Unknown policy '{name}'. Expected 'maniflow' or 'fm_baseline'.")


def _save_checkpoint(
    output_dir: str,
    name: str,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    normalizer_state: dict,
    extra: Optional[dict] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    payload = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "normalizer": normalizer_state,
        "policy_class": policy.__class__.__name__,
        "config": {
            "shape": CONFIG.shape.__dict__,
            "train": CONFIG.train.__dict__,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    # Also drop a small json so deployment scripts can read it without torch.
    with open(os.path.join(output_dir, "normalizer.json"), "w") as f:
        json.dump(normalizer_state, f, indent=2)
    return path


def _train_one_epoch(
    policy: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    policy.train()
    running = {"loss": 0.0, "loss_flow": 0.0, "loss_consistency": 0.0}
    n_batches = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        state = batch["robot_state"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)

        out = policy.compute_loss(
            rgb, state, action,
            flow_batch_ratio=CONFIG.train.flow_batch_ratio,
            consistency_batch_ratio=CONFIG.train.consistency_batch_ratio,
        )
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        for k in running:
            if k in out:
                running[k] += float(out[k].item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def _validate(policy, loader, device) -> float:
    policy.eval()
    total = 0.0
    n = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        state = batch["robot_state"].to(device)
        action = batch["action"].to(device)
        # 1-step deployment evaluation: forward() runs the integrator with
        # ``inference_steps`` Euler steps and returns the predicted chunk.
        pred = policy(rgb, state)
        total += float(torch.nn.functional.mse_loss(pred, action).item())
        n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(CONFIG.tasks))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=["maniflow", "fm_baseline"], default="maniflow")
    parser.add_argument("--batch-size", type=int, default=CONFIG.train.batch_size)
    parser.add_argument("--epochs", type=int, default=CONFIG.train.num_epochs)
    parser.add_argument("--lr", type=float, default=CONFIG.train.lr)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    torch.manual_seed(CONFIG.train.seed)

    # Split episodes (not windows) to avoid data leakage between train/val.
    episode_paths = _list_episodes(args.data_root, args.task)
    rng = np.random.default_rng(CONFIG.train.seed)
    rng.shuffle(episode_paths)
    n_val_ep = max(1, int(len(episode_paths) * CONFIG.train.val_ratio))
    train_paths = episode_paths[n_val_ep:]
    val_paths = episode_paths[:n_val_ep]

    train_dataset = HandoverChunkDataset(
        args.data_root, args.task,
        action_horizon=CONFIG.shape.action_horizon,
        episode_paths=train_paths,
    )
    val_dataset = HandoverChunkDataset(
        args.data_root, args.task,
        action_horizon=CONFIG.shape.action_horizon,
        episode_paths=val_paths,
        state_norm=train_dataset.state_norm,
        action_norm=train_dataset.action_norm,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    policy = _build_policy(args.policy).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=CONFIG.train.weight_decay,
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_log = _train_one_epoch(policy, train_loader, optimizer, device)
        val_loss = _validate(policy, val_loader, device)
        dt = time.time() - t0
        print(
            f"[{args.task}/{args.policy}] epoch={epoch:03d} "
            f"loss={train_log['loss']:.4f} flow={train_log['loss_flow']:.4f} "
            f"cons={train_log['loss_consistency']:.4f} val={val_loss:.4f} "
            f"time={dt:.1f}s"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_checkpoint(
                args.output_dir,
                name=f"{args.policy}_{args.task}_e{epoch:03d}.pt",
                policy=policy, optimizer=optimizer, epoch=epoch,
                normalizer_state=train_dataset.normalizer_state(),
            )

        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(
                args.output_dir,
                name=f"{args.policy}_{args.task}_best.pt",
                policy=policy, optimizer=optimizer, epoch=epoch,
                normalizer_state=train_dataset.normalizer_state(),
                extra={"best_val": best_val},
            )

    print(f"Done. best validation MSE = {best_val:.4f}")


if __name__ == "__main__":
    main()
