"""ManiFlow-style policy used for the bimanual handover reproduction.

This module follows the architecture summarised in the final report (Figure 3):
a 2D image encoder + state encoder produce conditioning tokens for a DiT-X style
flow-matching transformer, trained jointly with a continuous-time consistency
objective so that one-step inference still yields useful action chunks.

The implementation mirrors the spirit of Maniflow's
``maniflow/policy/maniflow_image_policy.py`` and
``maniflow/model/diffusion/ditx.py`` but is intentionally compact: the goal of
this file is to give the rest of the project (export, board runtime, bridge)
a stable, ONNX-friendly module with the same I/O contract that we used during
real-robot evaluation.

Tensor contract (matches Listing 1 of the report):
    rgb         : float32, shape [B, 3, 224, 224]
    robot_state : float32, shape [B, 14]
    action_chunk: float32, shape [B, 16, 14]
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """Lightweight ResNet-style encoder.

    The course report uses a TIMM image encoder during training. For ONNX export
    and on-device inference we keep a small, fully-convolutional path with the
    same output dimensionality (1024) so that the DiT-X conditioning tokens
    can be reused without changes.
    """

    def __init__(self, out_dim: int = 1024) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.stage1 = self._stage(32, 64)
        self.stage2 = self._stage(64, 128)
        self.stage3 = self._stage(128, 256)
        self.stage4 = self._stage(256, 512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, out_dim),
        )
        self.out_dim = out_dim

    @staticmethod
    def _stage(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.stem(rgb)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int = 14, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(dtype=t.dtype, device=t.device)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class DiTBlock(nn.Module):
    """Simplified DiT-X block: cross-attention against (image+state+time) cond."""

    def __init__(self, dim: int, n_head: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + a
        h = self.norm2(x)
        a, _ = self.cross_attn(h, cond, cond, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm3(x))
        return x


class DiTXTrunk(nn.Module):
    """Action-token transformer trunk.

    Inputs:
        action_tokens : [B, H, dim]       (current latent action chunk)
        cond_tokens   : [B, K, dim]       (image + state + time)
    Output:
        velocity_pred : [B, H, action_dim]
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        horizon: int,
        n_layer: int = 6,
        n_head: int = 8,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(action_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, n_head) for _ in range(n_layer)])
        self.norm_out = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, action_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, action_tokens: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(action_tokens) + self.pos_embed
        for block in self.blocks:
            x = block(x, cond_tokens)
        return self.out_proj(self.norm_out(x))


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class HandoverManiFlowPolicy(nn.Module):
    """One-step ManiFlow handover policy.

    Trained with the joint flow-matching + consistency objective (handled in
    ``src.training.train_maniflow``). At inference time we run a fixed,
    typically one-step Euler integrator from a Gaussian prior to the predicted
    action chunk, which matches the deployment regime described in the report.
    """

    def __init__(
        self,
        image_size: int = 224,
        state_dim: int = 14,
        action_dim: int = 14,
        action_horizon: int = 16,
        inference_steps: int = 1,
        use_consistency_training: bool = True,
        embed_dim: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
    ) -> None:
        super().__init__()
        assert image_size == 224, "Export script assumes 224x224 RGB input."
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.inference_steps = inference_steps
        self.use_consistency_training = use_consistency_training

        self.image_encoder = ImageEncoder(out_dim=1024)
        self.state_encoder = StateEncoder(state_dim, out_dim=embed_dim)
        self.time_encoder = TimeEmbedding(embed_dim)

        # Project image/state/time to common dim and form a small token set.
        self.img_proj = nn.Linear(1024, embed_dim)
        self.trunk = DiTXTrunk(
            dim=embed_dim,
            action_dim=action_dim,
            horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_cond_tokens(
        self,
        rgb: torch.Tensor,
        robot_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        img_feat = self.img_proj(self.image_encoder(rgb))         # [B, D]
        state_feat = self.state_encoder(robot_state)              # [B, D]
        time_feat = self.time_encoder(t)                          # [B, D]
        return torch.stack([img_feat, state_feat, time_feat], dim=1)

    def velocity(
        self,
        action_tokens: torch.Tensor,
        rgb: torch.Tensor,
        robot_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._build_cond_tokens(rgb, robot_state, t)
        return self.trunk(action_tokens, cond)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        robot_state: torch.Tensor,
        num_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate the learned flow from t=1 (noise) to t=0 (action chunk)."""
        bsz = rgb.shape[0]
        steps = num_steps if num_steps is not None else self.inference_steps
        if noise is None:
            noise = torch.randn(
                bsz,
                self.action_horizon,
                self.action_dim,
                device=rgb.device,
                dtype=rgb.dtype,
            )
        x = noise
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((bsz,), 1.0 - k * dt, device=rgb.device, dtype=rgb.dtype)
            v = self.velocity(x, rgb, robot_state, t)
            x = x - dt * v
        return x

    def forward(self, rgb: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """Deterministic one-step forward pass (also used for ONNX export).

        Runs the Euler integrator from a fixed zero-noise initial state so
        the graph is fully deterministic from external inputs.  For stochastic
        rollouts with Gaussian noise, call ``sample()`` directly instead.
        """
        bsz = rgb.shape[0]
        zero = torch.zeros(
            bsz, self.action_horizon, self.action_dim,
            device=rgb.device, dtype=rgb.dtype,
        )
        return self.sample(rgb, robot_state, num_steps=self.inference_steps, noise=zero)

    # ------------------------------------------------------------------
    # Loss (training time)
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        rgb: torch.Tensor,
        robot_state: torch.Tensor,
        action_gt: torch.Tensor,
        flow_batch_ratio: float = 0.75,
        consistency_batch_ratio: float = 0.25,
    ) -> dict:
        """Joint flow-matching + consistency loss.

        Reference: report Figure 3, ManiFlow paper Section 3.  We mix a
        flow-matching term on a random batch fraction and a consistency term
        on the remainder, following the same recipe as Maniflow.
        """
        assert abs(flow_batch_ratio + consistency_batch_ratio - 1.0) < 1e-6
        B = rgb.shape[0]
        n_flow = int(round(B * flow_batch_ratio))
        n_cons = B - n_flow

        device = rgb.device
        action_gt = action_gt.to(device)

        loss_flow = rgb.new_zeros(())
        loss_cons = rgb.new_zeros(())

        # ---- Flow matching ------------------------------------------------
        if n_flow > 0:
            x1 = action_gt[:n_flow]
            x0 = torch.randn_like(x1)
            t = torch.rand(n_flow, device=device)
            xt = (1.0 - t)[:, None, None] * x1 + t[:, None, None] * x0
            target = x0 - x1                       # straight-path velocity
            v_pred = self.velocity(xt, rgb[:n_flow], robot_state[:n_flow], t)
            loss_flow = F.mse_loss(v_pred, target)

        # ---- Consistency -------------------------------------------------
        if n_cons > 0 and self.use_consistency_training:
            x1 = action_gt[n_flow:]
            x0 = torch.randn_like(x1)
            t = torch.rand(n_cons, device=device).clamp_min(1e-3)
            dt = (t * torch.rand(n_cons, device=device)).clamp_min(1e-3)
            t_target = (t - dt).clamp_min(0.0)

            xt = (1.0 - t)[:, None, None] * x1 + t[:, None, None] * x0
            v_pred = self.velocity(xt, rgb[n_flow:], robot_state[n_flow:], t)

            with torch.no_grad():
                xt_target = (1.0 - t_target)[:, None, None] * x1 + t_target[:, None, None] * x0
                v_target = self.velocity(xt_target, rgb[n_flow:], robot_state[n_flow:], t_target)

            # Consistency: predictions at neighbouring t should agree on the
            # implied endpoint x1_hat = xt - t * v.
            x1_pred = xt - t[:, None, None] * v_pred
            x1_targ = xt_target - t_target[:, None, None] * v_target
            loss_cons = F.mse_loss(x1_pred, x1_targ)

        loss = flow_batch_ratio * loss_flow + consistency_batch_ratio * loss_cons
        return {
            "loss": loss,
            "loss_flow": loss_flow.detach(),
            "loss_consistency": loss_cons.detach(),
        }


class HandoverFlowMatchingBaseline(HandoverManiFlowPolicy):
    """Plain flow-matching baseline.

    Same architecture, same observation/action interface, same one-step
    deployment regime.  The only difference is that the consistency term is
    disabled, so training reduces to ``loss_flow`` only.  This matches the
    "FM baseline" used in Table 2 of the report.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["use_consistency_training"] = False
        super().__init__(**kwargs)

    def compute_loss(self, rgb, robot_state, action_gt, **_) -> dict:
        return super().compute_loss(
            rgb, robot_state, action_gt,
            flow_batch_ratio=1.0,
            consistency_batch_ratio=0.0,
        )
