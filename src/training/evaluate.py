"""Real-world rollout evaluation for the handover policies.

This script reproduces the evaluation protocol from the report:

    * For each of the three tasks (cucumber, pepper, banana) we run 30 rollouts.
    * Each rollout is counted as successful only if the object is grasped,
      transferred between the two arms, and handed over without dropping or
      requiring manual intervention.
    * Both ManiFlow and the flow-matching baseline are tested with one network
      evaluation per action chunk (one-step inference).

The script is structured so that the same ``Evaluator`` class can run either
in simulation (with the ``DummyEnv`` placeholder used for unit tests) or on
the real platform via ``src.robot.robot_interface.RobotInterface``.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from src.common.config import CONFIG
from src.models import HandoverFlowMatchingBaseline, HandoverManiFlowPolicy
from src.training.dataset import MinMaxNormalizer
from src.robot.robot_interface import RobotInterface
from src.robot.safety_checks import SafetyMonitor


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    task: str
    rollout_id: int
    success: bool
    failure_stage: Optional[str] = None         # "pickup", "transfer", "release"
    notes: str = ""


@dataclass
class TaskSummary:
    task: str
    trials: int = 0
    successes: int = 0
    failures_by_stage: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


# ---------------------------------------------------------------------------
# Loading checkpoints
# ---------------------------------------------------------------------------


def _build_policy_from_ckpt(ckpt_path: str, device: torch.device):
    payload = torch.load(ckpt_path, map_location="cpu")
    cls = payload.get("policy_class", "HandoverManiFlowPolicy")

    shape = CONFIG.shape
    common = dict(
        image_size=shape.image_size,
        state_dim=shape.state_dim,
        action_dim=shape.action_dim,
        action_horizon=shape.action_horizon,
        inference_steps=CONFIG.train.inference_steps,
    )

    if cls == "HandoverFlowMatchingBaseline":
        policy = HandoverFlowMatchingBaseline(**common)
    else:
        policy = HandoverManiFlowPolicy(use_consistency_training=True, **common)

    policy.load_state_dict(payload["model"])
    policy.eval().to(device)

    state_norm = MinMaxNormalizer.from_dict(payload["normalizer"]["state"])
    action_norm = MinMaxNormalizer.from_dict(payload["normalizer"]["action"])
    return policy, state_norm, action_norm


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    def __init__(
        self,
        policy: torch.nn.Module,
        state_norm: MinMaxNormalizer,
        action_norm: MinMaxNormalizer,
        robot: RobotInterface,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.state_norm = state_norm
        self.action_norm = action_norm
        self.robot = robot
        self.device = device
        self.safety = SafetyMonitor()

    @torch.no_grad()
    def predict(self, rgb_np: np.ndarray, state_np: np.ndarray) -> np.ndarray:
        rgb = torch.from_numpy(rgb_np).to(self.device).unsqueeze(0)
        state_norm = self.state_norm.normalize(state_np.astype(np.float32))
        state = torch.from_numpy(state_norm).to(self.device).unsqueeze(0)
        action_norm = self.policy(rgb, state).squeeze(0).cpu().numpy()
        return self.action_norm.denormalize(action_norm)

    def run_rollout(self, task: str, rollout_id: int, max_steps: int = 600) -> RolloutResult:
        self.robot.reset_for_task(task)
        for step in range(max_steps):
            obs = self.robot.read_observation()
            chunk = self.predict(obs.rgb, obs.state)

            # Receding-horizon execution: the report uses one-step / first
            # action of the chunk, both for ManiFlow and the FM baseline.
            action = chunk[: CONFIG.deploy.receding_horizon_first_n]
            ok, reason = self.safety.check(action, obs.state)
            if not ok:
                return RolloutResult(task, rollout_id, False, failure_stage="safety",
                                     notes=reason)

            done, stage = self.robot.step(action)
            if done:
                success = stage == "release_done"
                return RolloutResult(task, rollout_id, success,
                                     failure_stage=None if success else stage)

        return RolloutResult(task, rollout_id, False, failure_stage="timeout")


def _summarise(results: List[RolloutResult]) -> Dict[str, TaskSummary]:
    summary: Dict[str, TaskSummary] = {}
    for r in results:
        s = summary.setdefault(r.task, TaskSummary(task=r.task))
        s.trials += 1
        if r.success:
            s.successes += 1
        elif r.failure_stage:
            s.failures_by_stage[r.failure_stage] = s.failures_by_stage.get(r.failure_stage, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, choices=list(CONFIG.tasks) + ["all"])
    parser.add_argument("--rollouts", type=int, default=30,
                        help="Rollouts per task (default mirrors the report).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="./results/rollout_log.json")
    parser.add_argument("--simulate", action="store_true",
                        help="Run against the in-process DummyEnv instead of the real robot.")
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, state_norm, action_norm = _build_policy_from_ckpt(args.checkpoint, device)
    robot = RobotInterface(simulate=args.simulate)
    evaluator = Evaluator(policy, state_norm, action_norm, robot, device)

    tasks = list(CONFIG.tasks) if args.task == "all" else [args.task]
    results: List[RolloutResult] = []
    for task in tasks:
        for i in range(args.rollouts):
            r = evaluator.run_rollout(task, i)
            results.append(r)
            tag = "OK" if r.success else f"FAIL[{r.failure_stage}]"
            print(f"[{task}] rollout {i:02d}: {tag}")

    summary = _summarise(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "summary": {t: {"trials": s.trials, "successes": s.successes,
                                "rate": s.success_rate,
                                "failures_by_stage": s.failures_by_stage}
                            for t, s in summary.items()},
                "results": [r.__dict__ for r in results],
            }, f, indent=2,
        )

    total_trials = sum(s.trials for s in summary.values())
    total_succ = sum(s.successes for s in summary.values())
    print("\n=== Summary ===")
    for s in summary.values():
        print(f"{s.task:<10}: {s.successes}/{s.trials} ({s.success_rate * 100:.1f}%)")
    print(f"Overall   : {total_succ}/{total_trials} "
          f"({100 * total_succ / max(total_trials, 1):.1f}%)")


if __name__ == "__main__":
    main()
