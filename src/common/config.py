"""Shared configuration for the Team14 ManiFlow handover reproduction.

The numbers here come directly from Section "Appendix A: Qualcomm Device Usage"
of the final report:
    RGB input         : 1 x 3 x 224 x 224
    Robot-state input : 1 x 14
    Action chunk out  : 1 x 16 x 14
"""
from dataclasses import dataclass, field
from typing import List, Tuple


HANDOVER_TASKS: Tuple[str, ...] = ("cucumber", "pepper", "banana")
DEMOS_PER_TASK: int = 80
TOTAL_DEMOS: int = DEMOS_PER_TASK * len(HANDOVER_TASKS)


@dataclass
class HandoverShape:
    image_size: int = 224
    image_channels: int = 3
    state_dim: int = 14          # 7 DoF per arm x 2 arms
    action_dim: int = 14
    action_horizon: int = 16     # chunk length
    n_obs_steps: int = 1


@dataclass
class HandoverTraining:
    # Optimisation
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-6
    batch_size: int = 64
    num_epochs: int = 300
    val_ratio: float = 0.05
    seed: int = 42

    # ManiFlow consistency-flow training
    use_consistency_training: bool = True
    flow_batch_ratio: float = 0.75
    consistency_batch_ratio: float = 0.25
    denoise_timesteps: int = 10
    inference_steps: int = 1     # one-step deployment regime


@dataclass
class HandoverDeployment:
    # Networking
    upper_computer_ip: str = "192.168.3.101"
    upper_computer_port: int = 7001
    board_ssh: str = "radxa@192.168.3.42"
    board_dir: str = "/home/radxa/maniflow_q900"

    # QNN / QCS9075
    dsp_arch: str = "v73"
    soc_id: int = 77

    # Bridge
    receding_horizon_first_n: int = 1   # report uses one-step execution
    action_freshness_window_s: float = 0.25


@dataclass
class HandoverConfig:
    shape: HandoverShape = field(default_factory=HandoverShape)
    train: HandoverTraining = field(default_factory=HandoverTraining)
    deploy: HandoverDeployment = field(default_factory=HandoverDeployment)
    tasks: Tuple[str, ...] = HANDOVER_TASKS

    @property
    def rgb_shape(self) -> Tuple[int, int, int, int]:
        return (1, self.shape.image_channels, self.shape.image_size, self.shape.image_size)

    @property
    def state_shape(self) -> Tuple[int, int]:
        return (1, self.shape.state_dim)

    @property
    def action_shape(self) -> Tuple[int, int, int]:
        return (1, self.shape.action_horizon, self.shape.action_dim)


CONFIG = HandoverConfig()
