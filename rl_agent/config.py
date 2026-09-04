"""PPO hyper-parameters, in one place and explained in plain language.

Every field below has a one-line explanation of what it does, because the team
has to be able to justify each of these numbers in the viva. The defaults are
Stable-Baselines3' recommended starting values for discrete-action tasks,
adjusted only where the warehouse task clearly asks for it (longer rollouts
because episodes are long, a slightly higher entropy bonus because the agent
must keep exploring aisles it has not tried).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    """Training configuration for the PPO agent."""

    #: Scenario file (name under configs/ or an explicit path) to train on.
    scenario: str = "default"
    #: Total number of environment steps to collect during training.
    total_timesteps: int = 500_000
    #: Number of environment copies stepped in parallel.
    n_envs: int = 8
    #: Steps collected per environment before each policy update.
    n_steps: int = 512
    #: Mini-batch size used inside an update (must divide n_envs * n_steps).
    batch_size: int = 1024
    #: How many times each batch of experience is reused.
    n_epochs: int = 10
    #: Adam step size for both policy and value networks.
    learning_rate: float = 3e-4
    #: Discount factor: how much a reward one step later is worth.
    gamma: float = 0.99
    #: Generalised Advantage Estimation trade-off between bias and variance.
    gae_lambda: float = 0.95
    #: PPO clipping range; limits how far one update may move the policy.
    clip_range: float = 0.2
    #: Entropy bonus; keeps the policy stochastic enough to keep exploring.
    ent_coef: float = 0.01
    #: Weight of the value-function loss in the total loss.
    vf_coef: float = 0.5
    #: Gradient-norm clipping, guards against destructive updates.
    max_grad_norm: float = 0.5
    #: Hidden layer sizes of the shared actor-critic MLP.
    net_arch: tuple[int, ...] = (128, 128)
    #: Master seed for torch/numpy/env, so a run can be repeated exactly.
    seed: int = 0
    #: Run periodic evaluation every N training steps (0 disables it).
    eval_freq: int = 25_000
    #: Episodes per periodic evaluation.
    n_eval_episodes: int = 20
    #: Where models, logs and tensorboard traces are written.
    model_dir: str = "rl_agent/models"
    log_dir: str = "data/logs"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Fail early on hyper-parameter combinations PPO cannot use."""
        rollout = self.n_envs * self.n_steps
        if rollout % self.batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must divide n_envs * n_steps ({rollout})"
            )
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")

    def policy_kwargs(self) -> dict:
        return {"net_arch": list(self.net_arch)}

    def model_path(self, tag: str = "ppo_warehouse") -> Path:
        return Path(self.model_dir) / f"{tag}.zip"
