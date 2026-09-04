"""One episode loop, shared by every controller.

Both the classical baselines and the trained PPO policy are evaluated through
this function. Having a single loop is what guarantees that the two are scored
under identical conditions - the same seeds, the same step budget and the same
metric definitions.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from environment.warehouse_env import WarehouseEnv
from simulation.renderer import EpisodeRecorder

from .metrics import EpisodeMetrics


class Policy(Protocol):
    """Anything that can drive the robot."""

    name: str

    def reset(self, env: WarehouseEnv, seed: int) -> None: ...

    def act(self, observation: np.ndarray, env: WarehouseEnv) -> int: ...


def run_episode(
    env: WarehouseEnv,
    policy: Policy,
    seed: int,
    record: bool = False,
) -> tuple[EpisodeMetrics, EpisodeRecorder | None]:
    """Run a single episode and return its metrics (and optionally a replay)."""
    observation, _ = env.reset(seed=seed)
    policy.reset(env, seed)

    recorder: EpisodeRecorder | None = None
    if record:
        recorder = EpisodeRecorder.for_simulation(env.sim, policy.name, seed)
        recorder.capture(env.sim, events=["start"])

    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if recorder is not None:
            recorder.capture(
                env.sim,
                events=info.get("events", []),
                reward=reward,
                reward_components=info.get("reward_components"),
            )

    metrics = EpisodeMetrics.from_counters(
        env.sim.counters,
        controller=policy.name,
        scenario=env.config.name,
        seed=seed,
        total_reward=total_reward,
        tasks_required=env.config.tasks.tasks_per_episode,
    )
    if recorder is not None:
        recorder.finalise(env.sim, extra={"total_reward": round(total_reward, 3)})
    return metrics, recorder


def run_episodes(
    env: WarehouseEnv,
    policy: Policy,
    seeds: list[int],
    record_first: int = 0,
    record_dir: str | None = None,
) -> list[EpisodeMetrics]:
    """Run one episode per seed, optionally saving the first few as replays."""
    results: list[EpisodeMetrics] = []
    for index, seed in enumerate(seeds):
        record = index < record_first
        metrics, recorder = run_episode(env, policy, seed, record=record)
        results.append(metrics)
        if recorder is not None and record_dir:
            recorder.save(
                f"{record_dir}/{env.config.name}_{policy.name}_seed{seed}.json"
            )
    return results
