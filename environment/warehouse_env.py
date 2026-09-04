"""Gymnasium environment wrapping the warehouse simulation.

This is the only place in the project where a scalar *reward* exists. The
simulation itself reports facts (moved, collided, delivered, energy used); this
module turns those facts into the learning signal, which keeps reward shaping
experiments confined to one file.

Observation vector (see :func:`WarehouseEnv.observation_labels` for the exact
index of every entry)::

    [0:2]   robot position, normalised to [0, 1]
    [2:4]   target position, normalised to [0, 1]
    [4:6]   target offset (target - robot), normalised to [-1, 1]
    [6]     shortest-path distance to target, normalised
    [7]     battery level as a fraction of capacity
    [8]     1.0 when the robot carries a package
    [9:11]  offset to the nearest charging station, normalised to [-1, 1]
    [11]    shortest-path distance to that charger, normalised
    [12:]   local occupancy patch, row major (0 free, 0.5 obstacle, 1 blocked)

Everything is bounded to [-1, 1] so that no feature dominates the network input
scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.config import ScenarioConfig, load_scenario
from simulation.engine import Action, WarehouseSimulation
from simulation.navigation import UNREACHABLE, bfs_distance_field
from simulation.renderer import render_ascii

#: Number of scalar features that precede the local occupancy patch.
N_SCALAR_FEATURES = 12


class WarehouseEnv(gym.Env):
    """Single-robot warehouse navigation environment."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 8}

    def __init__(
        self,
        config: ScenarioConfig | str | Path | None = None,
        render_mode: str | None = None,
        layout=None,
    ) -> None:
        super().__init__()
        if config is None:
            config = ScenarioConfig()
        elif isinstance(config, (str, Path)):
            config = load_scenario(config)
        self.config = config
        self.render_mode = render_mode

        # ``layout``, when given, overrides the layout the config would
        # otherwise generate - this is how a user-drawn warehouse from the
        # dashboard editor is simulated with the same physics and reward as
        # every procedurally generated scenario.
        self.sim = WarehouseSimulation(config, layout=layout)
        self.window = int(config.observation_window)
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError("observation_window must be a positive odd number")

        patch = self.window * self.window
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(N_SCALAR_FEATURES + patch,),
            dtype=np.float32,
        )

        self._distance_scale = self._compute_distance_scale()
        self._charger_fields = {
            station: bfs_distance_field(self.sim.walkable, station)
            for station in self.sim.layout.charging_stations
        }
        self._last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        With an explicit ``seed`` the episode is an exact, repeatable problem
        instance - this is what the evaluation scripts use so that A* and PPO
        face identical tasks. Without one, a fresh episode seed is drawn from
        the environment RNG so that training sees varied tasks.
        """
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else int(self.np_random.integers(2**31 - 1))
        self.sim.reset(seed=episode_seed)
        self._episode_seed = episode_seed
        observation = self._observation()
        info = {
            "episode_seed": episode_seed,
            "task": self.sim.task.to_dict(),
            "optimal_path_length": self.sim.counters.optimal_path_length,
        }
        self._last_info = info
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        outcome = self.sim.step(int(action))
        reward, components = self._reward(outcome)

        info: dict[str, Any] = {
            "events": outcome.events,
            "reward_components": components,
            "battery": round(self.sim.robot.battery, 2),
            "distance_to_target": outcome.distance_after,
            "task_status": self.sim.task.status.value,
            "collided": outcome.collided,
        }
        terminated, truncated = outcome.terminated, outcome.truncated
        if terminated or truncated:
            info["is_success"] = bool(
                self.sim.counters.tasks_delivered
                == self.config.tasks.tasks_per_episode
            )
            info["episode_summary"] = self.sim.counters.to_dict()
        self._last_info = info
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return render_ascii(self.sim)
        if self.render_mode == "human":
            print(render_ascii(self.sim))
        return None

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------
    def _reward(self, outcome) -> tuple[float, dict[str, float]]:
        """Turn a :class:`~simulation.engine.StepOutcome` into a scalar reward.

        The progress term is *potential-based* shaping (Ng, Harada & Russell,
        1999) using the negated shortest-path distance as the potential:

            F(s, s') = gamma * Phi(s') - Phi(s),  Phi(s) = -distance(s)

        Shaping of this form provably does not change the optimal policy, which
        is why it is preferred here over an ad-hoc "reward for getting closer".
        The term is skipped on steps where the target itself changes (pickup or
        a new task), because the potential of the old and the new goal are not
        comparable.
        """
        cfg = self.config.reward
        target_changed = outcome.picked_up or "new_task" in outcome.events

        shaping = 0.0
        if not target_changed and outcome.distance_before >= 0 and outcome.distance_after >= 0:
            phi_before = -float(outcome.distance_before)
            phi_after = -float(outcome.distance_after)
            shaping = cfg.progress_weight * (cfg.shaping_gamma * phi_after - phi_before)

        components = {
            "step_penalty": -cfg.step_penalty,
            "progress": shaping,
            "pickup": cfg.pickup_reward if outcome.picked_up else 0.0,
            "delivery": cfg.delivery_reward if outcome.delivered else 0.0,
            "collision": -cfg.collision_penalty if outcome.collided else 0.0,
            "wait": -cfg.wait_penalty if outcome.waited else 0.0,
            "energy": -cfg.energy_weight * outcome.energy_used,
            "battery_depleted": (
                -cfg.battery_depleted_penalty if outcome.battery_depleted else 0.0
            ),
            "timeout": -cfg.timeout_penalty if outcome.truncated else 0.0,
        }
        return sum(components.values()), components

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _compute_distance_scale(self) -> float:
        """A stable constant used to normalise shortest-path distances.

        Computed once from the static map (the longest distance from any
        station to any cell), so the normalisation never changes between
        episodes or between controllers.
        """
        walkable = self.sim.walkable
        anchors = (
            self.sim.layout.packing_stations + self.sim.layout.charging_stations
        )
        longest = 1
        for anchor in anchors:
            field = bfs_distance_field(walkable, anchor)
            longest = max(longest, int(field[field != UNREACHABLE].max()))
        return float(longest)

    def _normalise_distance(self, distance: int) -> float:
        if distance < 0:  # unreachable
            return 1.0
        return float(min(distance / self._distance_scale, 1.0))

    def _local_patch(self) -> np.ndarray:
        """Occupancy of the cells around the robot.

        Values: 0 free, 0.5 dynamic obstacle, 1 static blocker. Cells outside
        the grid are reported as blocked, which is what they effectively are.
        """
        radius = self.window // 2
        row, col = self.sim.robot.position
        static_blocked = ~self.sim.walkable
        dynamic = self.sim.obstacles.occupancy(self.sim.layout.shape)

        patch = np.ones((self.window, self.window), dtype=np.float32)
        height, width = self.sim.layout.shape
        for i, r in enumerate(range(row - radius, row + radius + 1)):
            for j, c in enumerate(range(col - radius, col + radius + 1)):
                if not (0 <= r < height and 0 <= c < width):
                    continue
                if static_blocked[r, c]:
                    patch[i, j] = 1.0
                elif dynamic[r, c]:
                    patch[i, j] = 0.5
                else:
                    patch[i, j] = 0.0
        return patch.reshape(-1)

    def _observation(self) -> np.ndarray:
        layout = self.sim.layout
        height, width = layout.shape
        row_scale = max(height - 1, 1)
        col_scale = max(width - 1, 1)

        robot = self.sim.robot.position
        target = self.sim.task.target
        charger, charger_distance = self._nearest_charger()

        scalars = np.array(
            [
                robot[0] / row_scale,
                robot[1] / col_scale,
                target[0] / row_scale,
                target[1] / col_scale,
                (target[0] - robot[0]) / row_scale,
                (target[1] - robot[1]) / col_scale,
                self._normalise_distance(self.sim.distance_to_target()),
                self.sim.robot.battery_fraction(self.config.battery),
                1.0 if self.sim.robot.carrying else 0.0,
                (charger[0] - robot[0]) / row_scale,
                (charger[1] - robot[1]) / col_scale,
                self._normalise_distance(charger_distance),
            ],
            dtype=np.float32,
        )
        observation = np.concatenate([scalars, self._local_patch()])
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def _nearest_charger(self) -> tuple[tuple[int, int], int]:
        robot = self.sim.robot.position
        best_station = self.sim.layout.charging_stations[0]
        best_distance = int(self._charger_fields[best_station][robot])
        for station, field in self._charger_fields.items():
            distance = int(field[robot])
            if distance >= 0 and (best_distance < 0 or distance < best_distance):
                best_station, best_distance = station, distance
        return best_station, best_distance

    def observation_labels(self) -> list[str]:
        """Human-readable name of every observation index.

        Used by the tests and by the dashboard; being able to name each feature
        is also what makes the observation space explainable in a viva.
        """
        labels = [
            "robot_row",
            "robot_col",
            "target_row",
            "target_col",
            "target_offset_row",
            "target_offset_col",
            "distance_to_target",
            "battery_fraction",
            "carrying",
            "charger_offset_row",
            "charger_offset_col",
            "distance_to_charger",
        ]
        radius = self.window // 2
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                labels.append(f"patch[{i:+d},{j:+d}]")
        return labels
