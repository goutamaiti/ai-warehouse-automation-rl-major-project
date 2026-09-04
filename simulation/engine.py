"""The warehouse simulation engine: one step of simulated time.

This module owns *all* the dynamics of the world - movement, collisions,
pickup/delivery, battery drain and charging - and nothing else. In particular
it contains:

* no reward computation (that belongs to :mod:`environment.warehouse_env`),
* no rendering (that belongs to :mod:`simulation.renderer`),
* no policy (that belongs to :mod:`rl_agent` or :mod:`baselines`).

Keeping those layers apart is what allows the PPO policy and the A* planner to
be evaluated on byte-identical dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .config import ScenarioConfig
from .navigation import DIRECTIONS, Position, bfs_distance_field
from .obstacles import ObstacleField
from .robot import Robot
from .tasks import Task, TaskGenerator, TaskStatus
from .warehouse import CellType, WarehouseLayout, build_layout


class Action(IntEnum):
    """Discrete action space. The first four match ``navigation.DIRECTIONS``."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    WAIT = 4


@dataclass
class StepOutcome:
    """Everything that happened during one simulated step.

    The RL environment turns this into a scalar reward; the recorder turns it
    into a dashboard frame. Neither interpretation lives here.
    """

    moved: bool = False
    blocked_by_static: bool = False
    blocked_by_obstacle: bool = False
    waited: bool = False
    picked_up: bool = False
    delivered: bool = False
    charged: float = 0.0
    energy_used: float = 0.0
    distance_before: int = 0
    distance_after: int = 0
    battery_depleted: bool = False
    all_tasks_done: bool = False
    timed_out: bool = False
    events: list[str] = field(default_factory=list)

    @property
    def collided(self) -> bool:
        return self.blocked_by_static or self.blocked_by_obstacle

    @property
    def terminated(self) -> bool:
        """Episode ended for a reason inherent to the task (success or failure)."""
        return self.battery_depleted or self.all_tasks_done

    @property
    def truncated(self) -> bool:
        """Episode ended because the step budget ran out."""
        return self.timed_out and not self.terminated


@dataclass
class EpisodeCounters:
    """Raw counters accumulated over an episode.

    These are counts of things that actually happened in the simulation. The
    derived metrics (success rate, path efficiency, ...) are computed from them
    in :mod:`analytics.metrics`.
    """

    steps: int = 0
    moves: int = 0
    collisions: int = 0
    static_collisions: int = 0
    dynamic_collisions: int = 0
    idle_steps: int = 0
    energy_consumed: float = 0.0
    charging_events: int = 0
    tasks_delivered: int = 0
    tasks_failed: int = 0
    optimal_path_length: int = 0
    termination_reason: str = "running"

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "moves": self.moves,
            "collisions": self.collisions,
            "static_collisions": self.static_collisions,
            "dynamic_collisions": self.dynamic_collisions,
            "idle_steps": self.idle_steps,
            "energy_consumed": round(self.energy_consumed, 3),
            "charging_events": self.charging_events,
            "tasks_delivered": self.tasks_delivered,
            "tasks_failed": self.tasks_failed,
            "optimal_path_length": self.optimal_path_length,
            "termination_reason": self.termination_reason,
        }


class WarehouseSimulation:
    """A single-robot warehouse episode.

    Typical use::

        sim = WarehouseSimulation(config)
        sim.reset(seed=0)
        while not sim.done:
            outcome = sim.step(Action.UP)
    """

    def __init__(self, config: ScenarioConfig, layout: WarehouseLayout | None = None) -> None:
        self.config = config
        # The layout is a pure function of the config, so it is normally built
        # once and shared by every episode of this scenario. A caller that
        # already has a layout (a user-drawn warehouse from the dashboard
        # editor) passes it directly instead - everything downstream reads
        # ``self.layout``, never ``config.layout``, so the two paths are
        # interchangeable.
        self.layout: WarehouseLayout = layout if layout is not None else build_layout(config.layout)
        self._walkable = self.layout.walkable_mask()
        self.rng = np.random.default_rng(config.seed)
        self.robot: Robot
        self.obstacles: ObstacleField
        self.task: Task
        self.counters = EpisodeCounters()
        self._target_field: np.ndarray
        self._target_cache: Position | None = None
        self._done = False
        self.reset(seed=config.seed)

    # ------------------------------------------------------------------
    # episode life cycle
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Start a new episode. Passing a seed makes the episode reproducible."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        start = self.layout.charging_stations[0]
        self.robot = Robot(
            robot_id=0,
            position=start,
            battery=min(self.config.battery.start_level, self.config.battery.capacity),
        )
        self._task_generator = TaskGenerator(self.layout, self.config.tasks, self.rng)
        self.task = self._task_generator.generate(step=0)
        self._tasks_remaining = self.config.tasks.tasks_per_episode - 1
        self.counters = EpisodeCounters()
        self.counters.optimal_path_length = self._optimal_legs(start, self.task)

        forbidden = {start, self.task.pickup, self.task.dropoff}
        forbidden.update(self.layout.charging_stations)
        forbidden.update(self.layout.packing_stations)
        self.obstacles = ObstacleField.spawn(
            self._walkable, self.config.obstacles, self.rng, forbidden
        )

        self._target_cache = None
        self._refresh_target_field()
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # simulation step
    # ------------------------------------------------------------------
    def step(self, action: Action | int) -> StepOutcome:
        """Advance the world by one time step."""
        if self._done:
            raise RuntimeError("episode is finished; call reset() first")
        action = Action(int(action))
        battery_cfg = self.config.battery
        outcome = StepOutcome(distance_before=self.distance_to_target())

        # --- 1. robot movement ---------------------------------------
        if action is Action.WAIT:
            outcome.waited = True
            outcome.energy_used = battery_cfg.idle_cost
            self.counters.idle_steps += 1
        else:
            delta = DIRECTIONS[int(action)]
            target = (self.robot.position[0] + delta[0], self.robot.position[1] + delta[1])
            if not self.layout.is_walkable(target):
                outcome.blocked_by_static = True
                outcome.energy_used = battery_cfg.idle_cost
                outcome.events.append("collision_static")
                self.counters.static_collisions += 1
            elif target in self.obstacles.positions:
                outcome.blocked_by_obstacle = True
                outcome.energy_used = battery_cfg.idle_cost
                outcome.events.append("collision_obstacle")
                self.counters.dynamic_collisions += 1
            else:
                self.robot.position = target
                outcome.moved = True
                outcome.energy_used = battery_cfg.move_cost
                self.counters.moves += 1

        self.robot.consume(outcome.energy_used)
        self.counters.energy_consumed += outcome.energy_used
        self.counters.collisions = (
            self.counters.static_collisions + self.counters.dynamic_collisions
        )

        # --- 2. automatic interactions at the current cell -------------
        self._handle_pickup_and_delivery(outcome)
        self._handle_charging(outcome)

        # --- 3. the rest of the world moves ---------------------------
        self.obstacles.step(self._walkable, self.rng, blocked={self.robot.position})

        # --- 4. termination checks ------------------------------------
        self.counters.steps += 1
        if self.robot.is_depleted():
            outcome.battery_depleted = True
            outcome.events.append("battery_depleted")
            self.task.mark_failed()
            self.counters.tasks_failed += 1
            self.counters.termination_reason = "battery_depleted"
        elif self.counters.steps >= self.config.max_steps and not outcome.all_tasks_done:
            outcome.timed_out = True
            outcome.events.append("timeout")
            self.task.mark_failed()
            self.counters.tasks_failed += 1
            self.counters.termination_reason = "timeout"

        self._refresh_target_field()
        outcome.distance_after = self.distance_to_target()
        self._done = outcome.terminated or outcome.truncated
        return outcome

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _handle_pickup_and_delivery(self, outcome: StepOutcome) -> None:
        """Pickup and delivery happen automatically at the right cell.

        Explicit pickup/drop actions were deliberately left out of the action
        space: they would add two actions that the agent has to learn to use
        without changing the navigation problem being studied.
        """
        position = self.robot.position
        if self.task.status is TaskStatus.PENDING and position == self.task.pickup:
            self.task.mark_picked_up(self.counters.steps)
            self.robot.carrying = True
            outcome.picked_up = True
            outcome.events.append("pickup")
        elif self.task.status is TaskStatus.PICKED_UP and position == self.task.dropoff:
            self.task.mark_delivered(self.counters.steps)
            self.robot.carrying = False
            outcome.delivered = True
            outcome.events.append("delivery")
            self.counters.tasks_delivered += 1
            if self._tasks_remaining > 0:
                self._tasks_remaining -= 1
                self.task = self._task_generator.generate(step=self.counters.steps)
                self.counters.optimal_path_length += self._optimal_legs(
                    position, self.task
                )
                outcome.events.append("new_task")
            else:
                outcome.all_tasks_done = True
                self.counters.termination_reason = "all_tasks_delivered"

    def _handle_charging(self, outcome: StepOutcome) -> None:
        if self.layout.cell(self.robot.position) is CellType.CHARGING:
            gained = self.robot.charge(self.config.battery)
            if gained > 0:
                outcome.charged = gained
                outcome.events.append("charging")
                self.counters.charging_events += 1

    def _optimal_legs(self, start: Position, task: Task) -> int:
        """Shortest achievable path length for a task, ignoring dynamic traffic.

        Used as the denominator of the path-efficiency metric. Because dynamic
        obstacles are ignored, it is a lower bound: no planner can do better.
        """
        to_pickup = bfs_distance_field(self._walkable, task.pickup)[start]
        to_dropoff = bfs_distance_field(self._walkable, task.dropoff)[task.pickup]
        return int(to_pickup) + int(to_dropoff)

    def _refresh_target_field(self) -> None:
        """Recompute the distance field only when the target actually changes."""
        target = self.task.target
        if target != self._target_cache:
            self._target_field = bfs_distance_field(self._walkable, target)
            self._target_cache = target

    # ------------------------------------------------------------------
    # observations for other layers
    # ------------------------------------------------------------------
    def distance_to_target(self) -> int:
        """Shortest-path distance from the robot to its current target."""
        return int(self._target_field[self.robot.position])

    def occupancy_grid(self) -> np.ndarray:
        """``True`` where a cell is blocked, including dynamic obstacles."""
        blocked = ~self._walkable
        return blocked | self.obstacles.occupancy(self.layout.shape)

    @property
    def walkable(self) -> np.ndarray:
        """Static walkability mask (dynamic obstacles excluded)."""
        return self._walkable

    def snapshot(self, events: list[str] | None = None) -> dict:
        """JSON-serialisable state, consumed by the recorder and the API."""
        return {
            "step": self.counters.steps,
            "robot": self.robot.to_dict(self.config.battery),
            "task": self.task.to_dict(),
            "obstacles": self.obstacles.to_list(),
            "distance_to_target": self.distance_to_target(),
            "events": list(events or []),
        }
