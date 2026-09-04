"""Classical controllers that drive the simulation, for comparison with PPO.

Fairness note (important for the report): the classical controllers are given
the *full* map and the exact positions of all dynamic obstacles, while the RL
policy only ever sees its 37-dimensional observation. The comparison is
therefore deliberately biased in favour of A*; any advantage the learned policy
shows is achieved with strictly less information.
"""

from __future__ import annotations

import numpy as np

from environment.warehouse_env import WarehouseEnv
from simulation.engine import Action, WarehouseSimulation
from simulation.navigation import DIRECTIONS, Position, bfs_distance_field

from .astar import PlanResult, astar
from .bfs import bfs

PLANNERS = {"astar": astar, "bfs": bfs}


def _action_towards(current: Position, nxt: Position) -> int:
    """Translate a one-cell move into the matching discrete action."""
    delta = (nxt[0] - current[0], nxt[1] - current[1])
    try:
        return int(DIRECTIONS.index(delta))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{current} -> {nxt} is not a single 4-connected move") from exc


class RandomPolicy:
    """Uniformly random actions: the floor any learned policy must beat."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, env: WarehouseEnv, seed: int) -> None:
        self.rng = np.random.default_rng(self.seed + seed)

    def act(self, observation: np.ndarray, env: WarehouseEnv) -> int:
        return int(self.rng.integers(len(Action)))


class PlannerPolicy:
    """Battery-aware shortest-path controller (A* by default).

    Each step it

    1. decides whether the battery is sufficient to finish the current leg and
       still reach a charger afterwards - if not, the charger becomes the goal;
    2. replans from the robot's current cell on a grid that includes the
       dynamic obstacles, so it routes *around* visible traffic;
    3. falls back to waiting when an obstacle temporarily seals the only aisle.

    Replanning every step is a deliberate choice: it is the standard way a
    classical planner copes with a changing world and it gives the strongest
    reasonable classical baseline.
    """

    def __init__(
        self,
        planner: str = "astar",
        safety_factor: float = 1.2,
        charge_until: float = 0.9,
    ) -> None:
        if planner not in PLANNERS:
            raise ValueError(f"unknown planner {planner!r}, expected one of {sorted(PLANNERS)}")
        self.planner_name = planner
        self._plan_fn = PLANNERS[planner]
        self.safety_factor = safety_factor
        self.charge_until = charge_until
        self._charging = False
        self._charger_fields: dict[Position, np.ndarray] = {}
        self.nodes_expanded = 0

    @property
    def name(self) -> str:
        return self.planner_name

    def reset(self, env: WarehouseEnv, seed: int) -> None:
        sim = env.sim
        self._charging = False
        self.nodes_expanded = 0
        self._charger_fields = {
            station: bfs_distance_field(sim.walkable, station)
            for station in sim.layout.charging_stations
        }

    # ------------------------------------------------------------------
    def act(self, observation: np.ndarray, env: WarehouseEnv) -> int:
        sim = env.sim
        goal = self._select_goal(sim)
        position = sim.robot.position

        if position == goal:
            # Already there: the only useful thing to do is to keep charging.
            return int(Action.WAIT)

        blocked = sim.occupancy_grid().copy()
        blocked[position] = False
        plan = self._plan(blocked, position, goal)
        if not plan.found:
            # Obstacles sealed every route; fall back to the static map and
            # wait if the next cell is occupied right now.
            static_blocked = ~sim.walkable
            plan = self._plan(static_blocked, position, goal)
            if not plan.found or plan.length == 0:
                return int(Action.WAIT)
            if plan.path[1] in sim.obstacles.positions:
                return int(Action.WAIT)
        return _action_towards(position, plan.path[1])

    # ------------------------------------------------------------------
    def _plan(self, blocked: np.ndarray, start: Position, goal: Position) -> PlanResult:
        result = self._plan_fn(blocked, start, goal)
        self.nodes_expanded += result.nodes_expanded
        return result

    def _select_goal(self, sim: WarehouseSimulation) -> Position:
        """Return the cell to drive to: either the task target or a charger."""
        battery_cfg = sim.config.battery
        battery = sim.robot.battery
        target = sim.task.target

        if self._charging:
            if battery >= battery_cfg.capacity * self.charge_until:
                self._charging = False
            else:
                return self._nearest_charger(sim.robot.position)

        # Energy needed to finish this leg and still reach a charger afterwards.
        distance_to_target = sim.distance_to_target()
        distance_target_to_charger = min(
            int(field[target]) for field in self._charger_fields.values()
        )
        required = (
            (distance_to_target + distance_target_to_charger)
            * battery_cfg.move_cost
            * self.safety_factor
        )
        nearly_full = battery >= battery_cfg.capacity * self.charge_until
        if not nearly_full and (battery <= required or battery <= battery_cfg.critical_threshold):
            self._charging = True
            return self._nearest_charger(sim.robot.position)
        return target

    def _nearest_charger(self, position: Position) -> Position:
        reachable = {
            station: int(field[position])
            for station, field in self._charger_fields.items()
            if field[position] >= 0
        }
        if not reachable:  # pragma: no cover - layout validation prevents this
            raise RuntimeError("no reachable charging station")
        return min(reachable, key=reachable.get)
