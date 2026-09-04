"""Dynamic obstacles: other traffic moving through the aisles.

Modelling assumption (documented because it affects the collision metric):
a dynamic obstacle never steps onto the cell occupied by the robot. Obstacles
represent attentive humans and other managed robots, so collisions can only
happen when *our* robot drives into them. This keeps the collision count a
measure of the policy's behaviour instead of a measure of bad luck.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ObstacleConfig
from .navigation import Position, bfs_distance_field, neighbours

BEHAVIOURS = ("random_walk", "patrol")


@dataclass
class DynamicObstacle:
    """One moving obstacle."""

    obstacle_id: int
    position: Position
    behaviour: str = "random_walk"
    #: Patrol endpoints; unused for random walkers.
    waypoints: tuple[Position, ...] = ()
    #: Index of the waypoint currently being approached.
    waypoint_index: int = 0
    #: Distance field towards each waypoint, precomputed at spawn time.
    _fields: list[np.ndarray] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.obstacle_id,
            "position": list(self.position),
            "behaviour": self.behaviour,
        }


class ObstacleField:
    """Container that advances every dynamic obstacle by one step."""

    def __init__(self, obstacles: list[DynamicObstacle], config: ObstacleConfig) -> None:
        self.obstacles = obstacles
        self.config = config

    @classmethod
    def spawn(
        cls,
        walkable: np.ndarray,
        config: ObstacleConfig,
        rng: np.random.Generator,
        forbidden: set[Position],
    ) -> ObstacleField:
        """Place obstacles on random free cells, avoiding ``forbidden`` cells."""
        if config.behaviour not in BEHAVIOURS:
            raise ValueError(
                f"unknown obstacle behaviour {config.behaviour!r}, expected one of {BEHAVIOURS}"
            )
        free = [
            (int(r), int(c))
            for r, c in zip(*np.nonzero(walkable))
            if (int(r), int(c)) not in forbidden
        ]
        if config.n_dynamic > len(free):
            raise ValueError("not enough free cells to place the requested obstacles")

        chosen_idx = rng.choice(len(free), size=config.n_dynamic, replace=False)
        obstacles: list[DynamicObstacle] = []
        for obstacle_id, index in enumerate(np.atleast_1d(chosen_idx)):
            start = free[int(index)]
            obstacle = DynamicObstacle(
                obstacle_id=obstacle_id,
                position=start,
                behaviour=config.behaviour,
            )
            if config.behaviour == "patrol":
                end = free[int(rng.integers(len(free)))]
                obstacle.waypoints = (start, end)
                obstacle._fields = [
                    bfs_distance_field(walkable, waypoint)
                    for waypoint in obstacle.waypoints
                ]
                obstacle.waypoint_index = 1
            obstacles.append(obstacle)
        return cls(obstacles, config)

    @property
    def positions(self) -> set[Position]:
        return {obstacle.position for obstacle in self.obstacles}

    def occupancy(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean mask with ``True`` where an obstacle currently stands."""
        mask = np.zeros(shape, dtype=bool)
        for obstacle in self.obstacles:
            mask[obstacle.position] = True
        return mask

    def step(
        self,
        walkable: np.ndarray,
        rng: np.random.Generator,
        blocked: set[Position],
    ) -> None:
        """Advance every obstacle by at most one cell.

        ``blocked`` holds cells the obstacles must not enter (the robot's cell).
        Obstacles also avoid each other.
        """
        occupied = self.positions
        for obstacle in self.obstacles:
            if rng.random() > self.config.move_probability:
                continue
            options = [
                cell
                for cell in neighbours(obstacle.position, walkable)
                if cell not in blocked and cell not in occupied
            ]
            if not options:
                continue

            if obstacle.behaviour == "patrol" and obstacle._fields:
                target_field = obstacle._fields[obstacle.waypoint_index]
                reachable = [cell for cell in options if target_field[cell] >= 0]
                if reachable:
                    nxt = min(reachable, key=lambda cell: int(target_field[cell]))
                else:
                    nxt = options[int(rng.integers(len(options)))]
                if nxt == obstacle.waypoints[obstacle.waypoint_index]:
                    obstacle.waypoint_index = 1 - obstacle.waypoint_index
            else:
                nxt = options[int(rng.integers(len(options)))]

            occupied.discard(obstacle.position)
            occupied.add(nxt)
            obstacle.position = nxt

    def to_list(self) -> list[dict]:
        return [obstacle.to_dict() for obstacle in self.obstacles]
