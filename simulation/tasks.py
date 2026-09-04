"""Delivery tasks: pick a package up at a storage point, drop it at a station.

Scope note: the first version runs one task at a time for a single robot, as
planned in the project memory. A task queue and multiple robots are extensions
that build on this module rather than replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import TaskConfig
from .navigation import Position, manhattan
from .warehouse import WarehouseLayout


class TaskStatus(str, Enum):
    """Life cycle of a delivery task."""

    PENDING = "pending"        # package still at the storage point
    PICKED_UP = "picked_up"    # robot is carrying the package
    DELIVERED = "delivered"    # package reached the packing station
    FAILED = "failed"          # episode ended before delivery


@dataclass
class Task:
    """A single pickup-and-delivery job."""

    task_id: int
    pickup: Position
    dropoff: Position
    created_step: int = 0
    status: TaskStatus = TaskStatus.PENDING
    picked_step: int | None = None
    delivered_step: int | None = None

    @property
    def target(self) -> Position:
        """The position the robot currently has to reach.

        For a finished task (delivered or failed) this keeps returning the last
        target that was being pursued. Flipping back to the pickup point would
        make the distance-to-target jump on the final step, which corrupts the
        potential-based shaping term in the reward (the delivery step would be
        charged for "moving away" from a goal it had already reached).
        """
        if self.status is TaskStatus.PENDING:
            return self.pickup
        if self.status is TaskStatus.PICKED_UP:
            return self.dropoff
        return self.dropoff if self.picked_step is not None else self.pickup

    @property
    def is_open(self) -> bool:
        return self.status in (TaskStatus.PENDING, TaskStatus.PICKED_UP)

    def mark_picked_up(self, step: int) -> None:
        if self.status is not TaskStatus.PENDING:
            raise ValueError(f"cannot pick up a task in state {self.status}")
        self.status = TaskStatus.PICKED_UP
        self.picked_step = step

    def mark_delivered(self, step: int) -> None:
        if self.status is not TaskStatus.PICKED_UP:
            raise ValueError(f"cannot deliver a task in state {self.status}")
        self.status = TaskStatus.DELIVERED
        self.delivered_step = step

    def mark_failed(self) -> None:
        if self.is_open:
            self.status = TaskStatus.FAILED

    def to_dict(self) -> dict:
        return {
            "id": self.task_id,
            "pickup": list(self.pickup),
            "dropoff": list(self.dropoff),
            "status": self.status.value,
            "target": list(self.target),
            "created_step": self.created_step,
            "picked_step": self.picked_step,
            "delivered_step": self.delivered_step,
        }


class TaskGenerator:
    """Samples delivery tasks from the layout with a seeded RNG.

    Sampling is seeded so that scenario N of an A* evaluation run and scenario N
    of a PPO evaluation run are the *same* problem instance. Without that, any
    comparison between the two would be measuring luck.
    """

    def __init__(
        self,
        layout: WarehouseLayout,
        config: TaskConfig,
        rng: np.random.Generator,
    ) -> None:
        if not layout.storage_points or not layout.packing_stations:
            raise ValueError("layout needs at least one storage point and one station")
        self.layout = layout
        self.config = config
        self.rng = rng
        self._next_id = 0

    def generate(self, step: int) -> Task:
        """Create the next task, honouring the minimum separation constraint."""
        storage = self.layout.storage_points
        stations = self.layout.packing_stations

        pairs = [
            (p, d)
            for p in storage
            for d in stations
            if manhattan(p, d) >= self.config.min_separation
        ]
        if not pairs:
            # Fall back to the pair that is as far apart as the layout allows.
            pairs = [
                max(
                    ((p, d) for p in storage for d in stations),
                    key=lambda pair: manhattan(*pair),
                )
            ]
        pickup, dropoff = pairs[int(self.rng.integers(len(pairs)))]

        task = Task(
            task_id=self._next_id,
            pickup=pickup,
            dropoff=dropoff,
            created_step=step,
        )
        self._next_id += 1
        return task
