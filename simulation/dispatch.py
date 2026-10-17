"""Shared delivery queue and nearest-robot dispatch.

What this replaces, and why
---------------------------
In the first fleet version every robot generated its own task independently.
That is what the project memo describes for a first multi-robot pass, and it
works, but it has a measured failure: two robots can be sent to the *same*
storage point, meet in an aisle with no detour, and jam. Driving five robots on
``default`` with A* deadlocked for exactly that reason.

A shared queue fixes the cause rather than the symptom. A request is assigned to
**one** robot and a robot holds **one** request, so two robots are never routed
to the same pickup in the first place.

The assignment rule
-------------------
The nearest *idle* robot wins, measured by **geodesic** distance - the number of
cells actually driven through the aisles, not straight-line distance. On a floor
built from shelf blocks the two disagree constantly: a robot one aisle over is a
few metres away and twenty cells of driving away, and picking it would be worse
than picking a robot that looks further off.

Only idle robots are candidates. A robot already carrying a package is never
re-assigned: dropping a package to chase a closer job would be worse service,
not better, and it would make delivery times unexplainable.

This is a classical dispatching rule, not a learned one. That is deliberate -
the project's research question is about *navigation*, and keeping assignment
classical means a comparison between controllers measures the controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

from .navigation import Position
from .tasks import Task, TaskStatus


class RequestStatus(str, Enum):
    """Life cycle of a delivery request in the queue."""

    #: In the queue, nobody assigned yet.
    WAITING = "waiting"
    #: A robot is on its way or carrying it.
    ASSIGNED = "assigned"
    DELIVERED = "delivered"
    #: The assigned robot died, or the request can never be served.
    FAILED = "failed"


@dataclass
class DeliveryRequest:
    """One product to move, from the moment it is requested."""

    request_id: int
    pickup: Position
    dropoff: Position
    created_step: int = 0
    status: RequestStatus = RequestStatus.WAITING
    robot_id: int | None = None
    assigned_step: int | None = None
    delivered_step: int | None = None

    @property
    def waiting(self) -> bool:
        return self.status is RequestStatus.WAITING

    def to_dict(self) -> dict:
        return {
            "id": self.request_id,
            "pickup": list(self.pickup),
            "dropoff": list(self.dropoff),
            "status": self.status.value,
            "robot_id": self.robot_id,
            "created_step": self.created_step,
            "assigned_step": self.assigned_step,
            "delivered_step": self.delivered_step,
            # Ticks from request to delivery - the number a warehouse operator
            # actually cares about, and the one a single robot cannot show.
            "turnaround": (
                None
                if self.delivered_step is None
                else self.delivered_step - self.created_step
            ),
        }


#: Distance between two cells through the aisles. ``-1`` means unreachable.
DistanceFn = Callable[[Position, Position], int]


@dataclass
class DispatchDecision:
    """One assignment, recorded so a replay can explain why it happened."""

    request_id: int
    robot_id: int
    distance: int

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "robot_id": self.robot_id,
            "distance": self.distance,
        }


@dataclass
class TaskQueue:
    """The shared list of delivery requests, and who is serving them."""

    requests: dict[int, DeliveryRequest] = field(default_factory=dict)
    _next_id: int = 0

    # ------------------------------------------------------------------
    def add(self, pickup: Position, dropoff: Position, step: int = 0) -> DeliveryRequest:
        """Queue a product for collection. This is what the dashboard card calls."""
        request = DeliveryRequest(
            request_id=self._next_id,
            pickup=pickup,
            dropoff=dropoff,
            created_step=step,
        )
        self.requests[request.request_id] = request
        self._next_id += 1
        return request

    # ------------------------------------------------------------------
    def waiting(self) -> list[DeliveryRequest]:
        """Unassigned requests, oldest first - no request is starved."""
        return sorted(
            (r for r in self.requests.values() if r.waiting),
            key=lambda r: (r.created_step, r.request_id),
        )

    def assigned_to(self, robot_id: int) -> DeliveryRequest | None:
        for request in self.requests.values():
            if request.robot_id == robot_id and request.status is RequestStatus.ASSIGNED:
                return request
        return None

    # ------------------------------------------------------------------
    def dispatch(
        self,
        idle_robots: dict[int, Position],
        distance: DistanceFn,
        step: int = 0,
    ) -> list[DispatchDecision]:
        """Assign waiting requests to the nearest idle robots.

        Requests are served oldest first, and each takes the nearest *reachable*
        idle robot by path distance. Ties break on the lower robot id, so the
        outcome is deterministic and a run replays exactly.

        Greedy rather than globally optimal: a Hungarian-style assignment would
        minimise total distance across all pairs, but it is harder to explain
        and the difference only shows when several requests arrive on the same
        tick. If that ever matters it is a measurement to run, not a guess -
        and the greedy result is the honest baseline to beat.
        """
        available = dict(idle_robots)
        decisions: list[DispatchDecision] = []

        for request in self.waiting():
            if not available:
                break
            reachable = [
                (dist, robot_id)
                for robot_id, position in available.items()
                for dist in [distance(position, request.pickup)]
                if dist >= 0
            ]
            if not reachable:
                # Nobody can get there. Leave it queued rather than failing it:
                # a robot that is currently boxed in may be free next tick.
                continue
            best_distance, robot_id = min(reachable)
            request.status = RequestStatus.ASSIGNED
            request.robot_id = robot_id
            request.assigned_step = step
            del available[robot_id]
            decisions.append(
                DispatchDecision(
                    request_id=request.request_id,
                    robot_id=robot_id,
                    distance=best_distance,
                )
            )
        return decisions

    # ------------------------------------------------------------------
    def mark_delivered(self, request_id: int, step: int) -> None:
        request = self.requests[request_id]
        request.status = RequestStatus.DELIVERED
        request.delivered_step = step

    def release(self, robot_id: int, step: int) -> None:
        """Return a dead robot's request to the queue.

        The product still needs moving, so the request goes back to WAITING and
        is re-dispatched to someone else rather than being lost with the robot.
        """
        for request in self.requests.values():
            if (
                request.robot_id == robot_id
                and request.status is RequestStatus.ASSIGNED
            ):
                request.status = RequestStatus.WAITING
                request.robot_id = None
                request.assigned_step = None

    # ------------------------------------------------------------------
    def as_task(self, request: DeliveryRequest) -> Task:
        """Adapt a request to the Task the simulation already understands.

        Reusing Task keeps pickup/delivery handling, the status machine and the
        target-after-delivery rule identical to the single-robot engine, rather
        than growing a second copy of logic that is already tested.
        """
        return Task(
            task_id=request.request_id,
            pickup=request.pickup,
            dropoff=request.dropoff,
            created_step=request.created_step,
            status=TaskStatus.PENDING,
        )

    def stats(self) -> dict:
        counts: dict[str, int] = {status.value: 0 for status in RequestStatus}
        turnarounds: list[int] = []
        for request in self.requests.values():
            counts[request.status.value] += 1
            if request.delivered_step is not None:
                turnarounds.append(request.delivered_step - request.created_step)
        return {
            "total": len(self.requests),
            **counts,
            "mean_turnaround": (
                round(sum(turnarounds) / len(turnarounds), 2) if turnarounds else None
            ),
        }


def idle_robot_positions(
    members: Iterable, has_request: Callable[[int], bool]
) -> dict[int, Position]:
    """Positions of robots that are alive and not currently serving a request."""
    return {
        member.robot_id: member.position
        for member in members
        if member.alive and not has_request(member.robot_id)
    }
