"""Classical controller for a whole fleet: one decision per robot per tick.

Decentralised by design
-----------------------
Each robot is planned for independently, from its own position to its own
target, and no robot is told what the others intend to do. It sees only where
the others *are*. That is the honest description of what this does, and it is
what keeps a later comparison meaningful: swapping this controller for a
learned policy changes the decision rule and nothing else.

Fairness note, as for the single-robot baselines: this controller is given the
full map and the exact position of every other robot and every obstacle, while
a learned policy only ever sees its own local observation. The comparison is
therefore biased *in favour* of the classical side, which is the conservative
direction for any claim about reinforcement learning.

Why there is a static-map fallback
----------------------------------
Planning around peers can legitimately find no route at all - a robot boxed in
by its colleagues has none until one of them moves. Returning "wait" in that
case looks reasonable and quietly costs throughput, because a robot that never
attempts a move is also never recorded as blocked, so the congestion it is
suffering is invisible in the metrics. Falling back to a plan over the static
map means the robot attempts the move it would make if the floor were clear,
gets refused, and that refusal is counted. The single-robot
:class:`~baselines.controller.PlannerPolicy` does the same thing for the same
reason.
"""

from __future__ import annotations

import numpy as np

from simulation.engine import Action
from simulation.fleet import FleetSimulation
from simulation.navigation import DIRECTIONS, Position

from .astar import astar
from .bfs import bfs

PLANNERS = {"astar": astar, "bfs": bfs}

#: Multiplier on the charge a robot reserves to finish its leg and still reach
#: a charger. The single-robot PlannerPolicy uses 1.2, which is enough there
#: because nothing obstructs it. A fleet robot is repeatedly refused by peers,
#: so it spends ticks waiting and cells detouring that 1.2 never budgeted for -
#: measured, every fleet death was a robot already diverting to a charger that
#: died one to four cells short. The value here is set from the sweep recorded
#: in the commit message, not from reasoning.
#:
#: MEASURED on `default`, 600 ticks, 300 requests, seed 1000, fleets of 3 to 16:
#:   1.2  deadlocks at 3 and 8 robots, and loses robots at every size
#:   1.5  no deadlocks, still loses robots (2/3, 7/8, 7/12, 15/16 surviving)
#:   2.0  no deadlocks, every robot survives up to 8, best throughput (110 at 8)
#:   2.5  every robot survives at 12 and 16, but throughput falls (76 at 8)
#:   3.0  similar survival, no throughput gain
#: 2.0 is the knee: above it robots spend their time charging rather than
#: delivering, below it they die one to four cells short of a charger.
DEFAULT_SAFETY_FACTOR = 2.0


def _action_towards(current: Position, nxt: Position) -> int:
    delta = (nxt[0] - current[0], nxt[1] - current[1])
    try:
        return int(DIRECTIONS.index(delta))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{current} -> {nxt} is not a single 4-connected move") from exc


class FleetPlannerController:
    """Drives every robot with a shortest-path planner.

    Each tick it returns one action per living robot. A robot standing on its
    target - an idle robot already parked at its zone's waiting cell - waits
    rather than being given a pointless move.
    """

    def __init__(
        self,
        planner: str = "astar",
        battery_aware: bool = True,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
    ) -> None:
        if planner not in PLANNERS:
            raise ValueError(
                f"unknown planner {planner!r}, expected one of {sorted(PLANNERS)}"
            )
        self.planner_name = planner
        self._plan = PLANNERS[planner]
        #: When set, a robot that cannot finish its leg and still reach a
        #: charger diverts to charge first. Without this a fleet runs flat:
        #: on `default` a robot has 200 moves of charge and nothing else in the
        #: system routes it to power.
        self.battery_aware = battery_aware
        if safety_factor < 1.0:
            raise ValueError("safety_factor below 1.0 cannot even reach a charger")
        self.safety_factor = float(safety_factor)
        self.nodes_expanded = 0

    @property
    def name(self) -> str:
        return self.planner_name

    def reset(self, sim: FleetSimulation) -> None:
        self.nodes_expanded = 0

    # ------------------------------------------------------------------
    def act(self, sim: FleetSimulation) -> dict[int, int]:
        """One action per living robot."""
        actions: dict[int, int] = {}
        for robot_id, member in sim.members.items():
            if not member.alive:
                continue
            actions[robot_id] = self._act_one(sim, robot_id)
        return actions

    # ------------------------------------------------------------------
    def _act_one(self, sim: FleetSimulation, robot_id: int) -> int:
        member = sim.members[robot_id]
        position = member.position
        goal = self._goal_for(sim, robot_id)

        if position == goal:
            # Already where it wants to be: waiting here is the correct action,
            # and on a charging station it is also how the robot charges.
            return int(Action.WAIT)

        blocked = sim.blocked_grid(exclude=robot_id)
        blocked[position] = False
        plan = self._plan(blocked, position, goal)
        self.nodes_expanded += plan.nodes_expanded

        if not plan.found or plan.length == 0:
            # Boxed in by peers or traffic. Plan as if the floor were clear so
            # the attempt is made and the refusal is counted - see the module
            # docstring on why waiting here would hide the congestion.
            static_blocked = ~sim.walkable
            plan = self._plan(static_blocked, position, goal)
            self.nodes_expanded += plan.nodes_expanded
            if not plan.found or plan.length == 0:
                return int(Action.WAIT)

        return _action_towards(position, plan.path[1])

    def _goal_for(self, sim: FleetSimulation, robot_id: int) -> Position:
        """Where this robot should head: its task, a charger, or its waiting cell."""
        member = sim.members[robot_id]
        target = sim.target_for(robot_id)
        if not self.battery_aware:
            return target

        battery_cfg = sim.config.battery
        battery = member.robot.battery
        charger, to_charger = sim.nearest_charger(member.position)
        if to_charger < 0:  # pragma: no cover - layout validation prevents this
            return target

        if member.task is None:
            # Idle: top up first, then go and wait. An idle robot is the only
            # one that can afford a charging detour at no cost to a delivery,
            # so this is where a fleet should spend its charging time.
            if battery < battery_cfg.capacity * 0.9:
                return charger
            return target

        # Working: can it finish this leg and still reach power afterwards?
        to_target = sim.path_distance(member.position, target)
        from_target = sim.path_distance(target, charger)
        if to_target < 0 or from_target < 0:
            return target
        needed = (to_target + from_target) * battery_cfg.move_cost * self.safety_factor
        if battery <= needed or battery <= battery_cfg.critical_threshold:
            return charger
        return target
