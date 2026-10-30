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
from simulation.rmfs import allows, astar_directed

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
        directions=None,
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
        #: Charge level, as a fraction of capacity, at which a robot counts as
        #: full and stops diverting to top up.
        self.charge_until = 0.9
        if safety_factor < 1.0:
            raise ValueError("safety_factor below 1.0 cannot even reach a charger")
        self.safety_factor = float(safety_factor)
        #: Optional one-way grid from simulation.rmfs. When present the planner
        #: only ever proposes a legal first move, which is the structural fix
        #: for the aisle standoff measured in Phase 2 - on a one-way lane two
        #: robots cannot face each other at all.
        self.directions = directions
        self.nodes_expanded = 0
        #: robot id -> the task it left a charger full for. A committed robot
        #: does not turn back for a top-up unless its battery goes critical.
        self._committed: dict[int, object] = {}
        #: robots that have decided to charge. Such a robot keeps heading for
        #: (or sitting on) a charger until it reaches ``charge_until`` - the
        #: second threshold of a two-threshold hysteresis.
        self._charging: set[int] = set()

    @property
    def name(self) -> str:
        return self.planner_name

    def reset(self, sim: FleetSimulation) -> None:
        self.nodes_expanded = 0
        self._committed = {}
        self._charging = set()

    # ------------------------------------------------------------------
    def act(self, sim: FleetSimulation) -> dict[int, int]:
        """One action per living robot."""
        actions: dict[int, int] = {}
        for robot_id, member in sim.members.items():
            if not member.alive:
                continue
            actions[robot_id] = self._act_one(sim, robot_id)
        return self._make_way(sim, actions)

    # ------------------------------------------------------------------
    def _make_way(self, sim: FleetSimulation, actions: dict[int, int]) -> dict[int, int]:
        """Idle robots step aside for a robot that needs their cell.

        Without this an idle robot, once parked, never moves for anyone. On the
        21x15 floor the waiting cells beside a charger form a diagonal that
        three parked robots can turn into a wall: a robot that needed charge
        could not get in, a robot delivering to the station next to it could
        not get past, and the floor stopped as deadlocked with a product in
        flight. Measured before this rule: 14 of 60 stress runs (3-8 robots,
        bursts, steady arrivals and long gaps, four seeds) deadlocked or left a
        product undelivered.

        Only a robot with no task makes way, so a delivery is never delayed to
        let someone else through, and it moves only when it was not leaving
        anyway. It steps to a free neighbouring cell nobody else is entering,
        preferring the one furthest from the robot it is making way for; next
        tick it heads back to the nearest free waiting cell as usual.
        """
        positions = {rid: m.position for rid, m in sim.members.items() if m.alive}
        at = {pos: rid for rid, pos in positions.items()}
        intended = {
            rid: (positions[rid][0] + DIRECTIONS[a][0], positions[rid][1] + DIRECTIONS[a][1])
            if a != int(Action.WAIT) else positions[rid]
            for rid, a in actions.items()
        }
        obstacles = set(sim.obstacles.positions)
        claimed = set(intended.values())
        height, width = sim.walkable.shape

        for rid in sorted(actions):
            target = intended[rid]
            if target == positions[rid]:
                continue
            blocker = at.get(target)
            if blocker is None or blocker == rid or blocker not in actions:
                continue
            blocker_member = sim.members[blocker]
            if blocker_member.task is not None:
                continue
            # Only a robot with charge to spare makes way. One below the
            # top-up level is on its way to a charger (or charging on one);
            # pushing it aside instead was measured to run two robots flat,
            # one of them on a packing station, which then blocked it for good.
            #
            # Except head-on: if the blocker wants the requester's own cell,
            # neither can ever move, whatever the blocker's charge. That is the
            # charger doorway - a full robot trying to leave a charger while
            # the robots queued to enter it occupy both exits. Measured with 8
            # robots on the 21x15 floor: the queue waited there until six
            # robots ran flat. The blocker steps aside; it is next in line.
            capacity = sim.config.battery.capacity
            head_on = intended[blocker] == positions[rid]
            if blocker_member.robot.battery < capacity * self.charge_until and not head_on:
                continue
            leaving = intended[blocker]
            if leaving != positions[blocker] and leaving not in at:
                continue  # it is already moving out of the way
            origin = positions[blocker]
            options = []
            for index, (d_row, d_col) in enumerate(DIRECTIONS):
                cell = (origin[0] + d_row, origin[1] + d_col)
                if not (0 <= cell[0] < height and 0 <= cell[1] < width):
                    continue
                if not sim.walkable[cell] or cell in at or cell in claimed or cell in obstacles:
                    continue
                if self.directions is not None and not allows(self.directions, origin, cell):
                    continue
                away = abs(cell[0] - positions[rid][0]) + abs(cell[1] - positions[rid][1])
                options.append((-away, index, cell))
            if not options:
                continue
            _, _, cell = min(options)
            actions[blocker] = _action_towards(origin, cell)
            claimed.add(cell)
            intended[blocker] = cell
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
        if self.directions is not None:
            # The one-way rule has to be INSIDE the search. Filtering only the
            # first move returns routes that are illegal from step two, the
            # controller replans, proposes the same route, and the robot
            # thrashes - measured at 1 delivery against 19.
            path = astar_directed(blocked, self.directions, position, goal)
            if len(path) >= 2:
                return _action_towards(position, path[1])
            return int(Action.WAIT)
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
            self._committed.pop(robot_id, None)
            # Idle: top up first, then go and wait. An idle robot is the only
            # one that can afford a charging detour at no cost to a delivery,
            # so this is where a fleet should spend its charging time.
            if battery < battery_cfg.capacity * 0.9:
                return charger
            return target

        # Working: can it finish this leg and still reach power afterwards?
        # "Afterwards" means the charger nearest the TARGET - the one it would
        # actually use once there. Measuring the return to the charger nearest
        # the robot's current cell overstated a long trip on the realistic floor
        # (96 needed against a real 71), which fed the loop described below.
        to_target = sim.path_distance(member.position, target)
        _, from_target = sim.nearest_charger(target)
        if to_target < 0 or from_target < 0:
            return target
        needed = (to_target + from_target) * battery_cfg.move_cost * self.safety_factor

        # Charging is only worth a detour while it would actually add charge.
        # On a large floor the reserve can exceed the battery outright - a trip
        # of 60 cells plus a 45-cell return needs 105 against a capacity of 100 -
        # and a robot that diverts on an unsatisfiable reserve parks at a
        # charger at 100% for ever, because topping up can never clear the
        # condition. Measured on the realistic floor: a package picked up at
        # tick 30 was still undelivered at tick 340, the robot sitting on a
        # charger it no longer needed.
        #
        # So: once effectively full, go. There is nothing better available, and
        # running out en route is a real outcome the metrics should record
        # rather than one the controller hides by refusing to start.
        #
        # And once it has left a charger full for this task it is COMMITTED:
        # it only turns back if the battery goes critical. Without that there
        # is no hysteresis - it charged to 90%, drove a few cells, dipped under
        # 90%, re-checked the reserve, turned back, and repeated for ever while
        # holding the product. Measured on the realistic floor (8 robots, 25
        # products, seeds 0 and 2): two products each still undelivered after
        # 3,000 ticks, with robots shuttling between a charger and a lane.
        #
        # The other half of the hysteresis: once it DECIDES to charge, it
        # charges to ``charge_until`` before going back to work. Without this a
        # robot sitting just above the reserve line charged for a tick, judged
        # it had enough, left, judged it had not, and came back - for ever,
        # while holding the product and occupying the only charger's approach.
        # Measured on the 21x15 floor (3 robots, steady arrivals, seed 2): two
        # other robots circled that corner trying to reach the charger until
        # their batteries ran flat, and the floor deadlocked.
        effectively_full = battery >= battery_cfg.capacity * self.charge_until
        if effectively_full:
            self._charging.discard(robot_id)
            self._committed[robot_id] = member.task
            return target
        if robot_id in self._charging:
            return charger
        if battery <= battery_cfg.critical_threshold:
            self._committed.pop(robot_id, None)
            self._charging.add(robot_id)
            return charger
        if self._committed.get(robot_id) is member.task:
            return target
        if battery <= needed:
            self._charging.add(robot_id)
            return charger
        return target
