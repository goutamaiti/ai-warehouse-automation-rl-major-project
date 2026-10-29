"""Multi-robot warehouse simulation: several robots on one floor.

Relationship to :mod:`simulation.engine`
---------------------------------------
This module is a *sibling* of :class:`~simulation.engine.WarehouseSimulation`,
never a replacement. The single-robot engine produces every result committed
under ``data/results/`` and every policy under ``rl_agent/models/``, so it is
frozen: ``tests/test_frozen_contract.py`` fails if its behaviour moves at all.

Everything shared is reused unchanged - :mod:`simulation.warehouse` for the
map, :mod:`simulation.navigation` for grid primitives, :mod:`simulation.robot`
for a robot's state and battery bookkeeping, :mod:`simulation.tasks` for
delivery jobs. The only thing this file adds is what happens when more than one
robot wants the same floor at the same time.

As in the single-robot engine, this module reports *facts* and never a reward:
no reward is computed here, nothing is rendered here, and no policy lives here.

Scope note
----------
The project memo lists multiple robots as the extension to build once
single-robot PPO works, and multi-agent RL (MAPPO and similar) as future scope.
This is the former: each robot is driven independently, and nothing here learns.
In this first version each robot carries its own independent task, exactly as
the memo describes; a shared task queue with dispatch is the next phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ScenarioConfig
from .dispatch import DispatchDecision, TaskQueue
from .engine import Action
from .navigation import DIRECTIONS, Position, bfs_distance_field
from .obstacles import ObstacleField
from .robot import Robot
from .tasks import Task, TaskGenerator, TaskStatus
from .warehouse import CellType, WarehouseLayout, build_layout
from .zones import Zone, build_zones, zone_of

#: Consecutive steps with no fleet-wide progress before the run is declared
#: deadlocked. "Progress" is deliberately broad - any move, pickup, delivery or
#: charge counts - because a robot sitting on a charger is working, not stuck.
#: Eight is long enough that ordinary contention (two robots taking turns down
#: one aisle) never trips it, and short enough to catch a real standoff before
#: a continuous session wastes minutes on it.
DEADLOCK_STREAK = 8

#: Consecutive refusals by a peer before a robot steps aside to break the
#: standoff. Three is long enough that ordinary contention resolves on its own -
#: a robot waiting one or two ticks for an aisle to clear is normal - and short
#: enough to break a genuine standoff well inside DEADLOCK_STREAK.
#:
#: Why the engine does this and not the controller: the id-priority rule makes
#: swaps impossible, so two robots facing each other in a one-wide aisle retry
#: forever. That standoff is a property of the arbitration rule, so the remedy
#: belongs with it. It also has to be here for the comparison to stay fair - a
#: learned policy has no observation feature for "I have been blocked three
#: times", so requiring controllers to break standoffs would measure
#: deadlock-handling rather than navigation, which is what the project studies.
#:
#: MEASURED: off by default, because it did not earn its place. On `default`
#: with the shared queue, 180 ticks, seeds from 1000, yielding fired between 0
#: and 11 times across fleets of 3 to 16 robots and showed no consistent
#: benefit - at eight robots it was actively worse (27 delivered against 38
#: with it disabled). The standoff it was built for is already prevented by
#: exclusive assignment, which stops two robots being routed to the same
#: pickup in the first place.
#:
#: The mechanism is kept, tested and opt-in rather than deleted: it is the
#: documented remedy if a future layout with long one-wide aisles reintroduces
#: standoffs. Set to a positive number to enable it.
DEFAULT_YIELD_AFTER = 0


@dataclass
class RobotCounters:
    """Per-robot tally of things that actually happened.

    Mirrors :class:`~simulation.engine.EpisodeCounters` field for field where
    the meaning is the same, so fleet numbers can be read against single-robot
    ones without a translation table. ``blocked_by_robot`` is the one addition:
    it is the cost of sharing a floor, and keeping it separate from
    ``dynamic_collisions`` is what makes congestion measurable rather than
    hidden inside a collision count.
    """

    steps: int = 0
    moves: int = 0
    static_collisions: int = 0
    dynamic_collisions: int = 0
    #: Moves refused because another robot held, or won, the target cell.
    blocked_by_robot: int = 0
    #: Times this robot stepped aside to break a standoff.
    yields: int = 0
    idle_steps: int = 0
    energy_consumed: float = 0.0
    charging_events: int = 0
    tasks_delivered: int = 0
    optimal_path_length: int = 0

    @property
    def collisions(self) -> int:
        """Collisions with the world. Robot-on-robot blocking is counted apart."""
        return self.static_collisions + self.dynamic_collisions

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "moves": self.moves,
            "collisions": self.collisions,
            "static_collisions": self.static_collisions,
            "dynamic_collisions": self.dynamic_collisions,
            "blocked_by_robot": self.blocked_by_robot,
            "yields": self.yields,
            "idle_steps": self.idle_steps,
            "energy_consumed": round(self.energy_consumed, 3),
            "charging_events": self.charging_events,
            "tasks_delivered": self.tasks_delivered,
            "optimal_path_length": self.optimal_path_length,
        }


@dataclass
class FleetMember:
    """One robot, its current job and its tally."""

    robot: Robot
    #: The job in hand. ``None`` means idle - in queue mode a robot waits with
    #: no task until dispatch gives it one.
    task: Task | None
    counters: RobotCounters = field(default_factory=RobotCounters)
    #: The zone this robot waits in when idle. Not an assignment boundary:
    #: dispatch sends the nearest idle robot anywhere on the floor.
    home_zone: str = ""
    #: Which of its zone's waiting cells this robot parks on.
    waiting_point: Position | None = None
    #: Consecutive ticks this robot has been refused by a peer. Reset by any
    #: successful move; drives the yielding rule.
    blocked_streak: int = 0
    #: False once the robot's battery is flat; it stops acting but stays on the
    #: floor as an obstacle, because a dead robot does not vanish from an aisle.
    alive: bool = True

    @property
    def robot_id(self) -> int:
        return self.robot.robot_id

    @property
    def position(self) -> Position:
        return self.robot.position

    def to_dict(self, config) -> dict:
        return {
            **self.robot.to_dict(config),
            "task": self.task.to_dict() if self.task else None,
            "home_zone": self.home_zone,
            "waiting_point": list(self.waiting_point) if self.waiting_point else None,
            "alive": self.alive,
            "counters": self.counters.to_dict(),
        }


@dataclass
class RobotStepOutcome:
    """What happened to one robot during one tick. Facts only, no reward."""

    robot_id: int
    moved: bool = False
    blocked_by_static: bool = False
    blocked_by_obstacle: bool = False
    blocked_by_robot: bool = False
    #: True when the engine turned a refused move into a sidestep.
    yielded: bool = False
    waited: bool = False
    picked_up: bool = False
    delivered: bool = False
    charged: float = 0.0
    energy_used: float = 0.0
    battery_depleted: bool = False
    events: list[str] = field(default_factory=list)

    @property
    def collided(self) -> bool:
        """World collisions only - being blocked by a peer is not a collision."""
        return self.blocked_by_static or self.blocked_by_obstacle


@dataclass
class FleetStepOutcome:
    """One tick of the whole fleet."""

    robots: dict[int, RobotStepOutcome] = field(default_factory=dict)
    #: True when no robot made any progress for DEADLOCK_STREAK ticks.
    deadlocked: bool = False
    #: True when every robot's battery is flat.
    all_dead: bool = False

    @property
    def progressed(self) -> bool:
        """Did anything useful happen anywhere in the fleet this tick?"""
        return any(
            o.moved or o.yielded or o.picked_up or o.delivered or o.charged > 0
            for o in self.robots.values()
        )

    def events(self) -> list[tuple[int, str]]:
        return [(rid, e) for rid, o in self.robots.items() for e in o.events]


class FleetSimulation:
    """Several robots delivering packages on one shared warehouse floor.

    Movement conflicts
    ------------------
    Every robot names a target cell, then conflicts are resolved **in ascending
    robot-id order**: the lowest id that wants a cell gets it, and anyone else
    who wanted it is refused and stays put (recorded as ``blocked_by_robot``).

    Two consequences of that rule are worth stating because they are visible in
    any replay:

    * **A convoy moves only in id order.** A robot may step into a cell that a
      *lower-numbered* robot has already vacated this tick, so robot 1 can
      follow robot 0 immediately. Following in the other direction costs a
      tick, because robot 0 resolves while robot 1 is still in the way.
    * **Swaps are impossible.** Two robots facing each other both find the
      other still in place when they resolve, so neither moves and they do not
      pass through one another. A standoff like that is exactly what
      :data:`DEADLOCK_STREAK` exists to notice.

    Fleet size is not capped - it is a configuration choice - but a fleet that
    cannot physically be placed is rejected at construction rather than failing
    obscurely later, and a fleet that jams is reported as ``deadlocked`` rather
    than silently spinning.
    """

    def __init__(
        self,
        config: ScenarioConfig,
        n_robots: int = 3,
        layout: WarehouseLayout | None = None,
        task_queue: TaskQueue | None = None,
        yield_after: int = DEFAULT_YIELD_AFTER,
    ) -> None:
        if n_robots < 1:
            raise ValueError(f"a fleet needs at least one robot, got {n_robots}")
        if yield_after < 0:
            raise ValueError("yield_after must be zero (disabled) or positive")
        self.yield_after = int(yield_after)
        self.config = config
        self.n_robots = int(n_robots)
        self.layout = layout if layout is not None else build_layout(config.layout)
        self._walkable = self.layout.walkable_mask()
        # Enough waiting cells for the whole fleet in any single zone. With a
        # fixed cap, a fleet larger than the cap left robots with no waiting
        # point at all, and target_for() then returned their current cell - so
        # they froze wherever they happened to stop, mid-aisle, as permanent
        # obstacles. Measured at twelve robots: three frozen, and throughput
        # below the eight-robot fleet's.
        self.zones: tuple[Zone, ...] = build_zones(
            self.layout, max_waiting_points=max(8, int(n_robots))
        )
        #: When set, robots draw work from this shared queue instead of each
        #: generating its own task. This is what stops two robots being routed
        #: to the same storage point, which is how the five-robot run jammed.
        self.queue = task_queue
        #: Assignments made on the most recent tick, for replays and the API.
        self.last_dispatch: list[DispatchDecision] = []

        start_cells = self._candidate_start_cells()
        if self.n_robots > len(start_cells):
            raise ValueError(
                f"cannot place {self.n_robots} robots: this layout offers "
                f"{len(start_cells)} usable start cells "
                f"({len(self.layout.charging_stations)} charging stations plus "
                "reachable free floor beside them). Reduce the fleet or enlarge "
                "the warehouse."
            )

        self.rng = np.random.default_rng(config.seed)
        self.members: dict[int, FleetMember] = {}
        self.obstacles: ObstacleField
        self._target_fields: dict[Position, np.ndarray] = {}
        self._idle_streak = 0
        self._done = False
        self.reset(seed=config.seed)

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------
    def _candidate_start_cells(self) -> list[Position]:
        """Deterministic start cells: chargers first, then the floor beside them.

        Robots start on charging stations so a battery-constrained fleet begins
        in the same state a single robot does. Once the stations run out the
        fleet spills onto the nearest reachable free cells, ordered by distance
        from the first station and then by position, so the list is a pure
        function of the layout - no RNG, and identical on every run.

        Station cells themselves are excluded from the spill-over so a robot
        never starts parked on a packing station.
        """
        stations = list(self.layout.charging_stations)
        reserved = set(stations) | set(self.layout.packing_stations)
        fields = [bfs_distance_field(self._walkable, station) for station in stations]

        # Spill-over is taken round-robin from each zone in turn, nearest that
        # zone's own charger first. Sorting the whole floor by distance from
        # stations[0] instead - which is what this did - clustered every extra
        # robot around one charger: at twelve robots the homes split 11 to 1,
        # so one zone was crushed while the other's waiting cells went unused.
        per_zone: list[list[Position]] = []
        for index, station in enumerate(stations):
            field = fields[index]
            own = [
                cell
                for cell in (
                    (int(r), int(c)) for r, c in zip(*np.nonzero(self._walkable))
                )
                if cell not in reserved
                and field[cell] >= 0
                # The cell belongs to this zone: no other charger is nearer.
                # min() breaks ties on the index, matching build_zones.
                and min(
                    (int(f[cell]), i) for i, f in enumerate(fields) if f[cell] >= 0
                )[1]
                == index
            ]
            own.sort(key=lambda cell: (int(field[cell]), cell))
            per_zone.append(own)

        spill: list[Position] = []
        for rank in range(max((len(cells) for cells in per_zone), default=0)):
            for cells in per_zone:
                if rank < len(cells):
                    spill.append(cells[rank])
        return stations + spill

    # ------------------------------------------------------------------
    # episode life cycle
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Start a new run. The same seed rebuilds the same run exactly."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        starts = self._candidate_start_cells()[: self.n_robots]
        self._task_generator = TaskGenerator(self.layout, self.config.tasks, self.rng)

        self.members = {}
        taken: dict[str, int] = {}
        for robot_id, start in enumerate(starts):
            robot = Robot(
                robot_id=robot_id,
                position=start,
                battery=min(
                    self.config.battery.start_level, self.config.battery.capacity
                ),
            )
            task = None if self.queue is not None else self._task_generator.generate(0)
            member = FleetMember(robot=robot, task=task)
            if task is not None:
                member.counters.optimal_path_length = self._optimal_legs(start, task)

            # Home zone from the start cell, and the next free waiting cell in
            # it, so two robots homed in one zone never park on top of each other.
            zone = zone_of(self.zones, start)
            if zone is not None:
                member.home_zone = zone.name
                index = taken.get(zone.name, 0)
                taken[zone.name] = index + 1
                if index < len(zone.waiting_points):
                    member.waiting_point = zone.waiting_points[index]
            self.members[robot_id] = member

        forbidden = set(starts)
        forbidden.update(self.layout.charging_stations)
        forbidden.update(self.layout.packing_stations)
        for member in self.members.values():
            if member.task is not None:
                forbidden.update({member.task.pickup, member.task.dropoff})
            if member.waiting_point is not None:
                forbidden.add(member.waiting_point)
        self.obstacles = ObstacleField.spawn(
            self._walkable, self.config.obstacles, self.rng, forbidden
        )

        self._target_fields = {}
        self._idle_streak = 0
        self._done = False
        self.last_dispatch = []

    @property
    def done(self) -> bool:
        return self._done

    def has_outstanding_work(self) -> bool:
        """Is there anything the fleet is currently supposed to be doing?

        True while any living robot holds a task, or any request is still
        waiting to be assigned. False means the floor has caught up - which is a
        normal state for a warehouse that never resets, not a failure.
        """
        if any(m.task is not None for m in self.members.values() if m.alive):
            return True
        if self.queue is not None and self.queue.waiting():
            return True
        return False

    @property
    def positions(self) -> dict[int, Position]:
        return {rid: m.position for rid, m in self.members.items()}

    @property
    def walkable(self) -> np.ndarray:
        """Static walkability mask; peers and traffic excluded."""
        return self._walkable

    # ------------------------------------------------------------------
    # one tick
    # ------------------------------------------------------------------
    def step(self, actions: dict[int, int]) -> FleetStepOutcome:
        """Advance every robot by one tick.

        ``actions`` maps robot id to an :class:`~simulation.engine.Action`. A
        robot missing from the mapping, or one whose battery is flat, holds
        position.
        """
        if self._done:
            raise RuntimeError("fleet run is finished; call reset() first")

        # Reconcile aliveness before anyone acts. A robot is retired by its
        # battery, not by a flag, so a robot that is already flat must never
        # get a move - however its charge reached zero.
        for robot_id, member in self.members.items():
            if member.alive and member.robot.is_depleted():
                member.alive = False
                if member.task is not None:
                    member.task.mark_failed()
                if self.queue is not None:
                    self.queue.release(robot_id, member.counters.steps)
                    member.task = None

        # Dispatch before anyone moves, so a request added this tick is being
        # served on this tick - which is what "the robot should start moving the
        # instant I add the location" requires.
        self.last_dispatch = self._dispatch()

        outcome = FleetStepOutcome()
        resolved = self._resolve_moves(actions, outcome)

        for robot_id in sorted(self.members):
            member = self.members[robot_id]
            robot_outcome = outcome.robots[robot_id]
            if not member.alive:
                continue

            if robot_id in resolved:
                member.robot.position = resolved[robot_id]

            member.robot.consume(robot_outcome.energy_used)
            member.counters.energy_consumed += robot_outcome.energy_used
            member.counters.steps += 1

            self._handle_interactions(member, robot_outcome)
            self._handle_charging(member, robot_outcome)

            if member.robot.is_depleted():
                member.alive = False
                robot_outcome.battery_depleted = True
                robot_outcome.events.append("battery_depleted")
                if member.task is not None:
                    member.task.mark_failed()
                if self.queue is not None:
                    # The product still needs moving, so it returns to the
                    # queue for someone else rather than dying with the robot.
                    self.queue.release(robot_id, member.counters.steps)
                    member.task = None

        # The rest of the world moves once, after every robot has acted, and
        # routes around every robot - living or dead.
        self.obstacles.step(
            self._walkable, self.rng, blocked=set(self.positions.values())
        )

        # A fleet is only deadlocked if it has work it cannot make progress on.
        # An idle fleet parked at its waiting points with an empty queue makes
        # no moves, pickups, deliveries or charges either - and it is not stuck,
        # it is waiting for the next request. Conflating the two would stop a
        # continuous warehouse the moment it caught up with its queue.
        #
        # Gating the CHECK on outstanding work was not enough, because the
        # streak kept climbing through every idle tick. The threshold was then
        # already satisfied when work finally arrived, and the first tick after
        # a product is added makes no progress by construction - the dispatcher
        # assigns it, and the robot only moves on the tick after. Measured
        # before this fix: a floor left idle for 13 ticks declared a deadlock on
        # the exact tick a request was added, set done, and never delivered it -
        # which is the Live floor's whole interaction. So an idle tick resets
        # the streak instead of accumulating: it counts consecutive ticks of no
        # progress *while there is work*, which is what it is documented to be.
        has_work = self.has_outstanding_work()
        self._idle_streak = (
            0 if (outcome.progressed or not has_work) else self._idle_streak + 1
        )
        outcome.deadlocked = has_work and self._idle_streak >= DEADLOCK_STREAK
        outcome.all_dead = all(not m.alive for m in self.members.values())
        self._done = outcome.deadlocked or outcome.all_dead
        return outcome

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _resolve_moves(
        self, actions: dict[int, int], outcome: FleetStepOutcome
    ) -> dict[int, Position]:
        """Decide who actually moves, lowest robot id first.

        ``occupied`` is updated as each robot resolves, which is what lets a
        robot follow a lower-numbered one into a cell vacated this very tick -
        and what stops two robots swapping places, since each still sees the
        other in the way when its turn comes.
        """
        battery_cfg = self.config.battery
        occupied = dict(self.positions)
        moves: dict[int, Position] = {}

        for robot_id in sorted(self.members):
            member = self.members[robot_id]
            robot_outcome = RobotStepOutcome(robot_id=robot_id)
            outcome.robots[robot_id] = robot_outcome

            if not member.alive:
                continue

            action = Action(int(actions.get(robot_id, Action.WAIT)))
            if action is Action.WAIT:
                robot_outcome.waited = True
                robot_outcome.energy_used = battery_cfg.idle_cost
                member.counters.idle_steps += 1
                continue

            delta = DIRECTIONS[int(action)]
            target = (member.position[0] + delta[0], member.position[1] + delta[1])

            if not self.layout.is_walkable(target):
                robot_outcome.blocked_by_static = True
                robot_outcome.events.append("collision_static")
                member.counters.static_collisions += 1
                robot_outcome.energy_used = battery_cfg.idle_cost
            elif target in self.obstacles.positions:
                robot_outcome.blocked_by_obstacle = True
                robot_outcome.events.append("collision_obstacle")
                member.counters.dynamic_collisions += 1
                robot_outcome.energy_used = battery_cfg.idle_cost
            elif target in occupied.values():
                robot_outcome.blocked_by_robot = True
                member.counters.blocked_by_robot += 1
                member.blocked_streak += 1
                sidestep = (
                    self._sidestep(member, target, occupied)
                    if self.yield_after and member.blocked_streak >= self.yield_after
                    else None
                )
                if sidestep is not None:
                    # Step aside to break the standoff. Recorded as a yield, not
                    # as a move the controller asked for, so a replay shows the
                    # engine intervened rather than the policy changing its mind.
                    moves[robot_id] = sidestep
                    occupied[robot_id] = sidestep
                    robot_outcome.yielded = True
                    robot_outcome.events.append("yielded")
                    member.counters.yields += 1
                    member.counters.moves += 1
                    member.blocked_streak = 0
                    robot_outcome.energy_used = battery_cfg.move_cost
                else:
                    robot_outcome.events.append("blocked_by_robot")
                    robot_outcome.energy_used = battery_cfg.idle_cost
            else:
                moves[robot_id] = target
                occupied[robot_id] = target
                robot_outcome.moved = True
                robot_outcome.energy_used = battery_cfg.move_cost
                member.counters.moves += 1
                member.blocked_streak = 0

        return moves

    def _sidestep(
        self,
        member: FleetMember,
        refused: Position,
        occupied: dict[int, Position],
    ) -> Position | None:
        """A free neighbouring cell to retreat into, or ``None`` if boxed in.

        Candidates are every walkable neighbour that is free of racking, traffic
        and peers, excluding the cell the robot was just refused - stepping
        there is what it already failed to do. Among those, the cell *furthest*
        from the refused cell is chosen, so the robot genuinely clears the
        contested spot instead of shuffling alongside it; ties break on grid
        order, keeping the choice deterministic and a run replayable.

        A robot with nowhere to go is left blocked, and the deadlock detector
        remains the backstop for a fleet that is wedged solid.
        """
        taken = set(occupied.values()) | set(self.obstacles.positions)
        candidates = []
        for d_row, d_col in DIRECTIONS:
            cell = (member.position[0] + d_row, member.position[1] + d_col)
            if cell == refused or cell in taken:
                continue
            if not self.layout.is_walkable(cell):
                continue
            candidates.append(cell)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda cell: (
                abs(cell[0] - refused[0]) + abs(cell[1] - refused[1]),
                -cell[0],
                -cell[1],
            ),
        )

    def _handle_interactions(
        self, member: FleetMember, outcome: RobotStepOutcome
    ) -> None:
        """Pickup and delivery happen automatically at the right cell.

        Same rule as the single-robot engine, and for the same reason: explicit
        pickup/drop actions would enlarge the action space without changing the
        navigation problem being studied.
        """
        position = member.position
        task = member.task
        if task is None:
            return
        if task.status is TaskStatus.PENDING and position == task.pickup:
            task.mark_picked_up(member.counters.steps)
            member.robot.carrying = True
            outcome.picked_up = True
            outcome.events.append("pickup")
        elif task.status is TaskStatus.PICKED_UP and position == task.dropoff:
            task.mark_delivered(member.counters.steps)
            member.robot.carrying = False
            outcome.delivered = True
            outcome.events.append("delivery")
            member.counters.tasks_delivered += 1
            if self.queue is not None:
                # Request complete; the robot goes idle and becomes a candidate
                # for the next dispatch rather than inventing its own next job.
                self.queue.mark_delivered(task.task_id, member.counters.steps)
                member.task = None
            else:
                # Independent-task mode: the robot picks up a fresh job, so a
                # fleet without a queue keeps working instead of stopping.
                member.task = self._task_generator.generate(step=member.counters.steps)
                member.counters.optimal_path_length += self._optimal_legs(
                    position, member.task
                )
                outcome.events.append("new_task")

    def _handle_charging(self, member: FleetMember, outcome: RobotStepOutcome) -> None:
        if self.layout.cell(member.position) is CellType.CHARGING:
            gained = member.robot.charge(self.config.battery)
            if gained > 0:
                outcome.charged = gained
                outcome.events.append("charging")
                member.counters.charging_events += 1

    def _dispatch(self) -> list[DispatchDecision]:
        """Hand waiting requests to the nearest idle robots."""
        if self.queue is None:
            return []
        idle = {
            robot_id: member.position
            for robot_id, member in self.members.items()
            if member.alive and member.task is None
        }
        if not idle:
            return []

        step = max((m.counters.steps for m in self.members.values()), default=0)
        decisions = self.queue.dispatch(idle, self.path_distance, step=step)
        for decision in decisions:
            member = self.members[decision.robot_id]
            request = self.queue.requests[decision.request_id]
            member.task = self.queue.as_task(request)
            member.counters.optimal_path_length += self._optimal_legs(
                member.position, member.task
            )
        return decisions

    def path_distance(self, origin: Position, destination: Position) -> int:
        """Cells driven through the aisles, or -1 if there is no route.

        Geodesic, not straight-line: on a floor of shelf blocks a robot one
        aisle over can be metres away and twenty cells of driving away, and
        dispatching on straight-line distance would pick it over a better robot.
        """
        return int(self._field(destination)[origin])

    def target_for(self, robot_id: int) -> Position:
        """Where a robot is currently heading.

        A robot with a job heads for its task target. An idle robot heads for
        the **nearest free** waiting cell, anywhere on the floor.

        Nearest rather than a cell assigned up front, because dispatch sends the
        nearest idle robot to any request, so a robot has no stable work region
        to be "from". Pinning each robot to a home zone's cell made it commute:
        measured at eight robots, every one of them ended up jammed in the
        corner by one charger while four of their assigned waiting cells sat
        empty at the far end, 3,145 of 3,147 blocked moves happened to robots
        carrying work, and throughput fell from 110 deliveries to 34.

        Cells currently occupied are skipped. Two robots can still choose the
        same cell on the same tick; the id-priority rule settles that, and the
        loser picks another next tick - which is cheaper than coordinating
        reservations for somewhere to stand idle.
        """
        member = self.members[robot_id]
        if member.task is not None:
            return member.task.target

        taken = {
            other.position
            for other_id, other in self.members.items()
            if other_id != robot_id
        }
        best: Position | None = None
        best_distance = -1
        for zone in self.zones:
            for cell in zone.waiting_points:
                if cell in taken:
                    continue
                distance = int(self._field(cell)[member.position])
                if distance < 0:
                    continue
                if best is None or (distance, cell) < (best_distance, best):
                    best, best_distance = cell, distance
        if best is not None:
            return best
        # Every waiting cell is taken or unreachable: hold station rather than
        # wander, and leave the aisle-blocking to the deadlock detector.
        return member.position

    def _optimal_legs(self, start: Position, task: Task) -> int:
        """Shortest achievable length for a task, ignoring traffic and peers."""
        to_pickup = self._field(task.pickup)[start]
        to_dropoff = self._field(task.dropoff)[task.pickup]
        return int(to_pickup) + int(to_dropoff)

    def _field(self, target: Position) -> np.ndarray:
        """Distance field to a cell, cached - the static map never changes."""
        if target not in self._target_fields:
            self._target_fields[target] = bfs_distance_field(self._walkable, target)
        return self._target_fields[target]

    # ------------------------------------------------------------------
    # views for other layers
    # ------------------------------------------------------------------
    def distance_to_target(self, robot_id: int) -> int:
        member = self.members[robot_id]
        return int(self._field(self.target_for(robot_id))[member.position])

    def nearest_charger(self, position: Position) -> tuple[Position, int]:
        """Closest reachable charging station by path length, not straight line."""
        best, best_distance = self.layout.charging_stations[0], -1
        for station in self.layout.charging_stations:
            distance = int(self._field(station)[position])
            if distance >= 0 and (best_distance < 0 or distance < best_distance):
                best, best_distance = station, distance
        return best, best_distance

    def blocked_grid(self, exclude: int | None = None) -> np.ndarray:
        """Cells a robot may not enter: walls, shelves, traffic and peers.

        ``exclude`` leaves one robot's own cell out, so a controller planning
        for that robot does not treat itself as an obstacle.
        """
        blocked = ~self._walkable
        blocked = blocked | self.obstacles.occupancy(self.layout.shape)
        for robot_id, position in self.positions.items():
            if robot_id != exclude:
                blocked[position] = True
        return blocked

    def fleet_counters(self) -> dict:
        """Fleet totals plus the per-robot breakdown."""
        per_robot = {rid: m.counters for rid, m in self.members.items()}
        return {
            "n_robots": self.n_robots,
            "steps": max((c.steps for c in per_robot.values()), default=0),
            "moves": sum(c.moves for c in per_robot.values()),
            "collisions": sum(c.collisions for c in per_robot.values()),
            "blocked_by_robot": sum(c.blocked_by_robot for c in per_robot.values()),
            "yields": sum(c.yields for c in per_robot.values()),
            "idle_steps": sum(c.idle_steps for c in per_robot.values()),
            "energy_consumed": round(
                sum(c.energy_consumed for c in per_robot.values()), 3
            ),
            "charging_events": sum(c.charging_events for c in per_robot.values()),
            "tasks_delivered": sum(c.tasks_delivered for c in per_robot.values()),
            "robots_alive": sum(1 for m in self.members.values() if m.alive),
            "deadlock_streak": self._idle_streak,
            "outstanding_work": self.has_outstanding_work(),
            "per_robot": {rid: c.to_dict() for rid, c in per_robot.items()},
        }

    def snapshot(self, events: list[tuple[int, str]] | None = None) -> dict:
        """JSON-serialisable state, for the API and the dashboard."""
        return {
            "step": max((m.counters.steps for m in self.members.values()), default=0),
            "robots": [
                self.members[rid].to_dict(self.config.battery)
                for rid in sorted(self.members)
            ],
            "obstacles": self.obstacles.to_list(),
            "events": [list(e) for e in (events or [])],
            "deadlocked": self.has_outstanding_work()
            and self._idle_streak >= DEADLOCK_STREAK,
            "idle": not self.has_outstanding_work(),
            "zones": [z.to_dict() for z in self.zones],
            "dispatch": [d.to_dict() for d in self.last_dispatch],
            "queue": self.queue.stats() if self.queue is not None else None,
            "requests": (
                [r.to_dict() for r in self.queue.requests.values()]
                if self.queue is not None
                else []
            ),
        }
