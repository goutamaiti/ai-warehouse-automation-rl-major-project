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
from .engine import Action
from .navigation import DIRECTIONS, Position, bfs_distance_field
from .obstacles import ObstacleField
from .robot import Robot
from .tasks import Task, TaskGenerator, TaskStatus
from .warehouse import CellType, WarehouseLayout, build_layout

#: Consecutive steps with no fleet-wide progress before the run is declared
#: deadlocked. "Progress" is deliberately broad - any move, pickup, delivery or
#: charge counts - because a robot sitting on a charger is working, not stuck.
#: Eight is long enough that ordinary contention (two robots taking turns down
#: one aisle) never trips it, and short enough to catch a real standoff before
#: a continuous session wastes minutes on it.
DEADLOCK_STREAK = 8


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
    task: Task
    counters: RobotCounters = field(default_factory=RobotCounters)
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
            "task": self.task.to_dict(),
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
            o.moved or o.picked_up or o.delivered or o.charged > 0
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
    ) -> None:
        if n_robots < 1:
            raise ValueError(f"a fleet needs at least one robot, got {n_robots}")
        self.config = config
        self.n_robots = int(n_robots)
        self.layout = layout if layout is not None else build_layout(config.layout)
        self._walkable = self.layout.walkable_mask()

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
        field = bfs_distance_field(self._walkable, stations[0])
        reserved = set(stations) | set(self.layout.packing_stations)

        spill = [
            (int(r), int(c))
            for r, c in zip(*np.nonzero(self._walkable))
            if (int(r), int(c)) not in reserved and field[int(r), int(c)] >= 0
        ]
        spill.sort(key=lambda cell: (int(field[cell]), cell))
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
        for robot_id, start in enumerate(starts):
            robot = Robot(
                robot_id=robot_id,
                position=start,
                battery=min(
                    self.config.battery.start_level, self.config.battery.capacity
                ),
            )
            task = self._task_generator.generate(step=0)
            member = FleetMember(robot=robot, task=task)
            member.counters.optimal_path_length = self._optimal_legs(start, task)
            self.members[robot_id] = member

        forbidden = set(starts)
        forbidden.update(self.layout.charging_stations)
        forbidden.update(self.layout.packing_stations)
        for member in self.members.values():
            forbidden.update({member.task.pickup, member.task.dropoff})
        self.obstacles = ObstacleField.spawn(
            self._walkable, self.config.obstacles, self.rng, forbidden
        )

        self._target_fields = {}
        self._idle_streak = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

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
        for member in self.members.values():
            if member.alive and member.robot.is_depleted():
                member.alive = False
                member.task.mark_failed()

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
                member.task.mark_failed()

        # The rest of the world moves once, after every robot has acted, and
        # routes around every robot - living or dead.
        self.obstacles.step(
            self._walkable, self.rng, blocked=set(self.positions.values())
        )

        self._idle_streak = 0 if outcome.progressed else self._idle_streak + 1
        outcome.deadlocked = self._idle_streak >= DEADLOCK_STREAK
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
                robot_outcome.events.append("blocked_by_robot")
                member.counters.blocked_by_robot += 1
                robot_outcome.energy_used = battery_cfg.idle_cost
            else:
                moves[robot_id] = target
                occupied[robot_id] = target
                robot_outcome.moved = True
                robot_outcome.energy_used = battery_cfg.move_cost
                member.counters.moves += 1

        return moves

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
            # A fleet keeps working: the robot picks up a fresh job rather than
            # the run ending. A shared queue replaces this in the next phase.
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
        return int(self._field(member.task.target)[member.position])

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
            "idle_steps": sum(c.idle_steps for c in per_robot.values()),
            "energy_consumed": round(
                sum(c.energy_consumed for c in per_robot.values()), 3
            ),
            "charging_events": sum(c.charging_events for c in per_robot.values()),
            "tasks_delivered": sum(c.tasks_delivered for c in per_robot.values()),
            "robots_alive": sum(1 for m in self.members.values() if m.alive),
            "deadlock_streak": self._idle_streak,
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
            "deadlocked": self._idle_streak >= DEADLOCK_STREAK,
        }
