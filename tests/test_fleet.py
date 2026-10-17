"""Tests for the multi-robot simulation (:mod:`simulation.fleet`).

The fleet is a sibling of the single-robot engine, not a replacement, so the
first test here is the one that matters most: adding it must not have moved
single-robot behaviour at all. The rest pin the rules that only exist once
several robots share a floor - who wins a contested cell, what cannot happen
(swapping places), and what happens when the floor jams.
"""

from __future__ import annotations

import pytest

from simulation.config import ScenarioConfig, find_scenario, load_scenario
from simulation.engine import Action
from simulation.fleet import DEADLOCK_STREAK, DEFAULT_YIELD_AFTER, FleetSimulation
from simulation.warehouse import CellType


def scenario(name: str = "default") -> ScenarioConfig:
    return load_scenario(find_scenario(name))


def fleet(
    name: str = "default",
    n_robots: int = 3,
    seed: int = 1000,
    yield_after: int = DEFAULT_YIELD_AFTER,
) -> FleetSimulation:
    sim = FleetSimulation(scenario(name), n_robots=n_robots, yield_after=yield_after)
    sim.reset(seed=seed)
    return sim


# ----------------------------------------------------------------------
# The point of the whole design: the single-robot path is untouched
# ----------------------------------------------------------------------


def test_importing_the_fleet_does_not_disturb_the_single_robot_engine():
    """The fleet shares warehouse/navigation/robot/tasks but must not alter them.

    tests/test_frozen_contract.py is the thorough check; this is a fast guard
    that fails in the same file as the fleet code, so a change here is traced
    to its cause immediately rather than through a distant golden-file failure.
    """
    from environment import make_env

    env = make_env("default")
    observation, info = env.reset(seed=1000)
    _obs, reward, _term, _trunc, step_info = env.step(int(Action.UP))

    # Values from tests/golden/frozen_contract.json for default @ seed 1000.
    assert env.sim.robot.position == (12, 1)
    assert info["optimal_path_length"] == 31
    assert reward == pytest.approx(0.94, abs=1e-6)
    assert step_info["reward_components"]["progress"] == pytest.approx(1.0, abs=1e-6)
    assert observation.shape == (37,)


# ----------------------------------------------------------------------
# Placement and feasibility
# ----------------------------------------------------------------------


def test_robots_start_on_charging_stations_before_spilling_over():
    """A battery fleet must begin in the same state a single robot does."""
    sim = fleet(n_robots=2)
    stations = set(sim.layout.charging_stations)
    assert set(sim.positions.values()) == stations


def test_every_robot_starts_on_a_distinct_walkable_cell():
    sim = fleet(n_robots=8)
    positions = list(sim.positions.values())
    assert len(set(positions)) == len(positions), "two robots share a start cell"
    for position in positions:
        assert sim.layout.is_walkable(position)


def test_spillover_never_parks_a_robot_on_a_packing_station():
    """A robot parked on a drop-off would block deliveries from tick zero."""
    sim = fleet(n_robots=10)
    packing = set(sim.layout.packing_stations)
    assert not (set(sim.positions.values()) & packing)


def test_a_fleet_too_large_for_the_floor_is_rejected_at_construction():
    """No hard cap, but an unplaceable fleet fails clearly instead of obscurely."""
    with pytest.raises(ValueError, match="cannot place"):
        FleetSimulation(scenario(), n_robots=10_000)


def test_a_fleet_needs_at_least_one_robot():
    with pytest.raises(ValueError, match="at least one robot"):
        FleetSimulation(scenario(), n_robots=0)


def test_placement_is_a_pure_function_of_the_layout():
    """No RNG in placement: the same config gives the same cells every time."""
    assert fleet(n_robots=6).positions == fleet(n_robots=6).positions


# ----------------------------------------------------------------------
# Conflict resolution: lower robot id wins
# ----------------------------------------------------------------------


def _two_robots_facing_each_other(sim: FleetSimulation) -> tuple[int, int]:
    """Place robots 0 and 1 side by side in a free aisle, facing one another."""
    row = 1  # the top aisle is clear across the whole warehouse
    left, right = (row, 5), (row, 6)
    assert sim.layout.is_walkable(left) and sim.layout.is_walkable(right)
    sim.members[0].robot.position = left
    sim.members[1].robot.position = right
    return left, right


def test_two_robots_cannot_swap_places():
    """Facing robots must not pass through each other.

    Both find the other still in position when they resolve, so neither moves.
    """
    sim = fleet(n_robots=2)
    left, right = _two_robots_facing_each_other(sim)

    outcome = sim.step({0: Action.RIGHT, 1: Action.LEFT})

    assert sim.members[0].position == left
    assert sim.members[1].position == right
    assert outcome.robots[0].blocked_by_robot
    assert outcome.robots[1].blocked_by_robot


def test_the_lower_robot_id_wins_a_contested_cell():
    sim = fleet(n_robots=2)
    # Both aim at the same free cell in the top aisle from opposite sides.
    sim.members[0].robot.position = (1, 5)
    sim.members[1].robot.position = (1, 7)
    contested = (1, 6)
    assert sim.layout.is_walkable(contested)

    outcome = sim.step({0: Action.RIGHT, 1: Action.LEFT})

    assert sim.members[0].position == contested, "robot 0 should have won"
    assert sim.members[1].position == (1, 7), "robot 1 should have been refused"
    assert outcome.robots[0].moved
    assert outcome.robots[1].blocked_by_robot


def test_a_robot_may_follow_a_lower_numbered_robot_into_a_vacated_cell():
    """Convoys work in id order - robot 1 follows robot 0 the same tick."""
    sim = fleet(n_robots=2)
    sim.members[0].robot.position = (1, 5)
    sim.members[1].robot.position = (1, 4)

    outcome = sim.step({0: Action.RIGHT, 1: Action.RIGHT})

    assert sim.members[0].position == (1, 6)
    assert sim.members[1].position == (1, 5), "robot 1 should have taken the vacated cell"
    assert outcome.robots[0].moved and outcome.robots[1].moved


def test_being_blocked_by_a_peer_is_not_counted_as_a_collision():
    """Congestion and crashing into racking are different measurements."""
    sim = fleet(n_robots=2)
    _two_robots_facing_each_other(sim)
    sim.step({0: Action.RIGHT, 1: Action.LEFT})

    counters = sim.members[0].counters
    assert counters.blocked_by_robot == 1
    assert counters.collisions == 0
    assert counters.static_collisions == 0
    assert counters.dynamic_collisions == 0


def test_driving_into_a_wall_is_still_a_static_collision():
    sim = fleet(n_robots=1)
    # Robot 0 starts on the station row; the row below it is the outer wall.
    outcome = sim.step({0: Action.DOWN})
    assert outcome.robots[0].blocked_by_static
    assert sim.members[0].counters.static_collisions == 1
    assert sim.members[0].counters.blocked_by_robot == 0


def test_a_robot_with_no_action_holds_position():
    sim = fleet(n_robots=2)
    before = dict(sim.positions)
    outcome = sim.step({0: Action.WAIT})  # robot 1 omitted entirely
    assert sim.positions == before
    assert outcome.robots[1].waited


# ----------------------------------------------------------------------
# Deadlock
# ----------------------------------------------------------------------


def test_a_standoff_without_yielding_is_reported_as_a_deadlock():
    """The detector is the backstop, and yielding is off by default."""
    sim = fleet(n_robots=2)
    assert sim.yield_after == 0, "yielding is off by default; see DEFAULT_YIELD_AFTER"
    _two_robots_facing_each_other(sim)

    outcome = None
    for _ in range(DEADLOCK_STREAK):
        outcome = sim.step({0: Action.RIGHT, 1: Action.LEFT})

    assert outcome is not None and outcome.deadlocked
    assert sim.done
    with pytest.raises(RuntimeError, match="finished"):
        sim.step({0: Action.WAIT, 1: Action.WAIT})


def test_yielding_breaks_a_standoff_when_it_is_enabled():
    """The congestion fix: a robot refused repeatedly steps aside.

    Two robots facing each other in an aisle can never swap under the
    id-priority rule, so without this they retry until the deadlock detector
    fires. With it, the loser clears the contested cell and traffic resumes.
    """
    sim = fleet(n_robots=2, yield_after=3)
    left, right = _two_robots_facing_each_other(sim)

    yielded = False
    for _ in range(DEADLOCK_STREAK):
        outcome = sim.step({0: Action.RIGHT, 1: Action.LEFT})
        yielded = yielded or any(o.yielded for o in outcome.robots.values())
        assert not outcome.deadlocked, "yielding should have prevented the deadlock"

    assert yielded, "no robot ever stepped aside"
    assert sum(m.counters.yields for m in sim.members.values()) >= 1
    assert {sim.members[0].position, sim.members[1].position} != {left, right}


def test_a_robot_yields_only_after_being_refused_repeatedly():
    """Ordinary contention must resolve on its own, without a sidestep."""
    yield_after = 3
    sim = fleet(n_robots=2, yield_after=yield_after)
    _two_robots_facing_each_other(sim)

    for tick in range(yield_after - 1):
        outcome = sim.step({0: Action.RIGHT, 1: Action.LEFT})
        assert not any(o.yielded for o in outcome.robots.values()), (
            f"yielded too early, on tick {tick}"
        )

    outcome = sim.step({0: Action.RIGHT, 1: Action.LEFT})
    assert any(o.yielded for o in outcome.robots.values())


def test_a_boxed_in_robot_has_nowhere_to_yield_to():
    """With every escape taken, the sidestep must refuse rather than invent one.

    The deadlock detector is the backstop for a fleet that is wedged solid, so
    _sidestep returning None is the behaviour that hands over to it.
    """
    sim = fleet(n_robots=5, yield_after=3)
    member = sim.members[0]
    member.robot.position = (1, 5)
    refused = (1, 6)

    # Fill every other neighbour of (1, 5) with peers. Row 0 is the outer wall,
    # so only the remaining walkable neighbours need occupying.
    fillers = [
        cell
        for cell in ((0, 5), (2, 5), (1, 4))
        if sim.layout.is_walkable(cell)
    ]
    for robot_id, cell in zip([1, 2, 3, 4], fillers):
        sim.members[robot_id].robot.position = cell

    occupied = dict(sim.positions)
    assert sim._sidestep(member, refused, occupied) is None


def test_charging_counts_as_progress_so_a_charging_fleet_is_not_deadlocked():
    """A robot on a charger is working. Waiting there must not look like a jam."""
    sim = fleet("battery_tuned", n_robots=2)
    for member in sim.members.values():
        assert sim.layout.cell(member.position) is CellType.CHARGING
        member.robot.battery = 10.0

    for _ in range(DEADLOCK_STREAK + 4):
        outcome = sim.step({0: Action.WAIT, 1: Action.WAIT})
        assert not outcome.deadlocked

    assert sim.members[0].counters.charging_events > 0


def test_movement_resets_the_deadlock_streak():
    sim = fleet(n_robots=2)
    _two_robots_facing_each_other(sim)
    for _ in range(DEADLOCK_STREAK - 1):
        sim.step({0: Action.RIGHT, 1: Action.LEFT})
    assert sim.fleet_counters()["deadlock_streak"] == DEADLOCK_STREAK - 1

    # Row 0 is the outer wall, so "up" is not an escape from the top aisle -
    # the robots step apart along it instead.
    sim.step({0: Action.LEFT, 1: Action.RIGHT})
    assert sim.fleet_counters()["deadlock_streak"] == 0
    assert not sim.done


# ----------------------------------------------------------------------
# Tasks, battery and reproducibility
# ----------------------------------------------------------------------


def test_delivering_hands_the_robot_a_fresh_task():
    """A fleet keeps working; a delivery does not end the run."""
    sim = fleet(n_robots=1)
    member = sim.members[0]
    member.robot.position = member.task.pickup
    sim.step({0: Action.WAIT})
    assert member.robot.carrying

    first_task_id = member.task.task_id
    member.robot.position = member.task.dropoff
    outcome = sim.step({0: Action.WAIT})

    assert outcome.robots[0].delivered
    assert "new_task" in outcome.robots[0].events
    assert member.counters.tasks_delivered == 1
    assert member.task.task_id != first_task_id
    assert not member.robot.carrying


def test_a_flat_battery_retires_a_robot_but_leaves_it_on_the_floor():
    """A dead robot still occupies its aisle - it does not vanish."""
    sim = fleet("battery_tuned", n_robots=2)
    sim.members[0].robot.position = (1, 5)
    sim.members[0].robot.battery = 0.4  # one move empties it
    sim.step({0: Action.RIGHT, 1: Action.WAIT})

    assert not sim.members[0].alive
    dead_cell = sim.members[0].position
    assert dead_cell in sim.positions.values()

    # It is still an obstacle: robot 1 cannot enter its cell.
    sim.members[1].robot.position = (1, 7) if dead_cell == (1, 6) else (1, 5)
    towards = Action.LEFT if sim.members[1].position[1] > dead_cell[1] else Action.RIGHT
    outcome = sim.step({0: Action.UP, 1: towards})
    assert outcome.robots[1].blocked_by_robot


def test_a_retired_robot_stops_acting():
    sim = fleet("battery_tuned", n_robots=1)
    sim.members[0].robot.battery = 0.0
    before = sim.members[0].position
    outcome = sim.step({0: Action.UP})
    assert not sim.members[0].alive, "a flat battery must retire the robot"
    assert sim.members[0].position == before
    assert not outcome.robots[0].moved
    assert outcome.all_dead


def test_the_same_seed_rebuilds_the_same_run():
    """Reproducibility is what makes a fleet experiment an experiment."""
    actions = [
        {0: Action.UP, 1: Action.UP, 2: Action.LEFT},
        {0: Action.RIGHT, 1: Action.LEFT, 2: Action.UP},
        {0: Action.UP, 1: Action.UP, 2: Action.RIGHT},
    ]

    def run() -> list:
        sim = fleet("dynamic_obstacles", n_robots=3, seed=7)
        trace = []
        for step_actions in actions:
            outcome = sim.step(step_actions)
            trace.append((dict(sim.positions), sorted(outcome.events())))
        trace.append(sorted(o["position"] for o in sim.obstacles.to_list()))
        return trace

    assert run() == run()


def test_fleet_counters_aggregate_the_per_robot_tallies():
    sim = fleet(n_robots=3)
    for _ in range(5):
        sim.step({0: Action.UP, 1: Action.UP, 2: Action.UP})

    counters = sim.fleet_counters()
    assert counters["n_robots"] == 3
    assert set(counters["per_robot"]) == {0, 1, 2}
    assert counters["moves"] == sum(
        m.counters.moves for m in sim.members.values()
    )
    assert counters["robots_alive"] == 3


def test_blocked_grid_excludes_the_planning_robot_itself():
    """A controller must not treat the robot it is planning for as an obstacle."""
    sim = fleet(n_robots=3)
    grid = sim.blocked_grid(exclude=0)
    assert not grid[sim.members[0].position], "robot 0 blocked itself"
    assert grid[sim.members[1].position]
    assert grid[sim.members[2].position]


def test_nearest_charger_uses_path_distance_not_straight_line():
    sim = fleet(n_robots=1)
    station, distance = sim.nearest_charger(sim.layout.packing_stations[-1])
    assert station in sim.layout.charging_stations
    assert distance >= 0
    # The right-hand packing station must prefer the right-hand charger.
    assert station == max(sim.layout.charging_stations, key=lambda s: s[1])


def test_snapshot_is_json_serialisable():
    import json

    sim = fleet(n_robots=2)
    outcome = sim.step({0: Action.UP, 1: Action.UP})
    payload = sim.snapshot(outcome.events())
    json.dumps(payload)  # must not raise
    assert len(payload["robots"]) == 2
    assert payload["deadlocked"] is False
