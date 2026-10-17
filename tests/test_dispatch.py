"""Tests for zones and nearest-robot dispatch.

The queue exists to fix a measured failure, not a hypothetical one: with each
robot generating its own task, two robots could be sent to the same storage
point, meet in an aisle with no detour, and jam - five robots on ``default``
deadlocked for exactly that reason. These tests pin the rules that stop it, and
the properties the dashboard's product-location card depends on.
"""

from __future__ import annotations

import pytest

from simulation.config import ScenarioConfig, find_scenario, load_scenario
from simulation.dispatch import RequestStatus, TaskQueue
from simulation.engine import Action
from simulation.fleet import FleetSimulation
from simulation.warehouse import CellType
from simulation.zones import build_zones, zone_of


def scenario(name: str = "default") -> ScenarioConfig:
    return load_scenario(find_scenario(name))


def queued_fleet(
    name: str = "default", n_robots: int = 3, seed: int = 1000
) -> tuple[FleetSimulation, TaskQueue]:
    queue = TaskQueue()
    sim = FleetSimulation(scenario(name), n_robots=n_robots, task_queue=queue)
    sim.reset(seed=seed)
    return sim, queue


# ----------------------------------------------------------------------
# Zones
# ----------------------------------------------------------------------


def test_zones_partition_every_walkable_cell_exactly_once():
    layout = FleetSimulation(scenario(), n_robots=1).layout
    zones = build_zones(layout)
    walkable = {
        (int(r), int(c)) for r, c in zip(*layout.walkable_mask().nonzero())
    }

    covered: set = set()
    for zone in zones:
        assert not (covered & zone.cells), "zones overlap"
        covered |= zone.cells
    assert covered == walkable, "zones do not cover the whole floor"


def test_there_is_one_zone_per_charging_station():
    layout = FleetSimulation(scenario(), n_robots=1).layout
    zones = build_zones(layout)
    assert len(zones) == len(layout.charging_stations)
    assert {z.charger for z in zones} == set(layout.charging_stations)
    assert [z.name for z in zones] == ["A", "B"]


def test_a_cell_belongs_to_the_zone_of_its_nearest_charger_by_path():
    """Geodesic, not straight-line: a cell behind racking is not nearby."""
    sim = FleetSimulation(scenario(), n_robots=1)
    zones = build_zones(sim.layout)
    for zone in zones:
        own = sim.path_distance(zone.charger, zone.charger)
        assert own == 0
        for other in zones:
            if other is zone:
                continue
            # Every cell in a zone is at least as close to its own charger.
            for cell in list(zone.cells)[:40]:
                mine = sim.path_distance(cell, zone.charger)
                theirs = sim.path_distance(cell, other.charger)
                assert mine >= 0 and mine <= theirs


def test_waiting_points_are_never_stations_or_pickup_cells():
    """A waiting robot must block neither power, nor a drop-off, nor a pickup."""
    layout = FleetSimulation(scenario(), n_robots=1).layout
    reserved = (
        set(layout.charging_stations)
        | set(layout.packing_stations)
        | set(layout.storage_points)
    )
    for zone in build_zones(layout):
        assert zone.waiting_points, f"zone {zone.name} has nowhere to wait"
        for point in zone.waiting_points:
            assert point not in reserved
            assert layout.is_walkable(point)


def test_zones_are_a_pure_function_of_the_layout():
    layout = FleetSimulation(scenario(), n_robots=1).layout
    assert build_zones(layout) == build_zones(layout)


def test_zone_of_returns_none_off_the_floor():
    layout = FleetSimulation(scenario(), n_robots=1).layout
    assert zone_of(build_zones(layout), (0, 0)) is None


def test_robots_homed_in_one_zone_get_distinct_waiting_points():
    sim, _ = queued_fleet(n_robots=6)
    points = [
        m.waiting_point for m in sim.members.values() if m.waiting_point is not None
    ]
    assert len(set(points)) == len(points), "two robots would park on one cell"


# ----------------------------------------------------------------------
# The queue and the assignment rule
# ----------------------------------------------------------------------


def test_adding_a_request_assigns_it_on_the_very_next_tick():
    """"The instant I add the location, the robot should start moving."""
    sim, queue = queued_fleet(n_robots=3)
    assert all(m.task is None for m in sim.members.values())

    request = queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    assert request.waiting

    sim.step({})

    assert request.status is RequestStatus.ASSIGNED
    assert request.robot_id is not None
    assert sim.members[request.robot_id].task is not None
    assert len(sim.last_dispatch) == 1


def test_the_nearest_idle_robot_wins_by_path_distance():
    sim, queue = queued_fleet(n_robots=3)
    pickup = sim.layout.storage_points[0]
    queue.add(pickup, sim.layout.packing_stations[0])
    sim.step({})

    winner = queue.requests[0].robot_id
    distances = {
        rid: sim.path_distance(m.position, pickup)
        for rid, m in sim.members.items()
    }
    # The winner's start position must have been a minimum-distance one.
    best = min(d for d in distances.values() if d >= 0)
    assert distances[winner] == best, distances


def test_two_robots_are_never_sent_to_the_same_request():
    """The exact cause of the five-robot deadlock."""
    sim, queue = queued_fleet(n_robots=5)
    queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    sim.step({})

    serving = [rid for rid, m in sim.members.items() if m.task is not None]
    assert len(serving) == 1


def test_each_robot_holds_at_most_one_request():
    sim, queue = queued_fleet(n_robots=2)
    for _ in range(6):
        queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    sim.step({})

    assigned = [r for r in queue.requests.values() if r.robot_id is not None]
    assert len(assigned) == 2, "only the two idle robots should have been given work"
    assert len({r.robot_id for r in assigned}) == 2


def test_a_carrying_robot_is_never_reassigned():
    """Dropping a package to chase a closer job is worse service, not better."""
    sim, queue = queued_fleet(n_robots=1)
    queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    sim.step({})
    member = sim.members[0]
    first = member.task.task_id

    member.robot.position = member.task.pickup
    sim.step({0: Action.WAIT})
    assert member.robot.carrying

    queue.add(sim.layout.storage_points[1], sim.layout.packing_stations[0])
    sim.step({0: Action.WAIT})
    assert member.task.task_id == first, "robot was re-assigned mid-delivery"


def test_requests_are_served_oldest_first():
    sim, queue = queued_fleet(n_robots=1)
    first = queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0], step=0)
    queue.add(sim.layout.storage_points[1], sim.layout.packing_stations[0], step=1)
    sim.step({})
    assert queue.requests[first.request_id].status is RequestStatus.ASSIGNED


def test_delivering_completes_the_request_and_frees_the_robot():
    sim, queue = queued_fleet(n_robots=1)
    request = queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    sim.step({})
    member = sim.members[0]

    member.robot.position = member.task.pickup
    sim.step({0: Action.WAIT})
    member.robot.position = member.task.dropoff
    sim.step({0: Action.WAIT})

    assert request.status is RequestStatus.DELIVERED
    assert request.delivered_step is not None
    assert request.to_dict()["turnaround"] is not None
    assert member.task is None, "the robot should be idle and dispatchable again"
    assert member.counters.tasks_delivered == 1


def test_a_dead_robots_request_goes_back_to_the_queue():
    """The product still needs moving; it must not die with the robot."""
    sim, queue = queued_fleet("battery_tuned", n_robots=2)
    request = queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    sim.step({})
    assert request.status is RequestStatus.ASSIGNED

    victim = request.robot_id
    sim.members[victim].robot.battery = 0.0
    sim.step({})

    assert not sim.members[victim].alive
    assert request.robot_id != victim
    assert request.status in (RequestStatus.WAITING, RequestStatus.ASSIGNED)


def test_an_unreachable_request_stays_queued_rather_than_failing():
    """A boxed-in robot may be free next tick, so nothing is thrown away."""
    queue = TaskQueue()
    request = queue.add((1, 1), (2, 2))
    decisions = queue.dispatch({0: (5, 5)}, lambda a, b: -1, step=0)
    assert decisions == []
    assert request.status is RequestStatus.WAITING


def test_dispatch_ties_break_on_the_lower_robot_id():
    """Determinism: a run must replay exactly."""
    queue = TaskQueue()
    queue.add((1, 1), (2, 2))
    decisions = queue.dispatch({3: (0, 0), 1: (0, 0), 2: (0, 0)}, lambda a, b: 5, step=0)
    assert [d.robot_id for d in decisions] == [1]


def test_queue_stats_report_turnaround_only_once_something_is_delivered():
    queue = TaskQueue()
    assert queue.stats()["mean_turnaround"] is None
    queue.add((1, 1), (2, 2), step=0)
    queue.mark_delivered(0, step=10)
    assert queue.stats()["mean_turnaround"] == 10.0
    assert queue.stats()["delivered"] == 1


# ----------------------------------------------------------------------
# Idle behaviour
# ----------------------------------------------------------------------


def test_an_idle_robot_heads_for_its_zone_waiting_point():
    sim, _ = queued_fleet(n_robots=3)
    for robot_id, member in sim.members.items():
        assert member.task is None
        assert sim.target_for(robot_id) == member.waiting_point


def test_a_working_robot_heads_for_its_task():
    sim, queue = queued_fleet(n_robots=1)
    queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    sim.step({})
    assert sim.target_for(0) == sim.members[0].task.target


def test_independent_mode_still_works_without_a_queue():
    """Phase 1 behaviour is preserved: no queue means self-generated tasks."""
    sim = FleetSimulation(scenario(), n_robots=2)
    sim.reset(seed=1000)
    assert sim.queue is None
    assert all(m.task is not None for m in sim.members.values())
    assert sim.snapshot()["queue"] is None


def test_snapshot_exposes_zones_requests_and_dispatch():
    import json

    sim, queue = queued_fleet(n_robots=2)
    queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    outcome = sim.step({})
    payload = sim.snapshot(outcome.events())

    json.dumps(payload)
    assert [z["name"] for z in payload["zones"]] == ["A", "B"]
    assert payload["queue"]["total"] == 1
    assert len(payload["requests"]) == 1
    assert len(payload["dispatch"]) == 1
    assert payload["robots"][0]["home_zone"] in {"A", "B"}


# ----------------------------------------------------------------------
# An idle fleet is not a jammed fleet
# ----------------------------------------------------------------------


def test_an_idle_fleet_with_an_empty_queue_is_not_deadlocked():
    """Regression: the deadlock detector used to fire on a caught-up warehouse.

    Robots parked at their waiting points with nothing queued make no moves, no
    pickups, no deliveries and no charges - the same signature as a standoff.
    Conflating the two stopped a continuous warehouse the moment it caught up
    with its work, which is the normal state of a warehouse between requests.
    """
    from simulation.fleet import DEADLOCK_STREAK

    sim, queue = queued_fleet(n_robots=3)
    assert not queue.requests
    assert not sim.has_outstanding_work()

    for _ in range(DEADLOCK_STREAK * 2):
        outcome = sim.step({})
        assert not outcome.deadlocked, "an idle fleet was reported as deadlocked"
    assert not sim.done
    assert sim.snapshot()["idle"] is True


def test_a_fleet_with_queued_work_it_cannot_reach_is_still_deadlocked():
    """The detector must keep working once there IS something to do."""
    from simulation.fleet import DEADLOCK_STREAK

    sim, queue = queued_fleet(n_robots=1)
    queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    sim.step({})
    assert sim.has_outstanding_work()

    # Off the charging station first - a robot topping up is making progress,
    # and would legitimately reset the streak every tick.
    sim.members[0].robot.position = (1, 5)
    assert sim.layout.cell(sim.members[0].position) is not CellType.CHARGING

    outcome = None
    for _ in range(DEADLOCK_STREAK):
        # Drive into the outer wall every tick: work outstanding, no progress.
        outcome = sim.step({0: Action.UP})
    assert outcome is not None and outcome.deadlocked


def test_outstanding_work_follows_the_queue_and_the_robots():
    sim, queue = queued_fleet(n_robots=1)
    assert not sim.has_outstanding_work()

    queue.add(sim.layout.storage_points[0], sim.layout.packing_stations[0])
    assert sim.has_outstanding_work(), "a waiting request is outstanding work"

    sim.step({})
    assert sim.has_outstanding_work(), "an assigned task is outstanding work"

    member = sim.members[0]
    member.robot.position = member.task.pickup
    sim.step({0: Action.WAIT})
    member.robot.position = member.task.dropoff
    sim.step({0: Action.WAIT})
    assert not sim.has_outstanding_work(), "everything delivered, nothing queued"


# ----------------------------------------------------------------------
# Fleet battery management
# ----------------------------------------------------------------------


def test_no_waiting_point_blocks_access_to_a_charger():
    """Regression: idle robots used to park on the only routes into a charger.

    Measured before the fix: every approach cell to both chargers on `default`
    was handed out as a waiting point, and as the FIRST choice, so the first two
    idle robots in each zone parked squarely on the way in. Robots then died one
    to three cells from a charger that stood free for 203 of 600 ticks, and two
    corpses on the two approach cells sealed it permanently.
    """
    from simulation.navigation import DIRECTIONS

    sim, _ = queued_fleet(n_robots=1)
    approaches = {
        (station[0] + d_row, station[1] + d_col)
        for station in sim.layout.charging_stations
        for d_row, d_col in DIRECTIONS
        if sim.layout.is_walkable((station[0] + d_row, station[1] + d_col))
    }
    assert approaches, "the test is meaningless if no approach cells exist"
    for zone in sim.zones:
        blocking = [p for p in zone.waiting_points if p in approaches]
        assert not blocking, f"zone {zone.name} would park robots on {blocking}"


def test_every_charger_keeps_at_least_one_free_approach():
    """A charger nothing can reach is the same as no charger."""
    from simulation.navigation import DIRECTIONS

    sim, _ = queued_fleet(n_robots=8)
    parked = {m.waiting_point for m in sim.members.values()}
    for station in sim.layout.charging_stations:
        approaches = [
            (station[0] + d_row, station[1] + d_col)
            for d_row, d_col in DIRECTIONS
            if sim.layout.is_walkable((station[0] + d_row, station[1] + d_col))
        ]
        assert any(a not in parked for a in approaches), (
            f"charger {station} would be sealed in by parked robots"
        )


def test_the_fleet_reserves_more_charge_than_a_single_robot_does():
    """A fleet robot is refused by peers, so it must budget for the detours.

    The single-robot planner reserves 1.2x the charge it needs; copying that
    into a fleet killed robots one to four cells short of a charger, because it
    never budgeted for the ticks spent blocked and the cells spent detouring.
    """
    from baselines.controller import PlannerPolicy
    from baselines.fleet_controller import DEFAULT_SAFETY_FACTOR, FleetPlannerController

    assert DEFAULT_SAFETY_FACTOR > PlannerPolicy().safety_factor
    assert FleetPlannerController().safety_factor == DEFAULT_SAFETY_FACTOR


def test_a_safety_factor_below_one_is_refused():
    """Below 1.0 the reserve cannot even cover the trip to the charger."""
    from baselines.fleet_controller import FleetPlannerController

    with pytest.raises(ValueError, match="cannot even reach"):
        FleetPlannerController(safety_factor=0.5)


def test_an_idle_robot_tops_up_before_parking():
    """Idle time is the only time a charging detour costs no delivery."""
    from baselines.fleet_controller import FleetPlannerController

    sim, _ = queued_fleet(n_robots=2)
    controller = FleetPlannerController()
    member = sim.members[0]
    assert member.task is None

    member.robot.battery = sim.config.battery.capacity * 0.5
    charger, _ = sim.nearest_charger(member.position)
    assert controller._goal_for(sim, 0) == charger

    member.robot.battery = sim.config.battery.capacity
    assert controller._goal_for(sim, 0) == member.waiting_point
