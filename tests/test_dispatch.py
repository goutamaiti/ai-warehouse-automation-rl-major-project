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
