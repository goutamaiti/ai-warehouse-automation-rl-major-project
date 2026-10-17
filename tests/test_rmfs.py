"""Tests for the robotic-fulfilment floor (:mod:`simulation.rmfs`).

The first test is the one that matters most: adding a second, more realistic
floor must not have moved the one every committed result was measured on.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import find_scenario, load_scenario
from simulation.dispatch import TaskQueue
from simulation.fleet import FleetSimulation
from simulation.rmfs import (
    DOWN,
    FREE,
    LEFT,
    RIGHT,
    UP,
    RmfsConfig,
    allows,
    astar_directed,
    build_rmfs_layout,
    direction_summary,
)
from simulation.warehouse import CellType, build_layout


def floor():
    return build_rmfs_layout()


# ----------------------------------------------------------------------
# The measured floor is untouched
# ----------------------------------------------------------------------


def test_the_original_layout_generator_is_unchanged():
    """build_layout produces every committed result; this module must not touch it.

    tests/test_frozen_contract.py is the thorough check. This one fails in the
    same file as the new floor, so a regression is traced to its cause.
    """
    config = load_scenario(find_scenario("default")).layout
    layout = build_layout(config)
    assert (layout.height, layout.width) == (15, 21)
    assert layout.charging_stations == ((13, 1), (13, 19))
    assert layout.packing_stations == ((13, 4), (13, 16))
    assert len(layout.storage_points) == 24


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------


def test_the_floor_is_a_valid_warehouse_layout():
    """It must satisfy the same validation every other floor does.

    build_rmfs_layout calls validate_layout internally, so reaching this point
    already proves every point of interest is walkable and mutually reachable.
    """
    layout, direction = floor()
    assert layout.shape == direction.shape
    assert layout.storage_points and layout.packing_stations and layout.charging_stations
    for point in (
        layout.storage_points + layout.packing_stations + layout.charging_stations
    ):
        assert layout.is_walkable(point)


def test_the_floor_is_substantially_larger_than_the_original():
    """The point of it: 175 free cells is where the fleet saturated."""
    layout, _ = floor()
    original = build_layout(load_scenario(find_scenario("default")).layout)
    assert int(layout.walkable_mask().sum()) > 4 * int(original.walkable_mask().sum())


def test_it_uses_the_same_cell_types_as_every_other_floor():
    """Pods are SHELF, stations PACKING, bays CHARGING.

    This is what lets the engine, fleet, dispatch, zones and planners run on it
    with no changes: it is an ordinary WarehouseLayout.
    """
    layout, _ = floor()
    used = set(np.unique(layout.grid).tolist())
    assert used <= {int(member) for member in CellType}
    assert int(CellType.SHELF) in used, "a pod field with no pods"
    assert (layout.grid == int(CellType.SHELF)).sum() > 100


def test_storage_points_all_face_a_pod():
    """A pick point the robot cannot reach a pod from is not a pick point."""
    layout, _ = floor()
    for row, col in layout.storage_points:
        neighbours = [
            layout.grid[row + dr, col + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= row + dr < layout.height and 0 <= col + dc < layout.width
        ]
        assert int(CellType.SHELF) in neighbours, f"{(row, col)} faces no pod"


def test_a_floor_too_small_for_a_pod_field_is_refused():
    with pytest.raises(ValueError, match="at least"):
        build_rmfs_layout(RmfsConfig(width=10, height=8))


def test_the_layout_is_a_pure_function_of_its_config():
    first, first_dir = floor()
    second, second_dir = floor()
    assert np.array_equal(first.grid, second.grid)
    assert np.array_equal(first_dir, second_dir)


# ----------------------------------------------------------------------
# One-way aisles
# ----------------------------------------------------------------------


def test_the_floor_has_one_way_lanes_in_both_axes():
    _, direction = floor()
    summary = direction_summary(direction)
    assert summary["right"] > 0 and summary["left"] > 0
    assert summary["down"] > 0 and summary["up"] > 0


def test_parallel_lanes_run_in_opposite_directions():
    """Alternating direction is the point - two adjacent lanes agreeing would
    leave the field with no return path."""
    _, direction = floor()
    rights = {r for r in range(direction.shape[0]) if (direction[r] == RIGHT).any()}
    lefts = {r for r in range(direction.shape[0]) if (direction[r] == LEFT).any()}
    assert rights and lefts and not (rights & lefts)


def test_junctions_are_free_so_a_robot_can_turn():
    """Regression for the mistake that made one-way unusable.

    With junctions locked to one axis a robot cannot turn at an intersection,
    so every route becomes an enormous detour via a cross-aisle. Measured with
    all 96 junctions locked: a twelve-robot fleet delivered 2 packages against
    19 for the two-way floor. Freeing them is what makes one-way viable.
    """
    layout, direction = floor()
    horizontal = {
        r
        for r in range(direction.shape[0])
        if any(direction[r, c] in (RIGHT, LEFT) for c in range(direction.shape[1]))
    }
    vertical = {
        c
        for c in range(direction.shape[1])
        if any(direction[r, c] in (DOWN, UP) for r in range(direction.shape[0]))
    }
    walkable = layout.walkable_mask()
    junctions = [(r, c) for r in horizontal for c in vertical if walkable[r, c]]
    assert junctions, "the test is meaningless with no junctions"
    locked = [cell for cell in junctions if int(direction[cell]) != FREE]
    assert not locked, f"{len(locked)} junctions cannot be turned at"


def test_allows_enforces_the_rule_of_the_cell_being_left():
    _, direction = floor()
    cell = next(
        (r, c)
        for r in range(direction.shape[0])
        for c in range(direction.shape[1])
        if direction[r, c] == RIGHT
    )
    assert allows(direction, cell, (cell[0], cell[1] + 1))
    assert not allows(direction, cell, (cell[0], cell[1] - 1))
    assert not allows(direction, cell, (cell[0] + 1, cell[1]))


def test_a_free_cell_permits_any_move():
    _, direction = floor()
    cell = next(
        (r, c)
        for r in range(direction.shape[0])
        for c in range(direction.shape[1])
        if direction[r, c] == FREE
    )
    for delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        assert allows(direction, cell, (cell[0] + delta[0], cell[1] + delta[1]))


# ----------------------------------------------------------------------
# The directed planner
# ----------------------------------------------------------------------


def test_every_step_of_a_directed_path_is_legal():
    """The whole reason the rule lives inside the search.

    A path from an undirected search is illegal from its second step; the
    controller then replans, proposes the same route and the robot thrashes.
    """
    layout, direction = floor()
    blocked = ~layout.walkable_mask()
    path = astar_directed(
        blocked, direction, layout.storage_points[len(layout.storage_points) // 2],
        layout.packing_stations[0],
    )
    assert len(path) >= 2
    for origin, destination in zip(path, path[1:]):
        assert allows(direction, origin, destination), f"{origin} -> {destination}"


def test_a_directed_path_exists_from_every_sampled_pick_point_to_a_station():
    """A pick point with no legal route to a station is unservable."""
    layout, direction = floor()
    blocked = ~layout.walkable_mask()
    sample = layout.storage_points[:: max(1, len(layout.storage_points) // 25)]
    for point in sample:
        assert astar_directed(blocked, direction, point, layout.packing_stations[0]), (
            f"no legal route from {point}"
        )


def test_the_directed_planner_returns_nothing_for_an_unreachable_goal():
    layout, direction = floor()
    blocked = ~layout.walkable_mask()
    wall = (0, 0)
    assert astar_directed(blocked, direction, layout.packing_stations[0], wall) == ()


def test_a_directed_path_to_the_start_is_the_start():
    layout, direction = floor()
    start = layout.packing_stations[0]
    assert astar_directed(~layout.walkable_mask(), direction, start, start) == (start,)


# ----------------------------------------------------------------------
# A fleet actually runs on it
# ----------------------------------------------------------------------


def test_a_fleet_places_and_runs_on_the_realistic_floor():
    from baselines.fleet_controller import FleetPlannerController

    layout, direction = floor()
    config = load_scenario(find_scenario("default"))
    queue = TaskQueue()
    sim = FleetSimulation(config, n_robots=8, layout=layout, task_queue=queue)
    sim.reset(seed=1000)
    controller = FleetPlannerController(directions=direction)
    controller.reset(sim)

    for pickup in layout.storage_points[:6]:
        queue.add(pickup, layout.packing_stations[0], step=0)

    for _ in range(120):
        if sim.done:
            break
        sim.step(controller.act(sim))

    assert not sim.done, "the fleet should still be running"
    counters = sim.fleet_counters()
    assert counters["robots_alive"] == 8
    assert counters["moves"] > 0, "nobody moved"


def test_one_way_routing_all_but_eliminates_robot_on_robot_blocking():
    """The measured benefit of the one-way field.

    Twelve robots and a real backlog, because the effect is a congestion
    effect: at eight robots over two hundred ticks both floors record single
    digits and the comparison says nothing. At this load the two-way floor
    records thousands of refusals and the one-way field tens, because two
    robots cannot face each other in a lane at all.

    The assertion is deliberately loose - a fivefold gap against a measured
    hundredfold - so it tests the mechanism rather than a specific seed.
    """
    from baselines.fleet_controller import FleetPlannerController

    layout, direction = floor()
    config = load_scenario(find_scenario("default"))

    def blocked_events(directions):
        queue = TaskQueue()
        sim = FleetSimulation(config, n_robots=12, layout=layout, task_queue=queue)
        sim.reset(seed=1000)
        controller = FleetPlannerController(directions=directions)
        controller.reset(sim)
        rng = np.random.default_rng(1000)
        for _ in range(200):
            queue.add(
                layout.storage_points[int(rng.integers(len(layout.storage_points)))],
                layout.packing_stations[
                    int(rng.integers(len(layout.packing_stations)))
                ],
                step=0,
            )
        for _ in range(400):
            if sim.done:
                break
            sim.step(controller.act(sim))
        return sim.fleet_counters()["blocked_by_robot"]

    one_way = blocked_events(direction)
    two_way = blocked_events(None)
    assert two_way > 200, f"the load did not produce congestion to compare ({two_way})"
    assert one_way * 5 < two_way, f"one-way {one_way} vs two-way {two_way}"


def test_a_full_robot_does_not_park_at_a_charger_it_no_longer_needs():
    """Regression for a park that stalled deliveries indefinitely.

    On a large floor the reserve can exceed the battery outright - a 60-cell
    trip plus a 45-cell return needs 105 against a capacity of 100 - so a robot
    that diverts whenever the reserve is unmet sits at a charger at full charge
    for ever, since topping up can never clear the condition. Measured before
    the fix: a package picked up at tick 30 was still undelivered at tick 340.
    """
    from baselines.fleet_controller import FleetPlannerController

    layout, direction = floor()
    config = load_scenario(find_scenario("default"))
    queue = TaskQueue()
    sim = FleetSimulation(config, n_robots=12, layout=layout, task_queue=queue)
    sim.reset(seed=1000)
    controller = FleetPlannerController(directions=direction)
    controller.reset(sim)

    # A trip that crosses the floor, which is what makes the reserve
    # unsatisfiable: a far corner pick point to the far-side station.
    queue.add((2, 41), layout.packing_stations[0], step=0)

    delivered_at = None
    for tick in range(400):
        if sim.done:
            break
        sim.step(controller.act(sim))
        if queue.stats()["delivered"]:
            delivered_at = tick + 1
            break

    assert delivered_at is not None, "the robot never delivered; it is parked again"
    assert delivered_at < 400
