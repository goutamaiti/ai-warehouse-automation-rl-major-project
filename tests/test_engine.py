"""Tests for the simulation dynamics: movement, collisions, tasks, battery."""

from __future__ import annotations

import pytest

from simulation.config import (
    BatteryConfig,
    LayoutConfig,
    ObstacleConfig,
    ScenarioConfig,
    TaskConfig,
)
from simulation.engine import Action, WarehouseSimulation
from simulation.tasks import TaskStatus
from simulation.warehouse import CellType


def make_sim(**overrides) -> WarehouseSimulation:
    config = ScenarioConfig(**overrides)
    return WarehouseSimulation(config)


def drive_to(sim: WarehouseSimulation, target) -> None:
    """Move the robot to a cell using the engine's own distance field logic."""
    from baselines.astar import astar

    blocked = ~sim.walkable
    plan = astar(blocked, sim.robot.position, target)
    assert plan.found, f"no path from {sim.robot.position} to {target}"
    for nxt in plan.path[1:]:
        delta = (nxt[0] - sim.robot.position[0], nxt[1] - sim.robot.position[1])
        action = {(-1, 0): Action.UP, (1, 0): Action.DOWN, (0, -1): Action.LEFT, (0, 1): Action.RIGHT}[delta]
        sim.step(action)


def test_move_into_free_cell_updates_position_and_battery():
    sim = make_sim()
    start = sim.robot.position
    battery_before = sim.robot.battery
    outcome = sim.step(Action.UP)
    assert outcome.moved
    assert sim.robot.position == (start[0] - 1, start[1])
    assert sim.robot.battery == pytest.approx(battery_before - sim.config.battery.move_cost)
    assert sim.counters.moves == 1


def test_move_into_wall_is_a_collision_and_does_not_move_the_robot():
    sim = make_sim()
    start = sim.robot.position
    outcome = sim.step(Action.DOWN)  # the robot starts on the bottom station row
    assert outcome.blocked_by_static
    assert not outcome.moved
    assert sim.robot.position == start
    assert sim.counters.static_collisions == 1


def test_move_into_a_shelf_is_a_collision_and_does_not_move_the_robot():
    """Shelves are a static obstacle exactly like walls: is_walkable is False,
    so the engine blocks the move for every controller, not just the ones
    smart enough to plan around it."""
    sim = make_sim(max_steps=2000)
    drive_to(sim, sim.layout.storage_points[0])  # every storage point touches a shelf
    row, col = sim.robot.position
    directions = {
        Action.UP: (row - 1, col),
        Action.DOWN: (row + 1, col),
        Action.LEFT: (row, col - 1),
        Action.RIGHT: (row, col + 1),
    }
    shelf_action, shelf_cell = next(
        (a, cell) for a, cell in directions.items() if sim.layout.cell(cell) is CellType.SHELF
    )

    outcome = sim.step(shelf_action)
    assert outcome.blocked_by_static
    assert outcome.collided
    assert not outcome.moved
    assert sim.robot.position == (row, col)
    assert sim.robot.position != shelf_cell
    assert not sim.layout.is_walkable(shelf_cell)


def test_wait_costs_idle_energy_only():
    sim = make_sim()
    battery_before = sim.robot.battery
    outcome = sim.step(Action.WAIT)
    assert outcome.waited
    assert sim.counters.idle_steps == 1
    # The robot starts on a charging station, so the idle cost is refunded by
    # charging; what matters is that it never drops by a full move cost.
    assert sim.robot.battery >= battery_before - sim.config.battery.idle_cost


def test_pickup_and_delivery_complete_the_task():
    sim = make_sim(max_steps=2000)
    drive_to(sim, sim.task.pickup)
    assert sim.task.status is TaskStatus.PICKED_UP
    assert sim.robot.carrying
    drive_to(sim, sim.task.dropoff)
    assert sim.task.status is TaskStatus.DELIVERED
    assert not sim.robot.carrying
    assert sim.counters.tasks_delivered == 1
    assert sim.done
    assert sim.counters.termination_reason == "all_tasks_delivered"


def test_second_task_is_generated_when_more_are_requested():
    sim = make_sim(max_steps=4000, tasks=TaskConfig(tasks_per_episode=2))
    first_task_id = sim.task.task_id
    drive_to(sim, sim.task.pickup)
    drive_to(sim, sim.task.dropoff)
    assert not sim.done
    assert sim.task.task_id != first_task_id
    assert sim.counters.tasks_delivered == 1


def test_charging_refills_the_battery_and_is_capped():
    battery = BatteryConfig(start_level=50.0, charge_rate=8.0, capacity=100.0)
    sim = make_sim(battery=battery)
    assert sim.layout.cell(sim.robot.position) is CellType.CHARGING
    sim.step(Action.WAIT)
    assert sim.robot.battery == pytest.approx(50.0 - battery.idle_cost + battery.charge_rate)
    for _ in range(50):
        if sim.done:
            break
        sim.step(Action.WAIT)
    assert sim.robot.battery <= battery.capacity


def test_battery_depletion_ends_the_episode_as_a_failure():
    sim = make_sim(
        battery=BatteryConfig(start_level=2.0, move_cost=1.0, charge_rate=0.0),
        max_steps=500,
    )
    outcome = None
    for _ in range(10):
        outcome = sim.step(Action.UP)
        if sim.done:
            break
    assert outcome is not None and outcome.battery_depleted
    assert sim.counters.termination_reason == "battery_depleted"
    assert sim.task.status is TaskStatus.FAILED


def test_timeout_truncates_the_episode():
    sim = make_sim(max_steps=5)
    for _ in range(5):
        outcome = sim.step(Action.WAIT)
    assert outcome.truncated
    assert not outcome.terminated
    assert sim.counters.termination_reason == "timeout"


def test_stepping_a_finished_episode_raises():
    sim = make_sim(max_steps=1)
    sim.step(Action.WAIT)
    with pytest.raises(RuntimeError):
        sim.step(Action.WAIT)


def test_dynamic_obstacles_never_step_onto_the_robot():
    sim = make_sim(
        obstacles=ObstacleConfig(n_dynamic=10, move_probability=1.0),
        max_steps=200,
    )
    for _ in range(100):
        if sim.done:
            break
        sim.step(Action.WAIT)
        assert sim.robot.position not in sim.obstacles.positions


def test_driving_into_an_obstacle_is_recorded_as_a_dynamic_collision():
    sim = make_sim(obstacles=ObstacleConfig(n_dynamic=0), max_steps=200)
    # Place one obstacle by hand directly above the robot for a deterministic test.
    from simulation.obstacles import DynamicObstacle

    row, col = sim.robot.position
    sim.obstacles.obstacles.append(
        DynamicObstacle(obstacle_id=0, position=(row - 1, col), behaviour="random_walk")
    )
    outcome = sim.step(Action.UP)
    assert outcome.blocked_by_obstacle
    assert sim.counters.dynamic_collisions == 1
    assert sim.robot.position == (row, col)


def test_same_seed_produces_identical_episodes():
    def trace(seed: int):
        sim = make_sim(obstacles=ObstacleConfig(n_dynamic=6), max_steps=60)
        sim.reset(seed=seed)
        states = []
        for action in [Action.UP, Action.RIGHT, Action.UP, Action.LEFT, Action.WAIT] * 8:
            if sim.done:
                break
            sim.step(action)
            states.append((sim.robot.position, sorted(sim.obstacles.positions)))
        return states

    assert trace(42) == trace(42)
    assert trace(42) != trace(43)


def test_optimal_path_length_is_a_lower_bound_on_the_driven_path():
    sim = make_sim(max_steps=2000)
    optimal = sim.counters.optimal_path_length
    drive_to(sim, sim.task.pickup)
    drive_to(sim, sim.task.dropoff)
    assert sim.counters.moves >= optimal
    assert sim.counters.moves == optimal  # A* drives an optimal route here


def test_smaller_layout_still_simulates():
    sim = make_sim(layout=LayoutConfig(width=15, height=11, n_packing_stations=1), max_steps=100)
    sim.step(Action.UP)
    assert sim.counters.steps == 1


def test_target_stays_at_the_dropoff_after_delivery():
    sim = make_sim(max_steps=2000)
    drive_to(sim, sim.task.pickup)
    dropoff = sim.task.dropoff
    drive_to(sim, dropoff)
    assert sim.task.status is TaskStatus.DELIVERED
    assert sim.task.target == dropoff
    assert sim.distance_to_target() == 0


def test_target_stays_at_the_pickup_when_a_task_fails_before_pickup():
    sim = make_sim(max_steps=3)
    for _ in range(3):
        sim.step(Action.WAIT)
    assert sim.task.status is TaskStatus.FAILED
    assert sim.task.target == sim.task.pickup
