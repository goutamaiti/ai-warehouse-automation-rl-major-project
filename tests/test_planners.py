"""Tests for the classical planners and the controllers built on them."""

from __future__ import annotations

import numpy as np
import pytest

from baselines.astar import astar
from baselines.bfs import bfs
from baselines.controller import PlannerPolicy, RandomPolicy
from environment import make_env
from simulation.config import BatteryConfig, ObstacleConfig, ScenarioConfig, TaskConfig
from simulation.navigation import bfs_distance_field

PLANNERS = (astar, bfs)


def open_grid(size: int = 7) -> np.ndarray:
    return np.zeros((size, size), dtype=bool)


@pytest.mark.parametrize("planner", PLANNERS)
def test_path_is_contiguous_and_free(planner):
    blocked = open_grid()
    blocked[3, 1:6] = True  # a wall with a gap on both ends
    result = planner(blocked, (0, 0), (6, 6))
    assert result.found
    assert result.path[0] == (0, 0) and result.path[-1] == (6, 6)
    for current, nxt in zip(result.path, result.path[1:]):
        assert abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]) == 1
        assert not blocked[nxt]


@pytest.mark.parametrize("planner", PLANNERS)
def test_path_length_matches_the_bfs_distance_field(planner):
    blocked = open_grid(9)
    blocked[4, 0:7] = True
    walkable = ~blocked
    start, goal = (0, 0), (8, 8)
    expected = int(bfs_distance_field(walkable, goal)[start])
    assert planner(blocked, start, goal).length == expected


@pytest.mark.parametrize("planner", PLANNERS)
def test_unreachable_goal_reports_no_path(planner):
    blocked = open_grid()
    blocked[:, 3] = True  # full vertical wall splits the grid
    result = planner(blocked, (0, 0), (0, 6))
    assert not result.found
    assert result.path == ()


@pytest.mark.parametrize("planner", PLANNERS)
def test_start_equals_goal(planner):
    result = planner(open_grid(), (2, 2), (2, 2))
    assert result.found and result.length == 0


def test_astar_expands_no_more_nodes_than_bfs():
    """The heuristic must pay for itself; this is the measured evidence."""
    blocked = open_grid(15)
    blocked[7, 0:12] = True
    start, goal = (0, 0), (14, 14)
    astar_result = astar(blocked, start, goal)
    bfs_result = bfs(blocked, start, goal)
    assert astar_result.length == bfs_result.length
    assert astar_result.nodes_expanded <= bfs_result.nodes_expanded


def test_blocked_start_can_still_plan_out():
    blocked = open_grid()
    blocked[0, 0] = True  # robot standing on a cell marked blocked
    result = astar(blocked, (0, 0), (3, 3))
    assert result.found


def test_astar_controller_completes_the_default_scenario():
    env = make_env("default")
    policy = PlannerPolicy()
    observation, _ = env.reset(seed=7)
    policy.reset(env, 7)
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _reward, terminated, truncated, info = env.step(policy.act(observation, env))
    assert info["is_success"]
    assert env.sim.counters.collisions == 0
    assert env.sim.counters.moves == env.sim.counters.optimal_path_length


def test_astar_controller_charges_before_running_out():
    """With an expensive battery the controller must detour to a charger."""
    config = ScenarioConfig(
        name="battery_test",
        max_steps=600,
        battery=BatteryConfig(start_level=40.0, move_cost=1.5, charge_rate=10.0),
        tasks=TaskConfig(tasks_per_episode=2),
    )
    env = make_env(config)
    policy = PlannerPolicy()
    observation, _ = env.reset(seed=3)
    policy.reset(env, 3)
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _reward, terminated, truncated, info = env.step(policy.act(observation, env))
    assert info["is_success"], "controller failed to manage the battery"
    assert env.sim.counters.charging_events > 0


def test_astar_controller_avoids_dynamic_obstacles():
    config = ScenarioConfig(
        name="dynamic_test",
        max_steps=500,
        obstacles=ObstacleConfig(n_dynamic=10, move_probability=0.6),
    )
    env = make_env(config)
    policy = PlannerPolicy()
    successes, collisions = 0, 0
    for seed in range(5):
        observation, _ = env.reset(seed=seed)
        policy.reset(env, seed)
        terminated = truncated = False
        while not (terminated or truncated):
            observation, _r, terminated, truncated, info = env.step(policy.act(observation, env))
        successes += bool(info.get("is_success"))
        collisions += env.sim.counters.collisions
    assert successes == 5
    assert collisions == 0


def test_random_policy_is_reproducible_for_a_seed():
    env = make_env("default")
    def trace(seed: int) -> list[int]:
        observation, _ = env.reset(seed=seed)
        policy = RandomPolicy(seed=0)
        policy.reset(env, seed)
        return [policy.act(observation, env) for _ in range(20)]

    assert trace(5) == trace(5)
