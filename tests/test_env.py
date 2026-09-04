"""Tests for the Gymnasium environment: API compliance, observations, reward."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment import make_env
from environment.warehouse_env import N_SCALAR_FEATURES, WarehouseEnv
from simulation.config import ObstacleConfig, ScenarioConfig
from simulation.engine import Action


def test_environment_passes_the_gymnasium_api_checker():
    check_env(make_env("default"), skip_render_check=False)


def test_observation_shape_bounds_and_labels():
    env = make_env("default")
    observation, info = env.reset(seed=0)
    window = env.config.observation_window
    assert observation.shape == (N_SCALAR_FEATURES + window * window,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert len(env.observation_labels()) == observation.shape[0]
    assert info["optimal_path_length"] > 0


def test_observation_reports_battery_and_payload():
    env = make_env("default")
    observation, _ = env.reset(seed=0)
    labels = env.observation_labels()
    battery_index = labels.index("battery_fraction")
    carrying_index = labels.index("carrying")
    assert observation[battery_index] == pytest.approx(1.0)
    assert observation[carrying_index] == 0.0
    env.sim.robot.carrying = True
    assert env._observation()[carrying_index] == 1.0


def test_local_patch_marks_walls_as_blocked():
    env = make_env("default")
    env.reset(seed=0)
    labels = env.observation_labels()
    observation = env._observation()
    # The robot starts on the bottom station row, so the cell below is a wall.
    below_index = labels.index("patch[+1,+0]")
    assert observation[below_index] == 1.0


def test_reset_with_the_same_seed_gives_the_same_observation():
    env = make_env("dynamic_obstacles")
    first, _ = env.reset(seed=99)
    second, _ = env.reset(seed=99)
    assert np.array_equal(first, second)
    third, _ = env.reset(seed=100)
    assert not np.array_equal(first, third)


def test_waiting_far_from_the_goal_is_never_profitable():
    """Regression test for a reward-shaping exploit.

    With a shaping discount below 1 the shaping term pays ``distance * (1 -
    gamma)`` for standing still, which can exceed the step penalty and teach the
    agent to loiter. The default config must not allow that.
    """
    env = make_env("default")
    env.reset(seed=0)
    _, reward, _, _, info = env.step(int(Action.WAIT))
    assert reward < 0
    assert info["reward_components"]["progress"] <= 0


def test_bumping_into_a_wall_is_penalised():
    env = make_env("default")
    env.reset(seed=0)
    _, reward, _, _, info = env.step(int(Action.DOWN))  # wall below the start cell
    assert info["reward_components"]["collision"] < 0
    assert reward < 0


def test_moving_towards_the_target_earns_progress_reward():
    env = make_env("default")
    env.reset(seed=0)
    before = env.sim.distance_to_target()
    # Try each direction until one reduces the distance to the target.
    for action in (Action.UP, Action.LEFT, Action.RIGHT):
        candidate = make_env("default")
        candidate.reset(seed=0)
        _, reward, _, _, info = candidate.step(int(action))
        if candidate.sim.distance_to_target() < before:
            assert info["reward_components"]["progress"] == pytest.approx(1.0)
            assert reward > 0
            return
    pytest.fail("no action reduced the distance to the target")


def test_episode_ends_with_a_summary_and_success_flag():
    from baselines.controller import PlannerPolicy

    env = make_env("default")
    policy = PlannerPolicy()
    observation, _ = env.reset(seed=11)
    policy.reset(env, 11)
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _reward, terminated, truncated, info = env.step(policy.act(observation, env))
    assert terminated and not truncated
    assert info["is_success"] is True
    assert info["episode_summary"]["tasks_delivered"] == 1


def test_timeout_truncates_instead_of_terminating():
    env = WarehouseEnv(ScenarioConfig(max_steps=3))
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(int(Action.WAIT))
    assert truncated and not terminated
    assert info["is_success"] is False


def test_delivery_reward_is_paid_once():
    from baselines.controller import PlannerPolicy

    env = make_env("default")
    policy = PlannerPolicy()
    observation, _ = env.reset(seed=4)
    policy.reset(env, 4)
    deliveries = 0
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _reward, terminated, truncated, info = env.step(policy.act(observation, env))
        deliveries += info["reward_components"]["delivery"] > 0
    assert deliveries == 1


def test_invalid_observation_window_is_rejected():
    with pytest.raises(ValueError):
        WarehouseEnv(ScenarioConfig(observation_window=4))


def test_obstacles_appear_in_the_local_patch():
    config = ScenarioConfig(obstacles=ObstacleConfig(n_dynamic=25, move_probability=1.0))
    env = WarehouseEnv(config)
    env.reset(seed=1)
    patch_values = set()
    for _ in range(60):
        observation = env._observation()
        patch_values.update(np.unique(observation[N_SCALAR_FEATURES:]).tolist())
        if env.sim.done:
            break
        env.step(int(Action.WAIT))
    assert 0.5 in patch_values, "no dynamic obstacle was ever visible in the patch"


def test_delivery_step_is_not_punished_by_the_shaping_term():
    """Regression test: the target must not jump when a task completes.

    If ``Task.target`` fell back to the pickup point after delivery, the
    distance-to-target would jump from 1 to the full width of the warehouse on
    the final step, and the shaping term would cancel out the delivery reward.
    """
    from baselines.controller import PlannerPolicy

    env = make_env("default")
    policy = PlannerPolicy()
    observation, _ = env.reset(seed=21)
    policy.reset(env, 21)
    terminated = truncated = False
    final_components = None
    while not (terminated or truncated):
        observation, _reward, terminated, truncated, info = env.step(policy.act(observation, env))
        if info["reward_components"]["delivery"] > 0:
            final_components = info["reward_components"]
    assert final_components is not None
    assert final_components["progress"] >= 0
    assert sum(final_components.values()) > env.config.reward.delivery_reward / 2
