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


# ----------------------------------------------------------------------
# Battery-reserve shaping (configs/scenarios/battery_tuned.yaml)
# ----------------------------------------------------------------------


def test_battery_reserve_term_is_off_by_default():
    """Every scenario predating the term must keep its original reward.

    The committed results under data/results/ were produced before this term
    existed; if it were active by default those numbers would silently stop
    being reproducible.
    """
    env = make_env("default")
    assert env.config.reward.battery_reserve_weight == 0.0
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(Action.UP))
    assert info["reward_components"]["battery_reserve"] == 0.0


def test_battery_reserve_potential_is_zero_while_a_charger_is_in_reach():
    """The term must stay silent during ordinary navigation.

    The robot starts on a charging station, so its charge deficit is zero by
    construction and the potential must be too.
    """
    env = make_env("battery_tuned")
    env.reset(seed=0)
    assert env.config.reward.battery_reserve_weight > 0.0
    assert env._reserve_potential() == 0.0


def test_battery_reserve_potential_goes_negative_when_the_robot_is_short_of_charge():
    env = make_env("battery_tuned")
    env.reset(seed=0)
    # Strand the robot: far from the charger it started on, nearly flat.
    env.sim.robot.position = env.sim.task.pickup
    env.sim.robot.battery = 1.0
    stranded = env._reserve_potential()
    assert stranded < 0.0
    # Charging it up must raise the potential (reduce the deficit).
    env.sim.robot.battery = 40.0
    assert env._reserve_potential() > stranded


def test_battery_reserve_shaping_cannot_be_farmed_by_a_round_trip():
    """A loop returning to the same state must pay exactly zero.

    This is the property that a naive "+reward per charging event" bonus would
    not have: the robot could step off a charger and back on forever. Because
    the term is potential-based with shaping_gamma == 1, the shaping over any
    cycle telescopes to zero.
    """
    env = make_env("battery_tuned")
    env.reset(seed=0)
    start = env.sim.robot.position
    battery = env.sim.robot.battery

    total = 0.0
    # Drive away from the start cell and come back, pinning the battery so the
    # robot genuinely returns to the same state.
    for action in (Action.UP, Action.DOWN):
        before = env._reserve_potential()
        env.sim.step(int(action))
        env.sim.robot.battery = battery
        total += env.config.reward.shaping_gamma * env._reserve_potential() - before

    assert env.sim.robot.position == start
    assert total == pytest.approx(0.0, abs=1e-9)


def test_dying_is_never_profitable_under_the_corrected_penalty():
    """The fix, stated as an assertion.

    On battery_constrained a policy could deliver one package, die, and still
    score positively: the depletion penalty (10) was cheaper than the pickup,
    delivery and progress credit banked beforehand - the measured dying
    episodes scored up to 44.6. The corrected penalty must exceed everything an
    episode can bank, so that no amount of partial progress makes dying
    attractive.
    """
    bankable = (
        env_reward("battery_tuned").pickup_reward
        + env_reward("battery_tuned").delivery_reward
    ) * 2
    for scenario in ("battery_tuned",):
        reward = env_reward(scenario)
        assert reward.battery_depleted_penalty > bankable, scenario
        # ... and the old scenario is kept unchanged, as the comparison point.
    assert env_reward("battery_constrained").battery_depleted_penalty == 10.0


def env_reward(scenario: str):
    return make_env(scenario).config.reward


def test_battery_scenarios_share_one_world():
    """The three battery arms must differ only in the reward.

    If the warehouse, battery model or task count drifted between them, the
    comparison in docs/experiments.md would not be measuring the reward change.
    """
    import dataclasses

    base = make_env("battery_constrained").config
    for scenario in ("battery_tuned",):
        other = make_env(scenario).config
        for section in ("layout", "battery", "obstacles", "tasks"):
            assert getattr(base, section) == getattr(other, section), (scenario, section)
        assert base.max_steps == other.max_steps
        assert base.observation_window == other.observation_window
        # Only the reward differs.
        assert dataclasses.asdict(base.reward) != dataclasses.asdict(other.reward)


# ----------------------------------------------------------------------
# Randomised start charger (ScenarioConfig.random_start_charger)
# ----------------------------------------------------------------------


def _roaming_env():
    """battery_tuned's world with the start charger randomised.

    Built from the config rather than a scenario file: the flag exists because
    pinning the start to charging_stations[0] taught the trained policy that
    "the charger is on the left" - it charged there 62 times out of 62 across
    30 episodes and never used the other station, walking past a nearer charger
    in 19 of 30 episodes at a cost of ~7.6 cells each. The flag is the fix for
    that; no scenario currently ships with it enabled.
    """
    import dataclasses

    from simulation.config import find_scenario, load_scenario

    config = load_scenario(find_scenario("battery_tuned"))
    return WarehouseEnv(dataclasses.replace(config, random_start_charger=True))


def test_start_charger_is_pinned_by_default():
    """Scenarios predating the flag must keep their exact episode definitions."""
    env = make_env("battery_tuned")
    assert env.config.random_start_charger is False
    starts = set()
    for seed in range(1000, 1010):
        env.reset(seed=seed)
        starts.add(env.sim.robot.position)
    assert starts == {env.sim.layout.charging_stations[0]}


def test_random_start_visits_more_than_one_charger():
    """The fix for the "always charges on the left" habit.

    A fixed start made "the charger is to the left" true for the first charge
    of every episode, so the policy memorised one absolute route and ignored
    its nearest-charger observation.
    """
    env = _roaming_env()
    assert env.config.random_start_charger is True
    starts = {env.reset(seed=seed) and env.sim.robot.position for seed in range(1000, 1030)}
    assert len(starts) > 1, "randomised start never produced a second charger"
    assert starts <= set(env.sim.layout.charging_stations)


def test_random_start_is_still_reproducible():
    """Varied, not nondeterministic: reset(seed=k) must rebuild episode k."""
    env = _roaming_env()
    first = [env.reset(seed=seed) and env.sim.robot.position for seed in range(1000, 1012)]
    second = [env.reset(seed=seed) and env.sim.robot.position for seed in range(1000, 1012)]
    assert first == second


def test_random_start_keeps_the_robot_on_a_charger():
    """With a 45% starting charge, starting off-charger would change solvability."""
    from simulation.warehouse import CellType

    env = _roaming_env()
    for seed in range(1000, 1020):
        env.reset(seed=seed)
        assert env.sim.layout.cell(env.sim.robot.position) is CellType.CHARGING
