"""Tripwire: the single-robot path must not drift while the fleet is built.

The multi-robot work is deliberately *additive* - ``simulation/fleet.py`` is a
sibling of ``simulation/engine.py``, not a replacement - but it shares the
layout, navigation, robot and task modules. A change there could alter
single-robot behaviour without any test noticing, and the damage would be
quiet: the six scenarios would keep passing their own tests while producing
subtly different numbers from the ones committed in ``data/results/`` and from
the ones the policies in ``rl_agent/models/`` were trained against.

These tests compare the live system against ``tests/golden/frozen_contract.json``
and fail loudly on any difference. Each assertion names what breaking it would
mean, so a failure explains itself rather than just printing two blobs.

If a failure here is *intended* - a deliberate change to the environment - the
golden file is regenerated with ``python scripts/freeze_contract.py --write``,
and the experiments it invalidates have to be re-run. That is the whole point:
the cost of changing this behaviour is made visible instead of being absorbed
silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from environment import make_env
from scripts.freeze_contract import ACTIONS, GOLDEN_PATH, PRECISION, SCENARIOS, SEED

TOLERANCE = 10.0**-PRECISION


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():  # pragma: no cover - the file is committed
        pytest.fail(
            f"{GOLDEN_PATH} is missing. Create it with "
            "`python scripts/freeze_contract.py --write`."
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_the_golden_file_covers_every_scenario_with_committed_results(golden):
    """A scenario added to the project must be frozen too, or it is unprotected."""
    assert sorted(golden["scenarios"]) == sorted(SCENARIOS)
    configs = {p.stem for p in Path("configs/scenarios").glob("*.yaml")} | {"default"}
    assert configs == set(SCENARIOS), (
        "configs/ and the frozen contract disagree. A new scenario needs adding to "
        "scripts/freeze_contract.py::SCENARIOS, a removed one taking out."
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_layout_is_unchanged(golden, scenario):
    """The map is a pure function of the config.

    If this fails, every recorded episode and every trained policy is on a
    different warehouse from the one the committed results describe.
    """
    expected = golden["scenarios"][scenario]["layout"]
    layout = make_env(scenario).sim.layout
    grid_bytes = np.ascontiguousarray(layout.grid, dtype=np.int8).tobytes()

    assert layout.height == expected["height"]
    assert layout.width == expected["width"]
    assert hashlib.sha256(grid_bytes).hexdigest() == expected["grid_sha256"]
    assert [list(p) for p in layout.storage_points] == expected["storage_points"]
    assert [list(p) for p in layout.packing_stations] == expected["packing_stations"]
    assert [list(p) for p in layout.charging_stations] == expected["charging_stations"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_observation_space_and_feature_order_are_unchanged(golden, scenario):
    """Shape *and* meaning.

    A reordering would keep the shape valid while feeding the network different
    features at each index - the policy would still load and still run, and
    would quietly be wrong.
    """
    record = golden["scenarios"][scenario]
    env = make_env(scenario)

    assert list(env.observation_space.shape) == record["spaces"]["observation_shape"]
    assert int(env.action_space.n) == record["spaces"]["n_actions"]
    assert float(env.observation_space.low.min()) == pytest.approx(
        record["spaces"]["observation_low"], abs=TOLERANCE
    )
    assert float(env.observation_space.high.max()) == pytest.approx(
        record["spaces"]["observation_high"], abs=TOLERANCE
    )
    assert env.observation_labels() == record["observation_labels"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_reset_produces_the_same_observation_and_task(golden, scenario):
    """Same seed, same problem instance, same numbers.

    This is what makes `reset(seed=k)` a reproducible experiment: the policies
    were trained and scored against exactly these values.
    """
    record = golden["scenarios"][scenario]["reset"]
    env = make_env(scenario)
    observation, info = env.reset(seed=SEED)

    assert observation.tolist() == pytest.approx(record["observation"], abs=TOLERANCE)
    assert int(info["optimal_path_length"]) == record["optimal_path_length"]
    assert info["task"]["pickup"] == record["task"]["pickup"]
    assert info["task"]["dropoff"] == record["task"]["dropoff"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_reward_terms_are_unchanged(golden, scenario):
    """Catches a reward term being added, removed or renamed.

    Any of those changes what the agent is optimising, so a policy trained
    before the change is answering a different question.
    """
    record = golden["scenarios"][scenario]
    env = make_env(scenario)
    env.reset(seed=SEED)
    _obs, _reward, _term, _trunc, info = env.step(int(ACTIONS[0]))
    assert sorted(info["reward_components"]) == record["reward_component_keys"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_fixed_action_sequence_replays_identically(golden, scenario):
    """The strongest check: dynamics and reward, step by step.

    A fixed action sequence is driven through the environment and every step is
    compared - reward, each reward component, battery, distance to target, the
    events emitted and the termination flags. A change in movement rules, in
    collision handling, in battery bookkeeping or in any reward weight shows up
    here as a concrete difference on a named step.
    """
    expected_steps = golden["scenarios"][scenario]["steps"]
    env = make_env(scenario)
    env.reset(seed=SEED)

    for expected in expected_steps:
        assert not env.sim.done, f"episode ended before step {expected['i']}"
        _obs, reward, terminated, truncated, info = env.step(int(expected["action"]))

        where = f"{scenario} step {expected['i']} (action {expected['action']})"
        assert reward == pytest.approx(expected["reward"], abs=TOLERANCE), where
        assert sorted(info["reward_components"]) == sorted(expected["components"]), where
        for term, value in expected["components"].items():
            assert info["reward_components"][term] == pytest.approx(
                value, abs=TOLERANCE
            ), f"{where}: reward term {term!r}"
        assert info["battery"] == pytest.approx(expected["battery"], abs=TOLERANCE), where
        assert int(info["distance_to_target"]) == expected["distance_to_target"], where
        assert list(info["events"]) == expected["events"], where
        assert bool(terminated) == expected["terminated"], where
        assert bool(truncated) == expected["truncated"], where

    assert len(expected_steps) > 0, f"{scenario} recorded no steps"
