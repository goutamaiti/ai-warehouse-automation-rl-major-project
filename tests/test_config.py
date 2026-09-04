"""Tests for scenario configuration loading and inheritance."""

from __future__ import annotations

import pytest

from simulation.config import (
    ScenarioConfig,
    find_scenario,
    load_scenario,
    scenario_from_dict,
)

CONFIGS = "configs"


def test_default_scenario_loads():
    config = load_scenario(find_scenario("default", CONFIGS))
    assert config.name == "default"
    assert config.layout.width == 21
    assert config.reward.shaping_gamma == 1.0


@pytest.mark.parametrize(
    "name", ["simple_static", "complex_static", "dynamic_obstacles", "battery_constrained"]
)
def test_every_shipped_scenario_loads_and_keeps_its_name(name: str):
    config = load_scenario(find_scenario(name, CONFIGS))
    assert config.name == name
    assert config.max_steps > 0


def test_extends_inherits_unspecified_values():
    default = load_scenario(find_scenario("default", CONFIGS))
    dynamic = load_scenario(find_scenario("dynamic_obstacles", CONFIGS))
    assert dynamic.obstacles.n_dynamic == 8            # overridden
    assert dynamic.layout.width == default.layout.width  # inherited
    assert dynamic.reward == default.reward             # inherited wholesale


def test_partial_section_override_keeps_sibling_values():
    default = load_scenario(find_scenario("default", CONFIGS))
    battery = load_scenario(find_scenario("battery_constrained", CONFIGS))
    assert battery.battery.start_level == 45.0
    assert battery.battery.capacity == default.battery.capacity


def test_unknown_key_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match="unknown key"):
        scenario_from_dict({"name": "typo", "max_stepss": 10})


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ValueError, match="battery"):
        scenario_from_dict({"battery": {"capacty": 10}})


def test_missing_scenario_raises():
    with pytest.raises(FileNotFoundError):
        find_scenario("does_not_exist", CONFIGS)


def test_scenario_round_trips_through_a_dict():
    config = load_scenario(find_scenario("complex_static", CONFIGS))
    assert scenario_from_dict(config.to_dict()) == config


def test_defaults_are_usable_without_any_file():
    config = ScenarioConfig()
    assert config.layout.width > 0 and config.max_steps > 0
