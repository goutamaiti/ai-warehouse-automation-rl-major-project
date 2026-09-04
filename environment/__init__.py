"""Gymnasium environment package.

Importing this package registers ``Warehouse-v0`` with Gymnasium so the
environment can also be created with ``gymnasium.make("Warehouse-v0")``.
"""

from __future__ import annotations

from pathlib import Path

from gymnasium.envs.registration import register, registry

from simulation.config import ScenarioConfig, find_scenario, load_scenario

from .warehouse_env import N_SCALAR_FEATURES, WarehouseEnv

ENV_ID = "Warehouse-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="environment.warehouse_env:WarehouseEnv")


def make_env(
    scenario: str | Path | ScenarioConfig = "default",
    render_mode: str | None = None,
    configs_dir: str | Path = "configs",
    layout=None,
) -> WarehouseEnv:
    """Create an environment from a scenario name, path or config object.

    ``layout``, when given, overrides the layout the scenario's config would
    otherwise generate - used to simulate a user-drawn warehouse under an
    existing scenario's rules (battery, reward weights, obstacles, ...).
    """
    if isinstance(scenario, ScenarioConfig):
        config = scenario
    else:
        path = Path(scenario)
        config = load_scenario(path if path.is_file() else find_scenario(str(scenario), configs_dir))
    return WarehouseEnv(config=config, render_mode=render_mode, layout=layout)


__all__ = ["ENV_ID", "N_SCALAR_FEATURES", "WarehouseEnv", "make_env"]
