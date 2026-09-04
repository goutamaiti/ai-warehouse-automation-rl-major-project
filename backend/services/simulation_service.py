"""Service layer between the HTTP API and the simulation code.

The API layer stays thin on purpose: everything that knows about scenarios,
controllers and files lives here, so the same functions can be called from a
notebook or a script without going through HTTP.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from analytics.runner import run_episode
from baselines.controller import PlannerPolicy, RandomPolicy
from environment import make_env
from simulation.config import ScenarioConfig, load_scenario
from simulation.warehouse import WarehouseLayout, layout_from_grid

#: Repository root (backend/services/simulation_service.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = Path(os.getenv("WAREHOUSE_CONFIGS_DIR", REPO_ROOT / "configs"))
EPISODES_DIR = Path(os.getenv("WAREHOUSE_EPISODES_DIR", REPO_ROOT / "data" / "episodes"))
RESULTS_DIR = Path(os.getenv("WAREHOUSE_RESULTS_DIR", REPO_ROOT / "data" / "results"))
MODELS_DIR = Path(os.getenv("WAREHOUSE_MODELS_DIR", REPO_ROOT / "rl_agent" / "models"))

#: Upper bound on a live run requested through the API, so a single HTTP call
#: can never tie up the server for an unbounded amount of time.
MAX_LIVE_STEPS = 2000

CONTROLLERS = ("astar", "bfs", "random", "ppo")


class ServiceError(Exception):
    """Raised for conditions the API turns into a 4xx response."""


def scenario_files() -> list[Path]:
    """Every scenario YAML that ships with the project."""
    files = sorted(CONFIGS_DIR.glob("*.yaml")) + sorted((CONFIGS_DIR / "scenarios").glob("*.yaml"))
    return [path for path in files if path.is_file()]


@lru_cache(maxsize=32)
def load_scenario_config(name: str) -> ScenarioConfig:
    """Load a scenario by name, cached because layouts are deterministic."""
    for path in scenario_files():
        if path.stem == name:
            return load_scenario(path)
    raise ServiceError(f"unknown scenario {name!r}")


def list_scenarios() -> list[dict[str, Any]]:
    scenarios = []
    for path in scenario_files():
        config = load_scenario(path)
        scenarios.append(
            {
                "name": path.stem,
                "title": config.name,
                "description": config.description.strip(),
                "width": config.layout.width,
                "height": config.layout.height,
                "max_steps": config.max_steps,
                "dynamic_obstacles": config.obstacles.n_dynamic,
                "tasks_per_episode": config.tasks.tasks_per_episode,
                "battery_start": config.battery.start_level,
            }
        )
    return scenarios


def get_layout(name: str) -> dict[str, Any]:
    env = make_env(load_scenario_config(name))
    return env.sim.layout.to_dict()


#: Which trained policy belongs to which scenario. A scenario without an entry
#: falls back to ``ppo_<scenario>.zip`` and then to the default policy.
PPO_MODEL_TAGS = {
    "default": "ppo_default",
    "dynamic_obstacles": "ppo_dynamic",
    "battery_constrained": "ppo_battery",
}


def ppo_model_path(scenario: str) -> Path | None:
    """Best available trained policy for a scenario, or ``None`` if there is one."""
    candidates = [
        MODELS_DIR / f"{PPO_MODEL_TAGS.get(scenario, f'ppo_{scenario}')}.zip",
        MODELS_DIR / "ppo_default.zip",
    ]
    return next((path for path in candidates if path.exists()), None)


def _build_policy(controller: str, scenario: str):
    if controller == "astar":
        return PlannerPolicy(planner="astar")
    if controller == "bfs":
        return PlannerPolicy(planner="bfs")
    if controller == "random":
        return RandomPolicy(seed=0)
    if controller == "ppo":
        from rl_agent.evaluate import PPOPolicy

        model_path = ppo_model_path(scenario)
        if model_path is None:
            raise ServiceError(
                "no trained PPO model found; run `python -m rl_agent.train` first"
            )
        return PPOPolicy.load(model_path)
    raise ServiceError(f"unknown controller {controller!r}")


def run_live_episode(
    scenario: str,
    controller: str,
    seed: int,
    layout: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Run one episode now and return the full replay.

    This is what the dashboard's "Run" button calls, for both a named
    scenario and a user-drawn warehouse from the editor. When ``layout`` is
    given it overrides the scenario's own generated map; every other rule
    (battery, reward weights, obstacles, step budget) still comes from
    ``scenario``, so the numbers stay tied to a real, documented ruleset
    instead of an arbitrary one invented per request. The step budget is
    capped by that ruleset's ``max_steps``, validated here as well.
    """
    config = load_scenario_config(scenario)
    if config.max_steps > MAX_LIVE_STEPS:
        raise ServiceError(
            f"scenario {scenario!r} allows {config.max_steps} steps, "
            f"more than the API limit of {MAX_LIVE_STEPS}"
        )
    custom_layout: WarehouseLayout | None = None
    if layout is not None:
        try:
            custom_layout = layout_from_grid(layout)
        except ValueError as exc:
            raise ServiceError(f"invalid layout: {exc}") from exc
    env = make_env(config, layout=custom_layout)
    policy = _build_policy(controller, scenario)
    _metrics, recorder = run_episode(env, policy, seed=seed, record=True)
    assert recorder is not None
    return recorder.to_dict()


def list_recorded_episodes() -> list[dict[str, Any]]:
    """Replays that were saved by an evaluation run and committed to the repo."""
    episodes = []
    for path in sorted(EPISODES_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        episodes.append(
            {
                "id": path.stem,
                "scenario": data.get("scenario"),
                "controller": data.get("controller"),
                "seed": data.get("seed"),
                "frames": len(data.get("frames", [])),
                "summary": data.get("summary", {}),
            }
        )
    return episodes


def load_recorded_episode(episode_id: str) -> dict[str, Any]:
    path = (EPISODES_DIR / f"{episode_id}.json").resolve()
    # Guard against path traversal via a crafted episode id.
    if EPISODES_DIR.resolve() not in path.parents or not path.is_file():
        raise ServiceError(f"unknown episode {episode_id!r}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_results() -> list[dict[str, Any]]:
    """Evaluation result files produced by the evaluation scripts."""
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        results.append(
            {
                "id": path.stem,
                "scenario": data.get("scenario"),
                "generated_at": data.get("generated_at"),
                "summaries": data.get("summaries", []),
                "n_episodes": len(data.get("episodes", [])),
            }
        )
    return results


def available_controllers() -> list[dict[str, Any]]:
    """Controllers the API can run right now, and why one may be unavailable."""
    ppo_ready = any(MODELS_DIR.glob("ppo_*.zip"))
    return [
        {"name": "astar", "available": True, "note": "shortest-path planner with battery logic"},
        {"name": "bfs", "available": True, "note": "uninformed search baseline"},
        {"name": "random", "available": True, "note": "random actions, lower bound"},
        {
            "name": "ppo",
            "available": ppo_ready,
            "note": "trained policy" if ppo_ready else "no trained model in rl_agent/models",
        },
    ]
