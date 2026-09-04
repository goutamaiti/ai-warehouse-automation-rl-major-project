"""Typed configuration objects for the warehouse simulation.

Design note
-----------
Every tunable number used by the simulator is declared in this module and is
loaded from a YAML file in ``configs/``.  No other module is allowed to invent
a magic constant.  An experiment is therefore fully described by the pair
``(config file, seed)``, which is what makes the runs stored under
``data/results/`` reproducible.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class LayoutConfig:
    """Geometry of the generated warehouse grid.

    The generator lays out rectangular shelf blocks separated by aisles, keeps
    ``station_rows`` free rows at the bottom for packing/charging stations and
    surrounds everything with a wall.
    """

    width: int = 21
    height: int = 15
    shelf_block_height: int = 2
    shelf_block_width: int = 3
    aisle_width: int = 1
    top_aisle: int = 1
    station_rows: int = 2
    n_packing_stations: int = 2
    n_charging_stations: int = 2


@dataclass(frozen=True)
class BatteryConfig:
    """Virtual battery model of a robot (percentage based, 0-100)."""

    capacity: float = 100.0
    start_level: float = 100.0
    move_cost: float = 0.5
    idle_cost: float = 0.05
    charge_rate: float = 8.0
    low_threshold: float = 30.0
    critical_threshold: float = 10.0


@dataclass(frozen=True)
class ObstacleConfig:
    """Dynamic obstacles (simulated workers / other traffic)."""

    n_dynamic: int = 0
    move_probability: float = 0.6
    #: "random_walk" or "patrol"
    behaviour: str = "random_walk"


@dataclass(frozen=True)
class TaskConfig:
    """Delivery task generation."""

    tasks_per_episode: int = 1
    #: Minimum Manhattan distance between pickup and drop-off, keeps trivial
    #: tasks (pickup next to the packing station) out of the evaluation set.
    min_separation: int = 6


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights.

    IMPORTANT: these are *initial* values chosen by reasoning about the task
    (see ``docs/rl-formulation.md``), not values validated by a hyper-parameter
    search.  Any claim about their quality must come from a real experiment.
    """

    step_penalty: float = 0.05
    #: Weight of the potential-based shaping term (Ng et al., 1999).
    progress_weight: float = 1.0
    #: Discount used *inside* the shaping term. Keep this at 1.0: with a value
    #: below 1 a robot that stands still collects ``distance * (1 - gamma)``
    #: reward every step, which is larger than ``step_penalty`` whenever the
    #: goal is far away - i.e. the agent is paid to loiter. See
    #: docs/rl-formulation.md for the derivation.
    shaping_gamma: float = 1.0
    pickup_reward: float = 5.0
    delivery_reward: float = 20.0
    collision_penalty: float = 1.0
    wait_penalty: float = 0.1
    energy_weight: float = 0.02
    battery_depleted_penalty: float = 10.0
    timeout_penalty: float = 5.0


@dataclass(frozen=True)
class ScenarioConfig:
    """A complete, self-contained experiment definition."""

    name: str = "default"
    description: str = ""
    max_steps: int = 400
    seed: int = 0
    #: Side length of the square local occupancy patch given to the agent.
    observation_window: int = 5
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type[T], data: Mapping[str, Any], where: str) -> T:
    """Instantiate a dataclass from a mapping, rejecting unknown keys.

    Rejecting unknown keys is deliberate: a typo in a YAML file would otherwise
    be silently ignored and the experiment would quietly run with defaults.
    """
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in section '{where}'")
    return cls(**data)  # type: ignore[call-arg]


def scenario_from_dict(data: Mapping[str, Any]) -> ScenarioConfig:
    """Build a :class:`ScenarioConfig` from a plain (YAML/JSON) mapping."""
    data = dict(data)
    sections = {
        "layout": LayoutConfig,
        "battery": BatteryConfig,
        "obstacles": ObstacleConfig,
        "tasks": TaskConfig,
        "reward": RewardConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, cls in sections.items():
        section = data.pop(key, {}) or {}
        if not isinstance(section, Mapping):
            raise ValueError(f"section '{key}' must be a mapping, got {type(section)!r}")
        kwargs[key] = _build(cls, section, key)
    kwargs.update(data)
    return _build(ScenarioConfig, kwargs, "scenario")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``, recursing into nested mappings."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return dict(raw)


def load_scenario(path: str | Path, _seen: tuple[Path, ...] = ()) -> ScenarioConfig:
    """Load a scenario YAML file.

    A file may declare ``extends: <other file>`` to inherit every value of
    another scenario and override only what differs. Paths are resolved
    relative to the extending file first, then relative to ``configs/``.
    """
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"circular 'extends' chain involving {path}")
    raw = _read_yaml(path)

    parent = raw.pop("extends", None)
    if parent:
        candidates = [path.parent / parent, path.parent.parent / parent]
        base_path = next((c for c in candidates if c.is_file()), None)
        if base_path is None:
            raise FileNotFoundError(f"{path}: cannot resolve extends: {parent!r}")
        base = load_scenario(base_path, _seen + (path,)).to_dict()
        raw = _deep_merge(base, raw)

    return scenario_from_dict(raw)


def find_scenario(name: str, configs_dir: str | Path = "configs") -> Path:
    """Resolve a scenario name (``'dynamic'``) to a file path."""
    configs_dir = Path(configs_dir)
    candidates = [
        configs_dir / f"{name}.yaml",
        configs_dir / "scenarios" / f"{name}.yaml",
        Path(name),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no scenario named {name!r} under {configs_dir}")
