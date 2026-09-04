"""Episode metrics and their aggregation.

Every number produced here is derived from counters that the simulation
actually incremented during a run. Nothing in this module invents, smooths or
extrapolates a value: an empty list of episodes yields an empty summary rather
than a plausible-looking one.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from simulation.engine import EpisodeCounters

#: Metrics that are only meaningful for episodes that actually succeeded.
SUCCESS_ONLY_METRICS = ("steps", "path_length", "path_efficiency", "delivery_time")


@dataclass(frozen=True)
class EpisodeMetrics:
    """Result of one episode, ready to be aggregated or serialised."""

    controller: str
    scenario: str
    seed: int
    success: bool
    steps: int
    path_length: int
    optimal_path_length: int
    collisions: int
    static_collisions: int
    dynamic_collisions: int
    idle_steps: int
    energy_consumed: float
    charging_events: int
    tasks_delivered: int
    tasks_failed: int
    total_reward: float
    termination_reason: str

    @property
    def path_efficiency(self) -> float | None:
        """Optimal path length divided by the path actually driven.

        1.0 means the controller drove a shortest path; 0.5 means it drove
        twice as far as necessary. ``None`` for failed episodes, where the
        ratio would be meaningless.
        """
        if not self.success or self.path_length == 0:
            return None
        return self.optimal_path_length / self.path_length

    @property
    def delivery_time(self) -> int | None:
        """Steps needed to finish all tasks; ``None`` if the episode failed."""
        return self.steps if self.success else None

    @classmethod
    def from_counters(
        cls,
        counters: EpisodeCounters,
        *,
        controller: str,
        scenario: str,
        seed: int,
        total_reward: float,
        tasks_required: int,
    ) -> EpisodeMetrics:
        return cls(
            controller=controller,
            scenario=scenario,
            seed=seed,
            success=counters.tasks_delivered >= tasks_required,
            steps=counters.steps,
            path_length=counters.moves,
            optimal_path_length=counters.optimal_path_length,
            collisions=counters.collisions,
            static_collisions=counters.static_collisions,
            dynamic_collisions=counters.dynamic_collisions,
            idle_steps=counters.idle_steps,
            energy_consumed=round(counters.energy_consumed, 3),
            charging_events=counters.charging_events,
            tasks_delivered=counters.tasks_delivered,
            tasks_failed=counters.tasks_failed,
            total_reward=round(total_reward, 3),
            termination_reason=counters.termination_reason,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["path_efficiency"] = (
            round(self.path_efficiency, 4) if self.path_efficiency is not None else None
        )
        data["delivery_time"] = self.delivery_time
        return data


def _mean_std(values: Sequence[float]) -> dict[str, float | int | None]:
    """Mean, sample standard deviation and 95% CI half-width of a sample."""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95": round(ci95, 4),
        "n": n,
    }


def summarise(episodes: Iterable[EpisodeMetrics]) -> dict:
    """Aggregate episodes into the metric set required by the project plan.

    Metrics listed in :data:`SUCCESS_ONLY_METRICS` are averaged over successful
    episodes only, and the summary always reports how many episodes went into
    each average so a reader can tell a robust mean from a lucky one.
    """
    episodes = list(episodes)
    if not episodes:
        return {"n_episodes": 0}

    successful = [e for e in episodes if e.success]
    summary: dict = {
        "controller": episodes[0].controller,
        "scenario": episodes[0].scenario,
        "n_episodes": len(episodes),
        "n_successful": len(successful),
        "success_rate": round(len(successful) / len(episodes), 4),
        "termination_reasons": _count_values(e.termination_reason for e in episodes),
    }
    for name in SUCCESS_ONLY_METRICS:
        values = [getattr(episode, name) for episode in successful]
        summary[name] = _mean_std([value for value in values if value is not None])
    # Metrics that are meaningful for every episode, successful or not.
    summary["collisions"] = _mean_std([e.collisions for e in episodes])
    summary["energy_consumed"] = _mean_std([e.energy_consumed for e in episodes])
    summary["charging_events"] = _mean_std([e.charging_events for e in episodes])
    summary["idle_steps"] = _mean_std([e.idle_steps for e in episodes])
    summary["total_reward"] = _mean_std([e.total_reward for e in episodes])
    return summary


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
