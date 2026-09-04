"""Tests for metric aggregation and report generation."""

from __future__ import annotations

import json

import pytest

from analytics.metrics import EpisodeMetrics, summarise
from analytics.reports import build_results_payload, comparison_table, save_json
from analytics.runner import run_episode
from baselines.controller import PlannerPolicy
from environment import make_env
from simulation.engine import EpisodeCounters


def make_metrics(seed: int, success: bool, steps: int = 30, moves: int = 30) -> EpisodeMetrics:
    counters = EpisodeCounters(
        steps=steps,
        moves=moves,
        collisions=2,
        static_collisions=2,
        energy_consumed=15.0,
        tasks_delivered=1 if success else 0,
        optimal_path_length=27,
        termination_reason="all_tasks_delivered" if success else "timeout",
    )
    return EpisodeMetrics.from_counters(
        counters,
        controller="test",
        scenario="unit",
        seed=seed,
        total_reward=10.0 if success else -5.0,
        tasks_required=1,
    )


def test_failed_episodes_have_no_efficiency_or_delivery_time():
    failed = make_metrics(seed=0, success=False)
    assert failed.success is False
    assert failed.path_efficiency is None
    assert failed.delivery_time is None


def test_path_efficiency_is_optimal_over_driven():
    metrics = make_metrics(seed=0, success=True, moves=54)
    assert metrics.path_efficiency == pytest.approx(27 / 54)


def test_summary_averages_only_successful_episodes_for_path_metrics():
    episodes = [make_metrics(0, True, steps=30), make_metrics(1, False, steps=400)]
    summary = summarise(episodes)
    assert summary["n_episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["steps"]["n"] == 1          # only the successful episode
    assert summary["steps"]["mean"] == 30
    assert summary["collisions"]["n"] == 2     # collisions count for every episode
    assert summary["termination_reasons"] == {"all_tasks_delivered": 1, "timeout": 1}


def test_summary_of_no_episodes_is_empty_not_invented():
    assert summarise([]) == {"n_episodes": 0}


def test_table_reports_na_when_nothing_succeeded():
    summary = summarise([make_metrics(0, False), make_metrics(1, False)])
    table = comparison_table([summary])
    assert "n/a" in table
    assert "| test |" in table


def test_results_payload_round_trips_through_json(tmp_path):
    episodes = [make_metrics(0, True), make_metrics(1, False)]
    payload = build_results_payload("unit", [0, 1], [summarise(episodes)], episodes)
    path = save_json(payload, tmp_path / "results.json")
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["scenario"] == "unit"
    assert len(restored["episodes"]) == 2
    assert restored["episodes"][1]["path_efficiency"] is None


def test_run_episode_records_a_replay_matching_the_step_count():
    env = make_env("default")
    metrics, recorder = run_episode(env, PlannerPolicy(), seed=1, record=True)
    assert metrics.success
    assert recorder is not None
    # one frame for the initial state plus one per step
    assert len(recorder.frames) == metrics.steps + 1
    assert recorder.summary["tasks_delivered"] == 1
    assert recorder.frames[-1]["task"]["status"] == "delivered"


def test_recorder_file_contains_layout_and_frames(tmp_path):
    env = make_env("default")
    _metrics, recorder = run_episode(env, PlannerPolicy(), seed=2, record=True)
    path = recorder.save(tmp_path / "episode.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["layout"]["grid"]
    assert data["frames"][0]["robot"]["battery"] > 0
    assert data["summary"]["controller"] == "astar"


def test_recorded_frames_carry_reward_and_the_weights_that_produced_it():
    env = make_env("default")
    _metrics, recorder = run_episode(env, PlannerPolicy(), seed=3, record=True)
    assert "reward" not in recorder.frames[0]
    assert recorder.frames[1]["reward"] is not None
    assert set(recorder.frames[1]["reward_components"]) >= {
        "step_penalty", "progress", "pickup", "delivery", "collision",
        "wait", "energy", "battery_depleted", "timeout",
    }
    assert recorder.reward_config["delivery_reward"] == 20.0
    assert recorder.to_dict()["reward_config"] == recorder.reward_config
