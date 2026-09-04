"""Metrics, reporting and the shared episode runner."""

from .metrics import EpisodeMetrics, summarise
from .reports import build_results_payload, comparison_table, save_json, write_markdown_report
from .runner import Policy, run_episode, run_episodes

__all__ = [
    "EpisodeMetrics",
    "Policy",
    "build_results_payload",
    "comparison_table",
    "run_episode",
    "run_episodes",
    "save_json",
    "summarise",
    "write_markdown_report",
]
