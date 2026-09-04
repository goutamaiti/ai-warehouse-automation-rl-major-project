"""Turn aggregated metrics into files a human (or a report) can read.

Every artefact written here carries the information needed to reproduce it:
scenario, controller, episode seeds and the timestamp of the run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .metrics import EpisodeMetrics

#: Columns of the comparison table, as (label, key in the summary dict).
TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Success rate", "success_rate"),
    ("Steps", "steps"),
    ("Path length", "path_length"),
    ("Path efficiency", "path_efficiency"),
    ("Collisions", "collisions"),
    ("Energy %", "energy_consumed"),
    ("Charging events", "charging_events"),
    ("Reward", "total_reward"),
)


def _format_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, dict):
        mean, ci95, n = value.get("mean"), value.get("ci95"), value.get("n", 0)
        if mean is None or n == 0:
            return "n/a"
        if n == 1:
            return f"{mean:.2f}"
        return f"{mean:.2f} ± {ci95:.2f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def comparison_table(summaries: Sequence[dict]) -> str:
    """Markdown table comparing controllers on one scenario.

    Numbers are ``mean ± 95% confidence interval half-width`` over the episodes
    that entered the average; ``n/a`` means no episode qualified (for example
    path efficiency when nothing was ever delivered).
    """
    if not summaries:
        return "_no results_"
    header = "| Controller | Episodes | " + " | ".join(c[0] for c in TABLE_COLUMNS) + " |"
    divider = "|---" * (len(TABLE_COLUMNS) + 2) + "|"
    lines = [header, divider]
    for summary in summaries:
        cells = [
            str(summary.get("controller", "?")),
            f"{summary.get('n_successful', 0)}/{summary.get('n_episodes', 0)}",
        ]
        for _label, key in TABLE_COLUMNS:
            cells.append(_format_cell(summary.get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def build_results_payload(
    scenario: str,
    seeds: Sequence[int],
    summaries: Sequence[dict],
    episodes: Sequence[EpisodeMetrics],
    extra: dict | None = None,
) -> dict:
    """Assemble the machine-readable result file for one evaluation run."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": scenario,
        "seeds": list(seeds),
        "summaries": list(summaries),
        "episodes": [episode.to_dict() for episode in episodes],
        **(extra or {}),
    }


def write_markdown_report(
    path: str | Path,
    scenario: str,
    summaries: Sequence[dict],
    seeds: Sequence[int],
    notes: Sequence[str] = (),
) -> Path:
    """Write a short, self-describing markdown report next to the JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seed_range = f"{min(seeds)}-{max(seeds)}" if seeds else "none"
    body = [
        f"# Evaluation: {scenario}",
        "",
        f"- Generated: {generated}",
        f"- Episodes per controller: {len(seeds)} (seeds {seed_range})",
        "",
        comparison_table(summaries),
        "",
    ]
    if notes:
        body += ["## Notes", ""] + [f"- {note}" for note in notes] + [""]
    path.write_text("\n".join(body), encoding="utf-8")
    return path
