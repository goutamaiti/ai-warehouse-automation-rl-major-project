"""Run the classical controllers over a scenario and record the results.

Example::

    python -m baselines.evaluate_baselines --scenario default --episodes 30

Every controller is evaluated on the *same* list of episode seeds, so the
comparison is paired: differences come from the controller, not from an easier
draw of tasks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analytics.metrics import summarise
from analytics.reports import (
    build_results_payload,
    comparison_table,
    save_json,
    write_markdown_report,
)
from analytics.runner import run_episodes
from environment import make_env

from .controller import PlannerPolicy, RandomPolicy

CONTROLLERS = {
    "astar": lambda: PlannerPolicy(planner="astar"),
    "bfs": lambda: PlannerPolicy(planner="bfs"),
    "random": lambda: RandomPolicy(seed=0),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="default", help="scenario name or path")
    parser.add_argument("--episodes", type=int, default=30, help="episodes per controller")
    parser.add_argument("--seed", type=int, default=1000, help="first episode seed")
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["astar", "random"],
        choices=sorted(CONTROLLERS),
    )
    parser.add_argument(
        "--record",
        type=int,
        default=1,
        help="number of episodes per controller to save as a replay for the dashboard",
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--configs-dir", default="configs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = list(range(args.seed, args.seed + args.episodes))

    summaries = []
    all_episodes = []
    for name in args.controllers:
        env = make_env(args.scenario, configs_dir=args.configs_dir)
        policy = CONTROLLERS[name]()
        episodes = run_episodes(
            env,
            policy,
            seeds,
            record_first=args.record,
            record_dir=args.episodes_dir,
        )
        all_episodes.extend(episodes)
        summary = summarise(episodes)
        if isinstance(policy, PlannerPolicy):
            summary["nodes_expanded_last_episode"] = policy.nodes_expanded
        summaries.append(summary)

    scenario_name = env.config.name
    print(f"\nScenario: {scenario_name}   episodes per controller: {len(seeds)}\n")
    print(comparison_table(summaries))

    payload = build_results_payload(scenario_name, seeds, summaries, all_episodes)
    json_path = save_json(payload, Path(args.results_dir) / f"{scenario_name}_baselines.json")
    md_path = write_markdown_report(
        Path(args.results_dir) / f"{scenario_name}_baselines.md",
        scenario_name,
        summaries,
        seeds,
        notes=[
            "Classical controllers see the full map and all obstacle positions; "
            "the RL policy only sees its observation vector.",
            "Path efficiency is optimal-path-length / driven-path-length, "
            "averaged over successful episodes only.",
        ],
    )
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
