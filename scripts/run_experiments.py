"""Reproduce every number in ``data/results`` with a single command.

    python scripts/run_experiments.py            # full run (trains 3 policies)
    python scripts/run_experiments.py --quick    # short run, for a smoke test
    python scripts/run_experiments.py --skip-training

The script runs each step as a subprocess so that the exact command line is
printed and can be copied into the report's methodology section.

Scenarios that have a trained policy are evaluated with ``rl_agent.evaluate``
(PPO plus every classical controller in one table); the remaining scenarios are
evaluated with ``baselines.evaluate_baselines``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: scenario -> (model tag, training timesteps for a full run)
TRAINED_SCENARIOS = {
    "default": ("ppo_default", 300_000),
    "dynamic_obstacles": ("ppo_dynamic", 400_000),
    "battery_constrained": ("ppo_battery", 600_000),
}

#: Scenarios evaluated with classical controllers only.
BASELINE_ONLY_SCENARIOS = ("simple_static", "complex_static")

EPISODES = 30
FIRST_SEED = 1000
CONTROLLERS = ["astar", "bfs", "random"]


def run(command: list[str], dry_run: bool = False) -> None:
    printable = " ".join(command[1:] if command[0] == sys.executable else command)
    print(f"\n$ python {printable}", flush=True)
    if dry_run:
        return
    started = time.time()
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}")
    print(f"  ({time.time() - started:.1f}s)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short training runs")
    parser.add_argument("--skip-training", action="store_true", help="reuse existing models")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args(argv)

    python = sys.executable

    if not args.skip_training:
        for scenario, (tag, timesteps) in TRAINED_SCENARIOS.items():
            run(
                [
                    python, "-m", "rl_agent.train",
                    "--scenario", scenario,
                    "--timesteps", str(20_000 if args.quick else timesteps),
                    "--tag", tag,
                ],
                args.dry_run,
            )

    for scenario in BASELINE_ONLY_SCENARIOS:
        run(
            [
                python, "-m", "baselines.evaluate_baselines",
                "--scenario", scenario,
                "--episodes", str(args.episodes),
                "--seed", str(FIRST_SEED),
                "--controllers", *CONTROLLERS,
                "--record", "1",
            ],
            args.dry_run,
        )

    for scenario, (tag, _timesteps) in TRAINED_SCENARIOS.items():
        model = f"rl_agent/models/{tag}.zip"
        if not args.dry_run and not (REPO_ROOT / model).exists():
            print(f"! skipping {scenario}: no model at {model}")
            continue
        run(
            [
                python, "-m", "rl_agent.evaluate",
                "--model", model,
                "--scenario", scenario,
                "--episodes", str(args.episodes),
                "--seed", str(FIRST_SEED),
                "--compare", *CONTROLLERS,
                "--record", "1",
            ],
            args.dry_run,
        )

    print("\nAll experiments finished. Results in data/results, replays in data/episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
