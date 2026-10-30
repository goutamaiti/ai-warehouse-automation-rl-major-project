"""Evaluate a trained PPO policy and compare it with the classical baselines.

Example::

    python -m rl_agent.evaluate --model rl_agent/models/ppo_warehouse.zip \
        --scenario default --episodes 30 --compare astar

The PPO policy and every comparison controller are run on the *same* episode
seeds through the same episode loop (:mod:`analytics.runner`), so the resulting
table is a paired comparison on identical problem instances.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analytics.metrics import summarise
from analytics.reports import (
    build_results_payload,
    comparison_table,
    save_json,
    write_markdown_report,
)
from analytics.runner import run_episodes
from baselines.controller import PlannerPolicy, RandomPolicy
from environment import make_env
from environment.warehouse_env import WarehouseEnv

COMPARISON_CONTROLLERS = {
    "astar": lambda: PlannerPolicy(planner="astar"),
    "bfs": lambda: PlannerPolicy(planner="bfs"),
    "random": lambda: RandomPolicy(seed=0),
}


class PPOPolicy:
    """Adapter that lets a Stable-Baselines3 model drive the simulation."""

    name = "ppo"

    def __init__(self, model, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True) -> PPOPolicy:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit(
                "stable-baselines3 is required to evaluate a trained policy.\n"
                "Install it with:  pip install -r requirements.txt"
            ) from exc
        path = Path(path)
        if not path.exists():
            raise SystemExit(
                f"no model at {path}. Train one first:  python -m rl_agent.train"
            )
        return cls(PPO.load(path), deterministic=deterministic)

    def reset(self, env: WarehouseEnv, seed: int) -> None:
        """PPO here is a feed-forward policy, so there is no state to reset."""

    def act(self, observation: np.ndarray, env: WarehouseEnv) -> int:
        action, _state = self.model.predict(observation, deterministic=self.deterministic)
        return int(action)


class PPOChargerPolicy:
    """Hybrid: PPO drives, a classical rule picks the charging station.

    **This is not the PPO result.** It is a separate controller, and it must be
    reported under its own name. Every step it asks the classical battery logic
    (:meth:`baselines.controller.PlannerPolicy._select_goal`, the same code A*
    uses) whether the robot should be heading for a charger. If so, the move is
    planned by A* to the *nearest reachable* station; otherwise the trained
    policy chooses the action.

    Why it exists
    -------------
    The trained policy charges at the station it starts on and never at the
    other one, because every episode starts on the same charger - so "the
    charger is to the left" held for the first charge of every episode it saw
    (see docs/experiments.md). That costs a mean 7.6 cells per episode. The
    learning-side fix is to randomise the start via
    ``ScenarioConfig.random_start_charger``; this controller is the
    engineering-side alternative: keep the learned navigation and hand the
    station choice to a rule that cannot get it wrong.

    Why it is kept separate
    -----------------------
    The project's research question is whether a *learned* policy can match
    classical planning. A policy whose charger choice is made by A* is a hybrid
    of both, so folding these numbers into the "ppo" row would mean reporting
    A* under another name. Scoring it as its own controller keeps that honest -
    and the hybrid is a legitimate result in its own right, since real systems
    routinely pair learned local control with classical high-level decisions.
    """

    name = "ppo_charger"

    def __init__(self, model, deterministic: bool = True) -> None:
        from baselines.controller import PlannerPolicy

        self._ppo = PPOPolicy(model, deterministic=deterministic)
        self._planner = PlannerPolicy(planner="astar")
        #: Steps on which the classical rule overrode the learned policy.
        self.overrides = 0

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True) -> "PPOChargerPolicy":
        return cls(PPOPolicy.load(path, deterministic).model, deterministic=deterministic)

    def reset(self, env: WarehouseEnv, seed: int) -> None:
        self._ppo.reset(env, seed)
        self._planner.reset(env, seed)
        self.overrides = 0

    def act(self, observation: np.ndarray, env: WarehouseEnv) -> int:
        sim = env.sim
        goal = self._planner._select_goal(sim)
        if goal == sim.task.target:
            # Not a charging detour - let the learned policy drive.
            return self._ppo.act(observation, env)

        self.overrides += 1
        position = sim.robot.position
        if position == goal:
            from simulation.engine import Action

            return int(Action.WAIT)  # already on the charger: keep charging
        blocked = sim.occupancy_grid().copy()
        blocked[position] = False
        plan = self._planner._plan(blocked, position, goal)
        if not plan.found or plan.length == 0:
            return self._ppo.act(observation, env)
        from baselines.controller import _action_towards

        return _action_towards(position, plan.path[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--model", default="rl_agent/models/ppo_warehouse.zip")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="first episode seed; keep it identical to the baseline run to "
        "compare on the same problem instances",
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        default=["astar"],
        choices=sorted(COMPARISON_CONTROLLERS),
    )
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--configs-dir", default="configs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = list(range(args.seed, args.seed + args.episodes))

    policies = [PPOPolicy.load(args.model, deterministic=not args.stochastic)]
    policies += [COMPARISON_CONTROLLERS[name]() for name in args.compare]

    summaries, all_episodes = [], []
    scenario_name = args.scenario
    for policy in policies:
        env = make_env(args.scenario, configs_dir=args.configs_dir)
        scenario_name = env.config.name
        episodes = run_episodes(
            env,
            policy,
            seeds,
            record_first=args.record,
            record_dir=args.episodes_dir,
        )
        all_episodes.extend(episodes)
        summaries.append(summarise(episodes))

    print(f"\nScenario: {scenario_name}   episodes per controller: {len(seeds)}\n")
    print(comparison_table(summaries))

    payload = build_results_payload(
        scenario_name,
        seeds,
        summaries,
        all_episodes,
        extra={"model": str(args.model), "deterministic": not args.stochastic},
    )
    json_path = save_json(payload, Path(args.results_dir) / f"{scenario_name}_ppo_vs_baselines.json")
    md_path = write_markdown_report(
        Path(args.results_dir) / f"{scenario_name}_ppo_vs_baselines.md",
        scenario_name,
        summaries,
        seeds,
        notes=[
            f"PPO model: {args.model} (deterministic={not args.stochastic}).",
            "All controllers ran on identical episode seeds.",
            "The classical controllers see the full map; PPO sees only its observation vector.",
        ],
    )
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
