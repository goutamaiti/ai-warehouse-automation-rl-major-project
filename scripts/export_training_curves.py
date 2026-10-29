"""Export PPO learning curves from the training logs into committed JSON.

    python scripts/export_training_curves.py

``data/logs/`` is git-ignored (it holds TensorBoard files and per-environment
monitor CSVs), so neither GitHub nor the deployed dashboard ever sees it. This
script reads the one file in each run that is a learning curve in the usual
sense - Stable-Baselines3's ``eval/evaluations.npz``, written by the
EvalCallback every ``eval_freq`` steps from ``n_eval_episodes`` deterministic
episodes on a separate evaluation environment - and writes a small summary per
run to ``data/training/<tag>.json``.

Nothing is smoothed, resampled or interpolated: each point is one evaluation
that actually happened, with the mean and standard deviation of its episodes.
Runs whose logs were not retained are simply absent; the dashboard says so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "logs"
OUT = ROOT / "data" / "training"


def export(run_dir: Path) -> dict | None:
    evaluations = run_dir / "eval" / "evaluations.npz"
    metadata_file = run_dir / "run_metadata.json"
    if not evaluations.exists() or not metadata_file.exists():
        return None
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    config = metadata.get("config", {})
    data = np.load(evaluations)
    timesteps = data["timesteps"]
    rewards = data["results"]
    lengths = data["ep_lengths"]
    successes = data["successes"] if "successes" in data.files else None

    points = []
    for i, step in enumerate(timesteps):
        point = {
            "timesteps": int(step),
            "reward_mean": round(float(rewards[i].mean()), 3),
            "reward_std": round(float(rewards[i].std()), 3),
            "length_mean": round(float(lengths[i].mean()), 2),
        }
        if successes is not None:
            point["success_rate"] = round(float(successes[i].mean()), 4)
        points.append(point)

    return {
        "tag": metadata.get("tag", run_dir.name),
        "scenario": config.get("scenario"),
        "total_timesteps": config.get("total_timesteps"),
        "eval_freq": config.get("eval_freq"),
        "n_eval_episodes": int(rewards.shape[1]),
        "training_seed": config.get("seed"),
        "wall_clock_seconds": metadata.get("wall_clock_seconds"),
        "source": f"data/logs/{run_dir.name}/eval/evaluations.npz",
        "points": points,
    }


def main() -> int:
    if not LOGS.exists():
        print(f"no training logs at {LOGS}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for run_dir in sorted(p for p in LOGS.iterdir() if p.is_dir()):
        summary = export(run_dir)
        if summary is None:
            print(f"  skip {run_dir.name}: no evaluations.npz / run_metadata.json")
            continue
        target = OUT / f"{summary['tag']}.json"
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        first, last = summary["points"][0], summary["points"][-1]
        print(
            f"  {summary['tag']:22s} {summary['scenario']:16s} {len(summary['points']):3d} evals  "
            f"success {first.get('success_rate', 'n/a')} -> {last.get('success_rate', 'n/a')}  "
            f"reward {first['reward_mean']} -> {last['reward_mean']}"
        )
        written += 1
    print(f"wrote {written} curve(s) to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
