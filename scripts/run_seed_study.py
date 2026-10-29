"""Repeat every PPO training run with additional seeds - the multi-seed study.

    python scripts/run_seed_study.py                 # seeds 1 and 2 for every scenario
    python scripts/run_seed_study.py --seeds 1 2 3   # more seeds
    python scripts/run_seed_study.py --summary-only  # rebuild the summary table only

Every published result was trained once, with training seed 0. This trains
the same scenario with the same budget and the same hyper-parameters (the
command is identical to scripts/run_experiments.py apart from ``--seed`` and
``--tag``), then evaluates each new policy on the same 30 episode seeds
(1000-1029) as the published numbers.

It never touches the published evidence:

* new models are tagged ``<tag>_seed<N>`` (``ppo_battery_long_s<N>`` for the
  battery run, matching its existing ``_s0`` name);
* evaluation writes to ``data/seeds/<tag>/`` via ``--results-dir`` and
  ``--episodes-dir``, never to ``data/results/`` or ``data/episodes/``;
* before and after the run it hashes ``data/results`` and ``data/episodes`` and
  stops with an error if either changed.

It is resumable: a (scenario, seed) whose result file already exists is skipped,
so an interrupted night can simply be restarted. Progress is appended to
``data/seeds/progress.log``; the cross-seed table is written to
``data/seeds/summary.md`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiments import EPISODES, FIRST_SEED, TRAINED_SCENARIOS  # noqa: E402

OUT = ROOT / "data" / "seeds"
PROTECTED = [ROOT / "data" / "results", ROOT / "data" / "episodes"]


def tag_for(base_tag: str, seed: int) -> str:
    if base_tag.endswith("_s0"):
        return f"{base_tag[:-3]}_s{seed}"
    return f"{base_tag}_seed{seed}"


def fingerprint() -> str:
    digest = hashlib.sha256()
    for directory in PROTECTED:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (OUT / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str]) -> None:
    started = time.time()
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"exit code {result.returncode}: {' '.join(command[1:])}")
    log(f"    done in {time.time() - started:.0f}s")


def ppo_summary(result_file: Path) -> dict:
    data = json.loads(result_file.read_text(encoding="utf-8"))
    return next(s for s in data["summaries"] if s["controller"] == "ppo")


def mean_of(entry):
    return entry.get("mean") if isinstance(entry, dict) else entry


def build_summary(seeds: list[int]) -> None:
    rows = []
    for scenario, (base_tag, timesteps) in TRAINED_SCENARIOS.items():
        per_seed = []
        published = ROOT / "data" / "results" / f"{scenario}_ppo_vs_baselines.json"
        if published.exists():
            per_seed.append((0, ppo_summary(published)))
        for seed in seeds:
            f = OUT / tag_for(base_tag, seed) / f"{scenario}_ppo_vs_baselines.json"
            if f.exists():
                per_seed.append((seed, ppo_summary(f)))
        if not per_seed:
            continue
        successes = [s["n_successful"] for _, s in per_seed]
        steps = [mean_of(s.get("steps")) for _, s in per_seed]
        rows.append({
            "scenario": scenario,
            "timesteps": timesteps,
            "seeds": [seed for seed, _ in per_seed],
            "successes_out_of_30": successes,
            "mean_steps": steps,
            "collisions": [mean_of(s.get("collisions")) for _, s in per_seed],
            "charging_steps": [mean_of(s.get("charging_events")) for _, s in per_seed],
            "reward": [mean_of(s.get("total_reward")) for _, s in per_seed],
            "success_mean": round(statistics.fmean(successes), 2),
            "success_min": min(successes),
            "success_max": max(successes),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    def fmt(values, digits=2):
        return " / ".join("n/a" if v is None else f"{v:.{digits}f}" for v in values)

    lines = [
        "# PPO across training seeds",
        "",
        "Seed 0 is the published policy (`data/results/`). Every other seed is the",
        "same training command with only `--seed` changed, evaluated on the same 30",
        "episode seeds (1000-1029). Generated by `scripts/run_seed_study.py`.",
        "",
        "| Scenario | Training steps | Seeds | Success /30 | Mean steps (successful) | Collisions | Charging steps | Reward |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['scenario']}` | {r['timesteps']:,} | {', '.join(map(str, r['seeds']))} | "
            f"{' / '.join(map(str, r['successes_out_of_30']))} | {fmt(r['mean_steps'])} | "
            f"{fmt(r['collisions'])} | {fmt(r['charging_steps'])} | {fmt(r['reward'])} |"
        )
    lines += ["", "Values are listed in seed order. n/a = no successful episode to average over.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"summary written: {len(rows)} scenarios -> data/seeds/summary.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--scenarios", nargs="+", default=list(TRAINED_SCENARIOS))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    if 0 in args.seeds:
        parser.error("seed 0 is the published run; choose other seeds")

    if args.summary_only:
        build_summary(args.seeds)
        return 0

    before = fingerprint()
    python = sys.executable
    plan = [(sc, sd) for sd in args.seeds for sc in args.scenarios]
    log(f"seed study starting: {len(plan)} runs, seeds {args.seeds}")
    failures = []
    for index, (scenario, seed) in enumerate(plan, start=1):
        base_tag, timesteps = TRAINED_SCENARIOS[scenario]
        tag = tag_for(base_tag, seed)
        out_dir = OUT / tag
        result_file = out_dir / f"{scenario}_ppo_vs_baselines.json"
        if result_file.exists():
            log(f"[{index}/{len(plan)}] {tag}: already done, skipping")
            continue
        try:
            model = ROOT / "rl_agent" / "models" / f"{tag}.zip"
            if model.exists():
                log(f"[{index}/{len(plan)}] {tag}: model exists, skipping training")
            else:
                log(f"[{index}/{len(plan)}] {tag}: training {scenario} for {timesteps:,} steps (seed {seed})")
                run([python, "-m", "rl_agent.train", "--scenario", scenario,
                     "--timesteps", str(timesteps), "--seed", str(seed), "--tag", tag])
            log(f"[{index}/{len(plan)}] {tag}: evaluating on seeds {FIRST_SEED}-{FIRST_SEED + EPISODES - 1}")
            out_dir.mkdir(parents=True, exist_ok=True)
            run([python, "-m", "rl_agent.evaluate", "--model", str(model.relative_to(ROOT)),
                 "--scenario", scenario, "--episodes", str(EPISODES), "--seed", str(FIRST_SEED),
                 "--compare", "--record", "0",
                 "--results-dir", str(out_dir.relative_to(ROOT)),
                 "--episodes-dir", str(out_dir.relative_to(ROOT))])
            s = ppo_summary(result_file)
            log(f"[{index}/{len(plan)}] {tag}: PPO {s['n_successful']}/{s['n_episodes']} successful")
        except Exception as exc:  # keep going: one failed run must not cost the night
            failures.append(f"{tag}: {exc}")
            log(f"[{index}/{len(plan)}] {tag}: FAILED - {exc}")
        if fingerprint() != before:
            log("STOP: data/results or data/episodes changed - published evidence must never be touched")
            return 2

    build_summary(args.seeds)
    log(f"seed study finished: {len(plan) - len(failures)}/{len(plan)} runs ok")
    for f in failures:
        log(f"  failed: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
