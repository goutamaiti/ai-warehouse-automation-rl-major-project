"""Two PPO studies that answer "which reward terms matter" and "can training be steadier".

    python scripts/run_reward_studies.py                  # both studies
    python scripts/run_reward_studies.py --study ablation
    python scripts/run_reward_studies.py --study battery-lr
    python scripts/run_reward_studies.py --summary-only

1. Reward-weight ablation (data/ablation/). The moving-traffic scenario is the
   one where every reward term is active, so each arm retrains it with exactly
   one term switched off (configs/ablation/no_*.yaml) for training seeds 0, 1
   and 2, and evaluates on the usual 30 episodes (seeds 1000-1029). The full
   reward is the published dynamic_obstacles policy (seed 0) plus the seed
   study (seeds 1, 2). Arms are compared on TASK metrics - success, steps,
   collisions, path efficiency - never on reward, because each arm is scored
   by a different reward function.

2. Battery training stability (data/stability/). Every battery_tuned seed
   reached 100% evaluation success at some point in training and then swung
   (seed 2: 100% -> 25% -> 85%). This retrains it with a lower learning rate
   (1e-4 instead of 3e-4; everything else identical) on seeds 0, 1 and 2, and
   evaluates both the final policy and the best checkpoint.

Never touches the published evidence: outputs go only to data/ablation/ and
data/stability/; data/results and data/episodes are fingerprinted before and
after every run and the script stops if either changes. Resumable: finished
runs are skipped.
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
PROTECTED = [ROOT / "data" / "results", ROOT / "data" / "episodes"]
SEEDS = [0, 1, 2]
EPISODES, FIRST_SEED = 30, 1000

ABLATION_ARMS = ["progress", "collision", "energy", "step", "wait"]
ABLATION_STEPS = 400_000          # identical to the published dynamic_obstacles run
BATTERY_STEPS = 2_000_000         # identical to the published battery_tuned run
BATTERY_LR = 1e-4                 # published runs used the default 3e-4


def fingerprint() -> str:
    digest = hashlib.sha256()
    for directory in PROTECTED:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def log(out: Path, message: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)
    with (out / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"exit {result.returncode}: {' '.join(cmd[1:])}")


def evaluate(model: Path, scenario: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = str(out_dir.relative_to(ROOT))
    run([sys.executable, "-m", "rl_agent.evaluate", "--model", str(model.relative_to(ROOT)),
         "--scenario", scenario, "--episodes", str(EPISODES), "--seed", str(FIRST_SEED),
         "--compare", "--record", "0", "--results-dir", rel, "--episodes-dir", rel])
    result = next(out_dir.glob("*_ppo_vs_baselines.json"))
    return ppo(result)


def ppo(result_file: Path) -> dict:
    data = json.loads(result_file.read_text(encoding="utf-8"))
    return next(s for s in data["summaries"] if s["controller"] == "ppo")


def train(scenario: str, steps: int, seed: int, tag: str, extra: list[str] | None = None) -> Path:
    model = ROOT / "rl_agent" / "models" / f"{tag}.zip"
    if not model.exists():
        run([sys.executable, "-m", "rl_agent.train", "--scenario", scenario,
             "--timesteps", str(steps), "--seed", str(seed), "--tag", tag, *(extra or [])])
    return model


def m(entry):
    return entry.get("mean") if isinstance(entry, dict) else entry


# ----------------------------------------------------------------------------
def study_ablation(before: str) -> int:
    out = ROOT / "data" / "ablation"
    failures = 0
    for arm in ABLATION_ARMS:
        for seed in SEEDS:
            tag = f"ppo_ablation_no_{arm}_seed{seed}"
            result_dir = out / tag
            if any(result_dir.glob("*_ppo_vs_baselines.json")):
                log(out, f"{tag}: done, skipping")
                continue
            try:
                started = time.time()
                log(out, f"{tag}: training {ABLATION_STEPS:,} steps")
                model = train(f"configs/ablation/no_{arm}.yaml", ABLATION_STEPS, seed, tag)
                s = evaluate(model, f"configs/ablation/no_{arm}.yaml", result_dir)
                log(out, f"{tag}: {s['n_successful']}/30, {time.time() - started:.0f}s")
            except Exception as exc:  # one failed arm must not cost the rest
                failures += 1
                log(out, f"{tag}: FAILED - {exc}")
            if fingerprint() != before:
                log(out, "STOP: published data changed")
                raise SystemExit(2)
    return failures


def study_battery_lr(before: str) -> int:
    out = ROOT / "data" / "stability"
    failures = 0
    for seed in SEEDS:
        tag = f"ppo_battery_lr1e4_s{seed}"
        final_dir, best_dir = out / tag, out / f"{tag}_best"
        if any(final_dir.glob("*.json")) and any(best_dir.glob("*.json")):
            log(out, f"{tag}: done, skipping")
            continue
        try:
            started = time.time()
            log(out, f"{tag}: training {BATTERY_STEPS:,} steps at learning rate {BATTERY_LR}")
            model = train("battery_tuned", BATTERY_STEPS, seed, tag, ["--learning-rate", str(BATTERY_LR)])
            s = evaluate(model, "battery_tuned", final_dir)
            b = evaluate(ROOT / "rl_agent" / "models" / f"{tag}_best" / "best_model.zip", "battery_tuned", best_dir)
            log(out, f"{tag}: final {s['n_successful']}/30, best checkpoint {b['n_successful']}/30, {time.time() - started:.0f}s")
        except Exception as exc:
            failures += 1
            log(out, f"{tag}: FAILED - {exc}")
        if fingerprint() != before:
            log(out, "STOP: published data changed")
            raise SystemExit(2)
    return failures


# ----------------------------------------------------------------------------
def fmt(values, d=2):
    return " / ".join("n/a" if v is None else f"{v:.{d}f}" for v in values)


def summary_ablation() -> None:
    out = ROOT / "data" / "ablation"
    baseline = [ROOT / "data" / "results" / "dynamic_obstacles_ppo_vs_baselines.json"] + [
        ROOT / "data" / "seeds" / f"ppo_dynamic_seed{s}" / "dynamic_obstacles_ppo_vs_baselines.json" for s in (1, 2)]
    rows = [("full reward (published + seed study)", [ppo(p) for p in baseline if p.exists()])]
    for arm in ABLATION_ARMS:
        found = []
        for seed in SEEDS:
            hit = list((out / f"ppo_ablation_no_{arm}_seed{seed}").glob("*_ppo_vs_baselines.json"))
            if hit:
                found.append(ppo(hit[0]))
        rows.append((f"without `{arm}`", found))
    table = []
    lines = [
        "# Reward-weight ablation (dynamic_obstacles)",
        "",
        "Each arm retrains PPO with ONE reward term switched off (`configs/ablation/`),",
        "same world, budget (400k steps) and hyper-parameters, training seeds 0/1/2,",
        "evaluated on episodes 1000-1029. Values are per seed. Reward is not compared",
        "across arms - each arm is scored by a different reward function.",
        "",
        "| Arm | Seeds | Success /30 | Mean steps | Path efficiency | Collisions / episode |",
        "|---|---|---|---|---|---|",
    ]
    for name, runs in rows:
        if not runs:
            continue
        succ = [r["n_successful"] for r in runs]
        table.append({"arm": name, "success": succ, "steps": [m(r.get("steps")) for r in runs],
                      "path_efficiency": [m(r.get("path_efficiency")) for r in runs],
                      "collisions": [m(r.get("collisions")) for r in runs]})
        lines.append(f"| {name} | {len(runs)} | {' / '.join(map(str, succ))} | {fmt(table[-1]['steps'])} | "
                     f"{fmt(table[-1]['path_efficiency'])} | {fmt(table[-1]['collisions'])} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_battery() -> None:
    out = ROOT / "data" / "stability"
    def collect(paths):
        return [ppo(p) for p in paths if p.exists()]
    arms = [
        ("lr 3e-4, final policy (published + seed study)",
         [ROOT / "data" / "results" / "battery_tuned_ppo_vs_baselines.json"] +
         [ROOT / "data" / "seeds" / f"ppo_battery_long_s{s}" / "battery_tuned_ppo_vs_baselines.json" for s in (1, 2)]),
        ("lr 3e-4, best checkpoint",
         [ROOT / "data" / "seeds" / f"ppo_battery_long_s{s}_best" / "battery_tuned_ppo_vs_baselines.json" for s in SEEDS]),
        ("lr 1e-4, final policy",
         [out / f"ppo_battery_lr1e4_s{s}" / "battery_tuned_ppo_vs_baselines.json" for s in SEEDS]),
        ("lr 1e-4, best checkpoint",
         [out / f"ppo_battery_lr1e4_s{s}_best" / "battery_tuned_ppo_vs_baselines.json" for s in SEEDS]),
    ]
    lines = [
        "# Battery policy reliability (battery_tuned)",
        "",
        "Same scenario, reward and 2M-step budget throughout; training seeds 0/1/2;",
        "evaluated on episodes 1000-1029. 'Best checkpoint' is the checkpoint with the",
        "highest mean reward on the training run's own evaluation environment, whose",
        "episodes are drawn at random and are not the 1000-1029 test episodes.",
        "",
        "| Setting | Seeds | Success /30 | Total | Mean steps | Charging steps | Outcomes |",
        "|---|---|---|---|---|---|---|",
    ]
    table = []
    for name, paths in arms:
        runs = collect(paths)
        if not runs:
            continue
        succ = [r["n_successful"] for r in runs]
        outcomes = " ; ".join(", ".join(f"{k} {v}" for k, v in r["termination_reasons"].items()) for r in runs)
        table.append({"setting": name, "success": succ, "total": sum(succ), "of": 30 * len(runs)})
        lines.append(f"| {name} | {len(runs)} | {' / '.join(map(str, succ))} | {sum(succ)}/{30 * len(runs)} | "
                     f"{fmt([m(r.get('steps')) for r in runs])} | {fmt([m(r.get('charging_events')) for r in runs])} | {outcomes} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=["ablation", "battery-lr", "both"], default="both")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    failures = 0
    if not args.summary_only:
        before = fingerprint()
        if args.study in ("ablation", "both"):
            failures += study_ablation(before)
        if args.study in ("battery-lr", "both"):
            failures += study_battery_lr(before)
    summary_ablation()
    summary_battery()
    print("summaries written: data/ablation/summary.md, data/stability/summary.md")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
