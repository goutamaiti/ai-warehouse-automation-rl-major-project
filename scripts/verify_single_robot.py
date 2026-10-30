"""Single-robot proof. Writes nothing.

Usage:  python scripts/verify_single_robot.py    (a few minutes; exit code 0 = all pass)

Part 1 - the live demo path. POST /api/run for every scenario x controller x
         3 seeds: success, and at the pickup / delivery events the robot is
         physically on the pickup / drop-off cell.
Part 2 - the live demo loads the SAME PPO model the published results used.
Part 3 - every committed episode in data/results (540) re-run in memory with the
         current code: same steps, deliveries, outcome and reward.
"""
import sys
from pathlib import Path

# Run from anywhere: put the repository root on the import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)  # data/ and configs/ are read by relative path


import json
import glob
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.simulation_service import ppo_model_path, MODELS_DIR
from baselines.controller import PlannerPolicy
from environment import make_env
from analytics.runner import run_episode
from rl_agent.evaluate import PPOPolicy

client = TestClient(app, raise_server_exceptions=False)
SCENARIOS = ["simple_static", "default", "complex_static", "dynamic_obstacles",
             "battery_constrained", "battery_tuned"]
problems = []

print("=== PART 1: live /api/run, robot reaches pickup and drop-off ===")
for scenario in SCENARIOS:
    row = []
    for controller in ("astar", "bfs", "ppo"):
        ok_runs = 0
        reached = 0
        for seed in (1000, 1001, 1002):
            r = client.post("/api/run", json={"scenario": scenario, "controller": controller, "seed": seed})
            if r.status_code != 200:
                problems.append(f"{scenario}/{controller}/{seed}: HTTP {r.status_code} {r.text[:100]}")
                continue
            rep = r.json()
            frames = rep["frames"]
            success = rep["summary"]["termination_reason"] == "all_tasks_delivered"
            ok_runs += success
            events_ok = True
            for f in frames:
                ev = f.get("events") or []
                pos = tuple(f["robot"]["position"])
                task = f.get("task") or {}
                if "pickup" in ev and pos != tuple(task.get("pickup", ())):
                    events_ok = False
                    problems.append(f"{scenario}/{controller}/{seed} step {f['step']}: pickup recorded at {pos}, task pickup {task.get('pickup')}")
                if "delivery" in ev and pos != tuple(task.get("dropoff", ())):
                    events_ok = False
                    problems.append(f"{scenario}/{controller}/{seed} step {f['step']}: delivery recorded at {pos}, task drop-off {task.get('dropoff')}")
            n_pick = sum("pickup" in (f.get("events") or []) for f in frames)
            n_del = sum("delivery" in (f.get("events") or []) for f in frames)
            if success and (n_pick < 1 or n_del < 1):
                events_ok = False
                problems.append(f"{scenario}/{controller}/{seed}: success with {n_pick} pickups / {n_del} deliveries recorded")
            reached += events_ok
        row.append(f"{controller}: {ok_runs}/3 succeeded, cells verified {reached}/3")
    print(f"  {scenario:20s} " + " | ".join(row))

print("\n=== PART 2: live demo PPO == published-results PPO ===")
for p in sorted(glob.glob("data/results/*_ppo_vs_baselines.json")):
    d = json.load(open(p))
    published = Path(d["model"]).name
    live = ppo_model_path(d["scenario"])
    live_name = live.name if live else None
    same = live_name == published
    print(f"  {d['scenario']:20s} published: {published:28s} live demo loads: {live_name}  {'OK' if same else '<-- DIFFERENT'}")
    if not same:
        problems.append(f"{d['scenario']}: demo loads {live_name} but results were measured with {published}")

print("\n=== PART 3: every committed episode reproduces with today's code ===")
FIELDS = ("steps", "tasks_delivered", "tasks_failed", "termination_reason", "collisions", "charging_events")
grand = 0
grand_bad = 0
for p in sorted(glob.glob("data/results/*.json")):
    d = json.load(open(p))
    scenario = d["scenario"]
    env = make_env(scenario)
    policies = {"astar": PlannerPolicy(planner="astar"), "bfs": PlannerPolicy(planner="bfs"),
                "ppo": PPOPolicy.load(d["model"], deterministic=d.get("deterministic", True))}
    bad = 0
    for ep in d["episodes"]:
        pol = policies.get(ep["controller"])
        if pol is None:
            continue
        m, _ = run_episode(env, pol, seed=ep["seed"])
        got = m.to_dict()
        diff = [f for f in FIELDS if got[f] != ep[f]]
        if abs(got["total_reward"] - ep["total_reward"]) > 1e-2:
            diff.append(f"reward {got['total_reward']} vs {ep['total_reward']}")
        grand += 1
        if diff:
            bad += 1
            grand_bad += 1
            if bad <= 3:
                problems.append(f"{scenario}/{ep['controller']}/seed{ep['seed']} differs: {diff}")
    print(f"  {scenario:20s} {len(d['episodes'])} episodes re-run, {len(d['episodes']) - bad} identical")
print(f"  TOTAL: {grand - grand_bad}/{grand} committed episodes reproduce exactly")

print(f"\n=== {len(problems)} PROBLEM(S) ===")
for p in problems[:25]:
    print("  *", p)
sys.exit(0 if not problems else 1)
