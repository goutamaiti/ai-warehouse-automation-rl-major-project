"""Measured studies of the multi-robot Live floor, on seeds kept out of tuning.

    python scripts/run_fleet_studies.py                 # all of them, ~15 min
    python scripts/run_fleet_studies.py reliability     # or just one

Writes one JSON file per study to data/fleet/ and prints the same table. Nothing
else under data/ is touched, and no trained policy is involved: the Live floor
runs the classical controller.

reliability  Does the 46x30 floor deliver everything with nobody dead? Three
             battery rules, 5/10/20 robots, a burst of 40 or 200 products or a
             steady flow of 200, seeds 2000-2009. The traffic rules were tuned on
             seeds 0-4 and 1000-1009, so these seeds are their first real test.
one_way      One-way lanes against two-way, 12 and 20 robots, 200-product
             backlog, 400 ticks, seeds 1000-1004.
priority     Five late products marked urgent against normal on the 21x15 floor,
             4 robots, backlog of 20, seeds 0-4.
home_floor   One whole run recorded for the website's home page, which has no
             backend to run one: 10 robots, 40 products, the first held-out seed.
"""
from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.simulation_service import load_scenario_config  # noqa: E402
from backend.services.fleet_session import FleetSession  # noqa: E402
from baselines.fleet_controller import FleetPlannerController  # noqa: E402
from simulation.dispatch import TaskQueue  # noqa: E402
from simulation.fleet import FleetSimulation  # noqa: E402
from simulation.rmfs import build_rmfs_layout  # noqa: E402

OUT_DIR = ROOT / "data" / "fleet"
MAX_TICKS = 6000
RULES = ("battery_tuned", "battery_large_floor", "default")
FLEETS = (5, 10, 20)
HELD_OUT_SEEDS = tuple(range(2000, 2010))


def steady_gap(n_robots: int) -> int:
    """Ticks between products at about 80% of what this fleet can deliver.

    Measured on seeds 0-4: a robot delivers about one product per 90 ticks.
    """
    return round(112 / n_robots)


def _random_product(layout, rng) -> tuple:
    pickup = layout.storage_points[int(rng.integers(len(layout.storage_points)))]
    dropoff = layout.packing_stations[int(rng.integers(len(layout.packing_stations)))]
    return pickup, dropoff


def _reliability_run(job: tuple) -> dict:
    rules, n_robots, pattern, n_products, seed = job
    layout, directions = build_rmfs_layout()
    queue = TaskQueue()
    sim = FleetSimulation(
        load_scenario_config(rules),
        n_robots=n_robots,
        layout=layout,
        task_queue=queue,
        directions=directions,
    )
    sim.reset(seed=seed)
    controller = FleetPlannerController(directions=directions)
    controller.reset(sim)
    rng = np.random.default_rng(seed)
    gap = 0 if pattern == "burst" else steady_gap(n_robots)
    arrivals = [i * gap for i in range(n_products)]

    added = tick = stalled_ticks = 0
    while tick < MAX_TICKS and not sim.done:
        while added < n_products and arrivals[added] <= tick:
            queue.add(*_random_product(layout, rng), step=tick)
            added += 1
        sim.step(controller.act(sim))
        tick += 1
        counters = sim.fleet_counters()
        stalled_ticks += bool(counters["stalled_robots"])
        if counters["tasks_delivered"] >= n_products:
            break

    counters = sim.fleet_counters()
    return {
        "rules": rules,
        "robots": n_robots,
        "pattern": pattern,
        "products": n_products,
        "seed": seed,
        "delivered": counters["tasks_delivered"],
        "robots_dead": n_robots - counters["robots_alive"],
        "stopped": sim.done,
        "ticks": tick,
        "ticks_with_a_stall_warning": stalled_ticks,
    }


def reliability() -> dict:
    jobs = [
        (rules, n_robots, pattern, n_products, seed)
        for rules in RULES
        for n_robots in FLEETS
        for pattern, n_products in (("burst", 40), ("burst", 200), ("steady", 200))
        for seed in HELD_OUT_SEEDS
    ]
    with ProcessPoolExecutor(max_workers=6) as pool:
        runs = list(pool.map(_reliability_run, jobs))

    rows = []
    print(f"{'rules':20s} {'robots':>6s} {'load':>11s} {'all delivered':>14s} {'mean delivered':>15s} "
          f"{'dead robots':>12s} {'stopped':>8s} {'stall ticks':>12s} {'median ticks':>13s}")
    for rules in RULES:
        for n_robots in FLEETS:
            for pattern, n_products in (("burst", 40), ("burst", 200), ("steady", 200)):
                group = [
                    r for r in runs
                    if (r["rules"], r["robots"], r["pattern"], r["products"])
                    == (rules, n_robots, pattern, n_products)
                ]
                row = {
                    "rules": rules,
                    "robots": n_robots,
                    "pattern": pattern,
                    "products": n_products,
                    "runs": len(group),
                    "all_delivered": sum(r["delivered"] == n_products for r in group),
                    "mean_delivered": statistics.mean(r["delivered"] for r in group),
                    "robots_dead": sum(r["robots_dead"] for r in group),
                    "stopped": sum(r["stopped"] for r in group),
                    "ticks_with_a_stall_warning": sum(r["ticks_with_a_stall_warning"] for r in group),
                    "median_ticks": statistics.median(r["ticks"] for r in group),
                }
                rows.append(row)
                print(f"{rules:20s} {n_robots:6d} {pattern + ' ' + str(n_products):>11s} "
                      f"{row['all_delivered']:>11d}/{row['runs']:<2d} {row['mean_delivered']:>15.1f} "
                      f"{row['robots_dead']:>6d}/{n_robots * row['runs']:<5d} {row['stopped']:>8d} "
                      f"{row['ticks_with_a_stall_warning']:>12d} {row['median_ticks']:>13.0f}")
    return {"max_ticks": MAX_TICKS, "seeds": list(HELD_OUT_SEEDS), "summary": rows, "runs": runs}


def _one_way_run(job: tuple) -> dict:
    lanes, n_robots, seed = job
    layout, direction = build_rmfs_layout()
    directions = direction if lanes == "one-way" else None
    queue = TaskQueue()
    sim = FleetSimulation(
        load_scenario_config("default"),
        n_robots=n_robots,
        layout=layout,
        task_queue=queue,
        directions=directions,
    )
    sim.reset(seed=seed)
    controller = FleetPlannerController(directions=directions)
    controller.reset(sim)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        queue.add(*_random_product(layout, rng), step=0)
    for _ in range(400):
        if sim.done:
            break
        sim.step(controller.act(sim))
    counters = sim.fleet_counters()
    return {
        "lanes": lanes,
        "robots": n_robots,
        "seed": seed,
        "refusals": counters["blocked_by_robot"],
        "delivered": counters["tasks_delivered"],
        "moves": counters["moves"],
        "robots_alive": counters["robots_alive"],
    }


def one_way() -> dict:
    fleets = (12, 20)
    jobs = [
        (lanes, n_robots, seed)
        for lanes in ("one-way", "two-way")
        for n_robots in fleets
        for seed in range(1000, 1005)
    ]
    with ProcessPoolExecutor(max_workers=6) as pool:
        runs = list(pool.map(_one_way_run, jobs))
    summary = {}
    for n_robots in fleets:
        for lanes in ("one-way", "two-way"):
            group = [r for r in runs if (r["lanes"], r["robots"]) == (lanes, n_robots)]
            row = {
                "mean_refusals": statistics.mean(r["refusals"] for r in group),
                "mean_delivered": statistics.mean(r["delivered"] for r in group),
                "mean_moves": statistics.mean(r["moves"] for r in group),
                "robots_lost": sum(n_robots - r["robots_alive"] for r in group),
            }
            summary[f"{lanes}, {n_robots} robots"] = row
            print(f"{lanes:8s} {n_robots:2d} robots {row}  refusals {[r['refusals'] for r in group]}")
    return {"summary": summary, "runs": runs}


def _priority_run(job: tuple) -> dict:
    late_priority, seed = job
    config = load_scenario_config("default")
    queue = TaskQueue()
    sim = FleetSimulation(config, n_robots=4, task_queue=queue)
    sim.reset(seed=seed)
    controller = FleetPlannerController()
    controller.reset(sim)
    rng = np.random.default_rng(seed)
    layout = sim.layout
    backlog = [queue.add(*_random_product(layout, rng), step=0) for _ in range(20)]
    late = []
    for tick in range(4000):
        if tick == 10:
            late = [
                queue.add(*_random_product(layout, rng), step=tick, priority=late_priority)
                for _ in range(5)
            ]
        sim.step(controller.act(sim))
        if len(late) == 5 and all(r.delivered_step is not None for r in backlog + late):
            break
    return {
        "late_priority": late_priority,
        "seed": seed,
        "delivered": sum(r.delivered_step is not None for r in backlog + late),
        "late_turnaround": statistics.mean(r.delivered_step - r.created_step for r in late),
        "backlog_turnaround": statistics.mean(r.delivered_step - r.created_step for r in backlog),
    }


def priority() -> dict:
    jobs = [(late, seed) for late in ("urgent", "normal") for seed in range(5)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        runs = list(pool.map(_priority_run, jobs))
    summary = {}
    for late in ("urgent", "normal"):
        group = [r for r in runs if r["late_priority"] == late]
        summary[late] = {
            "late_turnaround": statistics.mean(r["late_turnaround"] for r in group),
            "backlog_turnaround": statistics.mean(r["backlog_turnaround"] for r in group),
            "delivered": sum(r["delivered"] for r in group),
        }
        print(f"late marked {late}: {summary[late]}")
    return {"summary": summary, "runs": runs}


def home_floor() -> dict:
    robots, products, seed = 10, 40, HELD_OUT_SEEDS[0]
    session = FleetSession(
        load_scenario_config("battery_large_floor"), n_robots=robots, seed=seed, floor="rmfs"
    )
    rng = np.random.default_rng(seed)
    for _ in range(products):
        session.queue.add(*_random_product(session.sim.layout, rng), step=0)

    def compact(frame: dict) -> dict:
        carrying = {r["id"] for r in frame["robots"] if r["carrying"]}
        return {
            "robots": [
                [*r["position"], int(r["carrying"]), round(r["battery"]), int(r["alive"])]
                for r in frame["robots"]
            ],
            "waiting": [
                q["pickup"]
                for q in frame["requests"]
                if q["status"] == "waiting"
                or (q["status"] == "assigned" and q["robot_id"] not in carrying)
            ],
            "delivered": frame["fleet"]["tasks_delivered"],
        }

    frames = [compact(session.state())]
    while frames[-1]["delivered"] < products and session.ticks < MAX_TICKS:
        frames.append(compact(session.tick()))
        if session.sim.done:
            break

    layout = session.layout()
    print(f"{frames[-1]['delivered']} of {products} delivered in {session.ticks} ticks")
    return {
        "rules": "battery_large_floor",
        "seed": seed,
        "products": products,
        "layout": {
            key: layout[key] for key in ("grid", "width", "height", "directions")
        },
        "frames": frames,
        # Full-charge range of the two battery plans the home page compares.
        "moves_on_full_charge": {
            rules: round(battery.capacity / battery.move_cost)
            for rules, battery in (
                (name, load_scenario_config(name).battery) for name in RULES
            )
        },
    }


STUDIES = {
    "reliability": reliability,
    "one_way": one_way,
    "priority": priority,
    "home_floor": home_floor,
}


def main() -> None:
    chosen = sys.argv[1:] or list(STUDIES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        print(f"\n--- {name} ---")
        result = STUDIES[name]()
        # The recording ships with the website, so it is kept small.
        text = (
            json.dumps(result, separators=(",", ":"))
            if name == "home_floor"
            else json.dumps(result, indent=2)
        )
        (OUT_DIR / f"{name}.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
