"""End-to-end proof that the Live floor does what the demo claims.

Usage:  python scripts/verify_live_floor.py      (about 1 minute; exit code 0 = all pass)

Drives the same path the browser does: POST /api/fleet/start, POST /api/fleet/task,
then session.tick() - the exact call the SSE stream makes once per tick. For every
product it proves, from observed positions and an INDEPENDENT BFS (not the fleet's
own distance code):

  1. it was delivered
  2. it went to the nearest idle robot at the moment of dispatch (ties -> lower id)
  3. that robot physically stood on the pickup cell, and later on the drop-off cell
     at the tick the delivery was recorded

and for the whole run: no two robots ever share a cell, no robot is ever on a
blocked cell, and the floor never false-deadlocks. Writes nothing.
"""
import sys
from pathlib import Path

# Run from anywhere: put the repository root on the import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)  # data/ and configs/ are read by relative path


from collections import deque

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
import backend.services.fleet_session as fs

client = TestClient(app, raise_server_exceptions=False)


def bfs_from(walkable, source):
    h, w = walkable.shape
    dist = np.full((h, w), -1, dtype=int)
    dist[source] = 0
    q = deque([source])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def run(name, scenario, floor, n_robots, n_products, max_ticks, schedule, seed=0):
    """schedule(i, rng) -> how many ticks to wait before adding product i."""
    r = client.post("/api/fleet/start", json={"scenario": scenario, "n_robots": n_robots,
                                              "controller": "astar", "seed": seed, "floor": floor})
    assert r.status_code == 200, r.text
    session = fs.current()
    sim = session.sim
    layout = sim.layout
    walkable = layout.walkable_mask()
    rng = np.random.default_rng(seed + 7)

    problems = []
    trace = {}            # request id -> dict
    positions_by_tick = []  # list of {rid: pos}
    pending_adds = []
    t_next = 0
    for i in range(n_products):
        t_next += schedule(i, rng)
        pending_adds.append(t_next)

    added = 0
    stopped_reason = None
    for tick in range(max_ticks):
        # add any products due now, through the HTTP endpoint the card uses
        while added < n_products and pending_adds[added] <= tick:
            pick = layout.storage_points[int(rng.integers(len(layout.storage_points)))]
            drop = layout.packing_stations[int(rng.integers(len(layout.packing_stations)))]
            resp = client.post("/api/fleet/task", json={"pickup": list(pick), "dropoff": list(drop)})
            if resp.status_code != 200:
                problems.append(f"add product {added} rejected: {resp.status_code} {resp.text[:120]}")
            else:
                rid = resp.json()["request"]["id"]
                trace[rid] = {"pickup": tuple(pick), "dropoff": tuple(drop), "added_tick": tick,
                              "robot": None, "stood_on_pickup": False, "on_dropoff_at_delivery": None}
            added += 1

        # who is idle BEFORE this tick (dispatch happens at the start of step)
        # Eligible for work: alive, no task, and above the low-battery threshold
        # (a robot below it is sent to charge instead of being given work).
        low = sim.config.battery.low_threshold
        idle_before = {rid: m.position for rid, m in sim.members.items()
                       if m.alive and m.task is None and m.robot.battery > low}

        frame = session.tick()
        if frame.get("stopped"):
            stopped_reason = frame.get("stopped_reason")
            break

        pos_now = {rid: m.position for rid, m in sim.members.items() if m.alive}
        positions_by_tick.append(pos_now)

        # --- whole-floor invariants ---------------------------------------
        cells = list(pos_now.values())
        if len(set(cells)) != len(cells):
            problems.append(f"tick {tick}: two robots on one cell {sorted(cells)}")
        for rid, p in pos_now.items():
            if not walkable[p]:
                problems.append(f"tick {tick}: robot {rid} on blocked cell {p}")

        # --- nearest-robot check for every dispatch this tick --------------
        # Requests dispatched on the same tick are served oldest first, each
        # from the robots NOT already taken by an earlier one this tick.
        still_free = dict(idle_before)
        for d in sim.last_dispatch:
            req = trace.get(d.request_id)
            if req is None:
                still_free.pop(d.robot_id, None)
                continue
            req["robot"] = d.robot_id
            field = bfs_from(walkable, req["pickup"])
            options = sorted((int(field[p]), rid) for rid, p in still_free.items() if field[p] >= 0)
            still_free.pop(d.robot_id, None)
            if not options:
                problems.append(f"request {d.request_id}: dispatched with no reachable idle robot?")
                continue
            best = options[0]
            req["dispatch_options"] = options
            if (int(field[idle_before[d.robot_id]]), d.robot_id) != best:
                problems.append(
                    f"request {d.request_id}: went to robot {d.robot_id} "
                    f"(distance {field[idle_before[d.robot_id]]}) but nearest idle was robot "
                    f"{best[1]} (distance {best[0]})")

        # --- did the assigned robot physically reach the cells? ------------
        for rid_req, req in trace.items():
            robot = req["robot"]
            if robot is None or robot not in pos_now:
                continue
            if pos_now[robot] == req["pickup"]:
                req["stood_on_pickup"] = True
        for rq in session.queue.requests.values():
            req = trace.get(rq.request_id)
            if req is not None and rq.delivered_step is not None and req["on_dropoff_at_delivery"] is None:
                req["on_dropoff_at_delivery"] = pos_now.get(rq.robot_id) == req["dropoff"]
                req["turnaround"] = rq.delivered_step - rq.created_step

        if added == n_products and not sim.has_outstanding_work():
            break

    # --- per-request verdicts -------------------------------------------------
    statuses = {rq.request_id: rq.status.value for rq in session.queue.requests.values()}
    delivered = sum(1 for s in statuses.values() if s == "delivered")
    for rid, req in trace.items():
        if statuses.get(rid) != "delivered":
            problems.append(f"request {rid} {req['pickup']}->{req['dropoff']} ended '{statuses.get(rid)}'")
            continue
        if not req["stood_on_pickup"]:
            problems.append(f"request {rid}: robot {req['robot']} never stood on pickup {req['pickup']}")
        if req["on_dropoff_at_delivery"] is not True:
            problems.append(f"request {rid}: robot {req['robot']} not on drop-off {req['dropoff']} when delivered")

    alive = sum(1 for m in sim.members.values() if m.alive)
    turn = [t.get("turnaround") for t in trace.values() if t.get("turnaround") is not None]
    charges = sum(m.counters.charging_events for m in sim.members.values())
    print(f"\n### {name}")
    print(f"    floor={floor} rules={scenario} robots={n_robots} ticks_run={len(positions_by_tick)}")
    print(f"    products delivered: {delivered}/{n_products}   robots alive at end: {alive}/{n_robots}   charging steps: {charges}")
    if turn:
        print(f"    turnaround (ticks): min {min(turn)}  mean {sum(turn)/len(turn):.1f}  max {max(turn)}")
    reached_p = sum(1 for t in trace.values() if t["stood_on_pickup"])
    reached_d = sum(1 for t in trace.values() if t["on_dropoff_at_delivery"])
    print(f"    robot physically stood on pickup: {reached_p}/{n_products}   on drop-off at delivery: {reached_d}/{n_products}")
    nearest_ok = sum(1 for t in trace.values() if t.get("dispatch_options")
                     and (t["dispatch_options"][0][1] == t["robot"]))
    print(f"    went to nearest idle robot (independent BFS): {nearest_ok}/{n_products}")
    if stopped_reason:
        problems.append(f"floor STOPPED: {stopped_reason}")
    print(f"    VERDICT: {'PASS' if not problems else 'FAIL'}")
    for p in problems[:12]:
        print(f"      - {p}")
    client.post("/api/fleet/stop")
    return not problems, problems


results = []
# A: products added one at a time with long idle gaps - the case the deadlock bug broke
results.append(run("A  default floor, idle gaps of 20-40 ticks between products",
                   "default", "default", 4, 10, 2000, lambda i, g: int(g.integers(20, 41))))
# B: a burst - more products than robots, so the queue has to hold work
results.append(run("B  default floor, burst of 12 products at once (4 robots)",
                   "default", "default", 4, 12, 2000, lambda i, g: 0))
# C: the realistic RMFS floor, 8 robots, steady arrivals
for seed in (0, 1, 2, 3):
    results.append(run(f"C  realistic RMFS floor, 8 robots, 25 products, seed {seed}",
                       "default", "rmfs", 8, 25, 3000, lambda i, g: int(g.integers(0, 6)), seed=seed))
# D: battery rules - robots must charge to keep working
results.append(run("D  battery-constrained rules, 4 robots, 10 products",
                   "battery_constrained", "default", 4, 10, 3000, lambda i, g: int(g.integers(0, 15))))
# E: moving traffic
results.append(run("E  moving-traffic rules, 4 robots, 10 products",
                   "dynamic_obstacles", "default", 4, 10, 2000, lambda i, g: int(g.integers(0, 15))))

print("\n=========== SUMMARY ===========")
print(f"{sum(ok for ok, _ in results)}/{len(results)} scenarios passed")
sys.exit(0 if all(ok for ok, _ in results) else 1)
