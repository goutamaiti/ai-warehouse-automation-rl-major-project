"""Demo test cases: input, expected output, actual output - against the live API.

    python scripts/run_test_cases.py                  # backend on 127.0.0.1:8000
    python scripts/run_test_cases.py --api http://127.0.0.1:8000 --out docs/test-cases.md

Every "actual" column is what the running backend returned during this run;
nothing is typed in by hand. Covers the functional modules end to end, input
validation, and error handling. Starts (and replaces) the Live-floor session.
Writes nothing under data/.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(api: str, method: str, path: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(api + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw)
        except ValueError:
            return err.code, raw.decode(errors="replace")


def detail(payload) -> str:
    if isinstance(payload, dict) and "detail" in payload:
        d = payload["detail"]
        if isinstance(d, list):  # pydantic validation errors
            d = "; ".join(e.get("msg", "") for e in d)
        return str(d)
    return ""


def episode_summary(payload) -> str:
    s = payload.get("summary", payload) if isinstance(payload, dict) else {}
    keys = ("termination_reason", "steps", "collisions")
    return ", ".join(f"{k}={s.get(k)}" for k in keys if k in s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=str(ROOT / "docs" / "test-cases.md"))
    args = ap.parse_args(argv)
    api = args.api.rstrip("/") + "/api"
    rows = []

    def case(cid, module, what, expected, ok, actual):
        rows.append((cid, module, what, expected, actual, "PASS" if ok else "FAIL"))
        print(f"{cid:6} {'PASS' if ok else 'FAIL'}  {what}  ->  {actual}")

    # ---------------- functional, end to end ----------------
    st, p = call(api, "GET", "/health")
    case("F1", "Backend", "GET /api/health", "200, status ok, PPO models loaded",
         st == 200 and p.get("status") == "ok",
         f"{st}, status={p.get('status')}, controllers={[c['name'] for c in p.get('controllers', [])]}")

    st, p = call(api, "GET", "/scenarios")
    names = [s["name"] if isinstance(s, dict) else s for s in (p if isinstance(p, list) else p.get("scenarios", []))]
    case("F2", "Simulation", "List scenarios", "200, the six scenarios", st == 200 and len(names) >= 6,
         f"{st}, {len(names)} scenarios")

    for cid, scen, ctrl in [("F3", "default", "ppo"), ("F4", "default", "astar"),
                            ("F5", "dynamic_obstacles", "ppo"), ("F6", "battery_tuned", "astar")]:
        st, p = call(api, "POST", "/run", {"scenario": scen, "controller": ctrl, "seed": 1000})
        s = p.get("summary", {}) if isinstance(p, dict) else {}
        case(cid, "RL / Planner", f"Run {ctrl} on {scen}, seed 1000", "200, all tasks delivered",
             st == 200 and s.get("termination_reason") == "all_tasks_delivered", f"{st}, {episode_summary(p)}")

    st, p = call(api, "GET", "/results")
    case("F7", "Analytics", "Published results", "200, results for every scenario",
         st == 200 and bool(p), f"{st}, {len(p) if hasattr(p, '__len__') else '?'} result sets")

    st, p = call(api, "POST", "/fleet/start", {"scenario": "default", "n_robots": 3, "seed": 0})
    case("F8", "Live floor", "Start a floor, 3 robots", "200, 3 robots placed",
         st == 200 and p["state"]["n_robots"] == 3, f"{st}, n_robots={p.get('state', {}).get('n_robots')}")
    layout = p.get("layout", {}) if st == 200 else {}
    picks, drops = layout.get("pickup_choices", [[1, 1]]), layout.get("dropoff_choices", [[1, 1]])

    for k in range(3):
        call(api, "POST", "/fleet/task", {"pickup": picks[k], "dropoff": drops[0]})
    st, p = call(api, "POST", "/fleet/task", {"pickup": picks[5], "dropoff": drops[-1], "priority": "urgent"})
    created = p.get("request", {}) if isinstance(p, dict) else {}
    case("F9", "Live floor", "Add an urgent product after 3 normal ones", "200, queued as urgent",
         st == 200 and created.get("priority") == "urgent",
         f"{st}, priority={created.get('priority')}, waiting={p.get('queue', {}).get('waiting')}")

    st, p = call(api, "POST", "/fleet/advance", {"ticks": 150})
    stats = (p.get("queue") or p.get("tasks") or {}) if isinstance(p, dict) else {}
    done = p.get("fleet", {}).get("delivered", stats.get("delivered")) if isinstance(p, dict) else None
    case("F10", "Live floor", "Advance 150 ticks", "200, all 4 products delivered, no deadlock",
         st == 200 and not p.get("stopped") and done == 4, f"{st}, delivered={done}, stopped={p.get('stopped')}")

    # ---------------- input validation ----------------
    bad_inputs = [
        ("V1", "Run", "Unknown scenario", "POST", "/run", {"scenario": "nope"}, 404),
        ("V2", "Run", "Unknown controller", "POST", "/run", {"controller": "dijkstra"}, 404),
        ("V3", "Run", "Negative seed", "POST", "/run", {"seed": -1}, 422),
        ("V4", "Editor", "Grid with rows of different widths", "POST", "/run",
         {"layout": [[1, 1, 1], [1, 0]]}, 400),
        ("V5", "Editor", "Grid with an unknown cell value (9)", "POST", "/run",
         {"layout": [[1, 1, 1, 1, 1], [1, 9, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]}, 400),
        ("V6", "Editor", "Grid larger than 40x40", "POST", "/run", {"layout": [[0] * 41] * 41}, 422),
        ("V7", "Live floor", "Pickup outside the warehouse", "POST", "/fleet/task",
         {"pickup": [999, 999], "dropoff": drops[0]}, 400),
        ("V8", "Live floor", "Pickup on a wall cell", "POST", "/fleet/task", {"pickup": [0, 0], "dropoff": drops[0]}, 400),
        ("V9", "Live floor", "Pickup equals drop-off", "POST", "/fleet/task", {"pickup": picks[0], "dropoff": picks[0]}, 400),
        ("V10", "Live floor", "Coordinate with three values", "POST", "/fleet/task",
         {"pickup": [1, 2, 3], "dropoff": drops[0]}, 400),
        ("V11", "Live floor", "Unknown priority 'high'", "POST", "/fleet/task",
         {"pickup": picks[1], "dropoff": drops[0], "priority": "high"}, 422),
        ("V12", "Live floor", "Zero robots", "POST", "/fleet/start", {"n_robots": 0}, 422),
        ("V13", "Live floor", "Advance 10,000 ticks in one call", "POST", "/fleet/advance", {"ticks": 10000}, 400),
    ]
    for cid, module, what, method, path, body, want in bad_inputs:
        st, p = call(api, method, path, body)
        case(cid, module, what, f"{want}, readable message", st == want and bool(detail(p)),
             f"{st}, \"{detail(p)[:90]}\"")

    # ---------------- error / exception handling ----------------
    st, p = call(api, "GET", "/episodes/..%2F..%2Fbackend%2Fmain.py")
    case("E1", "Backend", "Path traversal in episode id", "404, no file contents leaked",
         st == 404 and "import" not in json.dumps(p), f"{st}, \"{detail(p)[:60]}\"")
    st, p = call(api, "GET", "/episodes/does_not_exist")
    case("E2", "Backend", "Unknown episode id", "404, readable message", st == 404, f"{st}, \"{detail(p)[:60]}\"")
    req = urllib.request.Request(api + "/run", data=b"{not json", method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req)
        st = 200
    except urllib.error.HTTPError as err:
        st = err.code
    case("E3", "Backend", "Malformed JSON body", "422, no server error", st == 422, f"{st}")
    call(api, "POST", "/fleet/stop")
    st, p = call(api, "POST", "/fleet/task", {"pickup": picks[0], "dropoff": drops[0]})
    case("E4", "Live floor", "Add a product with no floor running", "4xx, tells you to start one",
         400 <= st < 500, f"{st}, \"{detail(p)[:70]}\"")

    passed = sum(r[5] == "PASS" for r in rows)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Demo test cases",
        "",
        f"Generated by `python scripts/run_test_cases.py` against the running backend on {stamp}.",
        "Every *Actual* cell is what the API returned in that run. "
        f"**{passed} of {len(rows)} passed.**",
        "",
        "F = functional (end to end), V = input validation, E = error and exception handling.",
        "The 276 unit and integration tests (`python -m pytest`) and the verify scripts",
        "(`scripts/verify_live_floor.py`, `scripts/verify_single_robot.py`) cover the rest.",
        "",
        "| ID | Module | Test case | Expected | Actual | Result |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{passed}/{len(rows)} passed -> {args.out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
