"""Stress sweep of the Live floor: robot counts x arrival patterns x seeds.

    python scripts/stress_live_floor.py            # 21x15 floor, 60 runs (~5 min)
    python scripts/stress_live_floor.py rmfs       # 46x30 one-way floor, 60 runs

Every run goes through the same checks as scripts/verify_live_floor.py (every
product delivered by the nearest eligible robot standing on its pickup and
drop-off, no shared cells, no deaths, no deadlock). 3, 4, 5, 6 and 8 robots;
a burst of 12 products, steady arrivals and long idle gaps; seeds 1000, 0, 1, 2.
Writes nothing. Exit code 0 only if every run passes.
"""
import contextlib
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
src = open("scripts/verify_live_floor.py", encoding="utf-8").read().split("results = []")[0]
exec(src)

FLOOR = sys.argv[1] if len(sys.argv) > 1 else "default"
rows = []
for n in (3, 4, 5, 6, 8):
    for pattern, sched, n_prod in (("burst", lambda i, g: 0, 12),
                                   ("steady", lambda i, g: int(g.integers(0, 10)), 15),
                                   ("gaps", lambda i, g: int(g.integers(15, 40)), 8)):
        for seed in (1000, 0, 1, 2):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ok, problems = run(f"{n} robots {pattern} seed {seed}", "default", FLOOR, n, n_prod, 2500, sched, seed=seed)
            rows.append((n, pattern, seed, ok, problems))
fails = [r for r in rows if not r[3]]
print(f"floor={FLOOR}: {len(rows) - len(fails)}/{len(rows)} runs passed")
for n, pattern, seed, ok, problems in fails:
    print(f"  FAIL {n} robots {pattern:6s} seed {seed}: {problems[-1][:110]}")
sys.exit(0 if not fails else 1)
