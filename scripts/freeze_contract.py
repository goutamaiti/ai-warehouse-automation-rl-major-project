"""Record the single-robot behaviour that must not change, as a golden file.

    python scripts/freeze_contract.py            # show what would change
    python scripts/freeze_contract.py --write    # overwrite the golden file

Why this exists
---------------
``tests/test_frozen_contract.py`` compares the live system against
``tests/golden/frozen_contract.json`` and fails if anything in the single-robot
path drifts. That file is the project's tripwire while the multi-robot work
goes on alongside it: the fleet is a new layer, and the whole point is that the
six scenarios and the policies trained on them keep behaving *exactly* as they
did when the results in ``data/results/`` were measured.

Regenerating is deliberately a separate command, never a pytest flag. A golden
file that refreshes itself on the next test run protects nothing - the first
accidental change would be silently blessed. If this script reports a
difference, the question to answer is "did I mean to change that?", and only
then ``--write``.

What is captured, and why each entry earns its place
----------------------------------------------------
* layout digest        - the map is a pure function of the config; a change
                         here means every recorded episode is on a different
                         warehouse.
* observation space    - shape and bounds; a change invalidates every trained
                         policy, because the network's input layer no longer
                         matches.
* observation labels   - the *meaning* of each index. A reordering would keep
                         the shape valid while silently feeding the policy
                         different features.
* reset observation    - the actual numbers at a fixed seed, so a change in
                         normalisation or in any single feature is caught.
* reward component keys- catches a reward term being added or removed.
* stepped trajectory   - a fixed action sequence through the reward function,
                         so any change in dynamics *or* in reward weighting
                         shows up as a concrete per-step difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from environment import make_env  # noqa: E402
from simulation.engine import Action  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "frozen_contract.json"

#: Every scenario whose results are committed under data/results/.
SCENARIOS = (
    "simple_static",
    "default",
    "complex_static",
    "dynamic_obstacles",
    "battery_constrained",
    "battery_tuned",
)

#: Seed used for the recorded trajectory. 1000 is the first evaluation seed,
#: so this is the same problem instance the committed results were measured on.
SEED = 1000

#: A fixed action sequence rather than a policy: no model file is needed, it
#: runs in milliseconds, and it exercises movement, collisions and waiting
#: without depending on anything that is itself allowed to change.
ACTIONS = (
    [Action.UP] * 3
    + [Action.RIGHT] * 3
    + [Action.DOWN] * 2
    + [Action.LEFT] * 2
    + [Action.WAIT]
    + [Action.UP] * 2
    + [Action.RIGHT] * 2
    + [Action.DOWN]
    + [Action.LEFT]
    + [Action.WAIT] * 2
)

#: Decimal places kept for floats. Six is far tighter than any change that
#: would matter and loose enough not to trip on platform float noise.
PRECISION = 6


def _round(value: float) -> float:
    return round(float(value), PRECISION)


def capture_scenario(name: str) -> dict:
    """Record everything that must stay constant for one scenario."""
    env = make_env(name)
    observation, info = env.reset(seed=SEED)

    layout = env.sim.layout
    grid_bytes = np.ascontiguousarray(layout.grid, dtype=np.int8).tobytes()

    record: dict = {
        "layout": {
            "height": layout.height,
            "width": layout.width,
            "grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
            "storage_points": [list(p) for p in layout.storage_points],
            "packing_stations": [list(p) for p in layout.packing_stations],
            "charging_stations": [list(p) for p in layout.charging_stations],
        },
        "spaces": {
            "observation_shape": list(env.observation_space.shape),
            "observation_low": _round(float(env.observation_space.low.min())),
            "observation_high": _round(float(env.observation_space.high.max())),
            "n_actions": int(env.action_space.n),
        },
        "observation_labels": env.observation_labels(),
        "reset": {
            "observation": [_round(v) for v in observation.tolist()],
            "optimal_path_length": int(info["optimal_path_length"]),
            "task": {
                "pickup": info["task"]["pickup"],
                "dropoff": info["task"]["dropoff"],
            },
        },
        "steps": [],
    }

    for index, action in enumerate(ACTIONS):
        if env.sim.done:
            break
        _obs, reward, terminated, truncated, step_info = env.step(int(action))
        components = step_info["reward_components"]
        record["steps"].append(
            {
                "i": index,
                "action": int(action),
                "reward": _round(reward),
                "components": {k: _round(v) for k, v in sorted(components.items())},
                "battery": _round(step_info["battery"]),
                "distance_to_target": int(step_info["distance_to_target"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "events": list(step_info["events"]),
            }
        )
        if terminated or truncated:
            break

    record["reward_component_keys"] = sorted(components)
    return record


def capture_all() -> dict:
    return {
        "_comment": (
            "Golden record of single-robot behaviour. Regenerate ONLY via "
            "`python scripts/freeze_contract.py --write`, and only when a change "
            "to this behaviour was intended. See the module docstring."
        ),
        "seed": SEED,
        "actions": [int(a) for a in ACTIONS],
        "precision": PRECISION,
        "scenarios": {name: capture_scenario(name) for name in SCENARIOS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="overwrite the golden file (only when the change was intended)",
    )
    args = parser.parse_args(argv)

    captured = capture_all()
    serialised = json.dumps(captured, indent=2, sort_keys=True) + "\n"

    if not GOLDEN_PATH.exists():
        if not args.write:
            print(f"no golden file at {GOLDEN_PATH}; run with --write to create it")
            return 1
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(serialised, encoding="utf-8")
        print(f"created {GOLDEN_PATH}")
        return 0

    existing = GOLDEN_PATH.read_text(encoding="utf-8")
    if existing == serialised:
        print(f"unchanged: {GOLDEN_PATH}")
        return 0

    differing = [
        name
        for name in SCENARIOS
        if json.loads(existing)["scenarios"].get(name) != captured["scenarios"][name]
    ]
    print(f"DIFFERS from {GOLDEN_PATH}")
    print(f"  scenarios that changed: {', '.join(differing) or '(metadata only)'}")
    if not args.write:
        print("  run with --write only if this change was intended")
        return 1
    GOLDEN_PATH.write_text(serialised, encoding="utf-8")
    print("  overwritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
