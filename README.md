# AI-Based Warehouse Automation System Using Reinforcement Learning for Intelligent Robot Navigation

A software-only simulation of an automated warehouse in which a virtual robot
learns to navigate, pick up and deliver packages while avoiding obstacles and
managing a battery - and a like-for-like comparison against classical path
planning.

**Research question.** Can a Reinforcement Learning navigation policy match or
improve on conventional path planning for warehouse delivery tasks under
simulated warehouse conditions - and where does each approach break down?

No hardware is involved: no ESP32, no motors, no LiDAR, no ROS. Everything runs
in Python plus a React dashboard.

---

## Status

| Area | State |
|---|---|
| Warehouse simulator (grid, shelves, stations, tasks, battery, dynamic obstacles) | Implemented, unit tested |
| Gymnasium environment (37-value observation, 5 actions, shaped reward) | Implemented, passes `gymnasium` API checker |
| A* / BFS planners + battery-aware controller | Implemented, unit tested |
| PPO training pipeline (Stable-Baselines3) | Implemented, **run** - six policies trained and committed |
| Evaluation, metrics, reports | Implemented - 6 scenarios x 3 controllers x 30 paired episodes in `data/results/` |
| Battery-aware RL | **Works after a reward fix** - PPO 0/30 on `battery_constrained` (original reward) -> **27/30** on `battery_tuned` (corrected reward, 2M steps); across 3 training seeds 27, 26 and 19 of 30 (vs 0, 0 and 6 before the fix), **82 of 90 keeping the best checkpoint**; still below A*'s 30/30 |
| Multi-robot fleet (Live floor) | Implemented with **classical** control - shared task queue with normal/urgent priority (urgent served 3.2x faster under load), nearest-robot dispatch, charger zones, deadlock detection, 46x30 one-way robotic-fulfilment floor. No learned policy drives it. |
| FastAPI backend | Implemented, tested, accepts a user-drawn custom layout |
| React dashboard (Dashboard / Live floor / Editor / Info tabs) | Implemented, replays real recordings, live per-step reward breakdown |
| Multi-agent RL (MAPPO etc.) | **Not implemented** - future scope |
| Multiple training seeds | **Run** - every PPO policy trained with 3 seeds; see [Across training seeds](#across-training-seeds) |
| Reward-weight ablation study | **Run** - 5 terms x 3 seeds; progress shaping and the collision penalty matter, energy/step/waiting showed no measurable effect; see [Reward-weight ablation](#reward-weight-ablation) |

Tests: `276 passed`, including a 50-test frozen-contract tripwire that pins the
exact single-robot behaviour every result was measured with. Two one-command
checks prove the demo end to end (`scripts/verify_live_floor.py`,
`scripts/verify_single_robot.py`; see [Testing](#testing)). Every number in this
README and in `data/` comes from a run that actually happened; see
[Academic integrity](#academic-integrity).

---

## Results

Numbers below are read from `data/results/`. 30 episodes per controller,
identical episode seeds (1000-1029) for every controller, so the comparison is
paired. Values are mean ± 95% confidence half-width; steps and path efficiency
are averaged over successful episodes only.

### Static warehouses - PPO matches optimal planning exactly

| Scenario | Controller | Success | Steps | Path efficiency | Collisions |
|---|---|---|---|---|---|
| `simple_static` (15x11) | PPO / A* / BFS | 30/30 each | 19.60 ± 1.84 | 1.00 | 0.00 |
| `default` (21x15) | PPO / A* / BFS | 30/30 each | 27.87 ± 2.43 | 1.00 | 0.00 |
| `complex_static` (31x21) | PPO / A* / BFS | 30/30 each | 44.37 ± 3.18 | 1.00 | 0.00 |

On all three static floors the three controllers are identical to the last
digit. PPO drives a shortest path on every episode using only a 5x5 local view
while A* sees the whole map. On a static grid A* is provably optimal, so this
is the ceiling: it can be matched, never beaten.

### Moving traffic (`dynamic_obstacles`) - a real trade-off appears

| Controller | Success | Steps | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|
| PPO | 30/30 | **28.93 ± 2.55** | **0.98 ± 0.01** | 0.47 ± 0.34 | 49.67 ± 2.22 |
| A* | 30/30 | 30.50 ± 3.12 | 0.93 ± 0.03 | **0.00** | 50.02 ± 2.26 |
| BFS | 30/30 | 31.57 ± 3.59 | 0.91 ± 0.03 | **0.00** | 49.95 ± 2.23 |

With eight moving obstacles PPO finishes about 1.6 steps sooner than A* because
it squeezes past traffic instead of routing around it, and pays for that with
0.47 collisions per episode where the replanning A* has none. Total reward is
statistically tied: the reward function prices exactly that trade-off.

### Battery - the failure, its diagnosis, and the fix

| Scenario | Controller | Success | Steps | Charging steps | Reward |
|---|---|---|---|---|---|
| `battery_constrained` (original reward) | PPO | **0/30** | n/a | 0.00 | 33.13 ± 3.44 |
| | A* / BFS | 30/30 | 70.50 ± 4.63 | 9.17 ± 0.94 | 97.50 ± 3.07 |
| `battery_tuned` (corrected reward) | PPO | **27/30** | 87.89 ± 4.90 | 15.43 ± 0.90 | 82.66 ± 9.88 |
| | A* / BFS | 30/30 | 70.50 ± 4.63 | 9.17 ± 0.94 | 94.21 ± 2.87 |

Same world in both rows: two tasks, 45% starting charge, 1.5% per move. Under
the original reward PPO never charged and died in all 30 episodes - and still
scored **+33** on average, because the -10 depletion penalty was cheaper than
the pickup, delivery and progress credit it could bank first. It was not
failing to learn; it had learnt that dying paid. `battery_tuned` raises the
depletion penalty to 60 (above anything an episode can bank), adds a
potential-based battery-reserve term and a higher wait price, and trains for
2M steps instead of 600k. PPO then charges and delivers in 27/30 episodes.

It is not a clean win: PPO charges more than it needs (15.4 vs 9.2 steps) and
drives longer routes (path efficiency 0.73 vs 0.91). The claim is that the
battery task became learnable, not that PPO beats the planner on it.

### Across training seeds

Every table above is one trained policy (training seed 0). Each was retrained
with seeds 1 and 2 using the identical command and evaluated on the same 30
episodes (`python scripts/run_seed_study.py`; results in `data/seeds/`):

| Scenario | Seed 0 (published) | Seed 1 | Seed 2 |
|---|---|---|---|
| `simple_static` | 30/30, 19.60 steps | 30/30, 19.60 | **28/30**, 27.64 |
| `default` | 30/30, 27.87 | 30/30, 27.87 | 30/30, 27.87 |
| `complex_static` | 30/30, 44.37 | 30/30, 44.37 | 30/30, 44.37 |
| `dynamic_obstacles` | 30/30, 28.93 steps, 0.47 collisions | 30/30, 29.37, 0.77 | 30/30, 29.07, 0.53 |
| `battery_constrained` (original reward) | 0/30 | 0/30 | 6/30 |
| `battery_tuned` (corrected reward) | 27/30 | 26/30 | **19/30** |

* **Holds on every seed:** the exact tie with A* on `default` and
  `complex_static`; on `dynamic_obstacles` PPO is faster than A* (30.50 steps)
  *and* collides on all three seeds; the original battery reward fails on all
  three.
* **Seed-dependent:** on `simple_static` seed 2 is not optimal (28/30, path
  efficiency 0.78 - two episodes wander until the battery runs out). On
  `battery_tuned` the fix holds on every seed but varies: seed 2 fell into
  *camping* in 10 episodes, sitting on a charger for all 600 steps without
  delivering. In total the corrected reward delivers in 72 of 90 episodes
  against 6 of 90 under the original one.

### Making the battery policy reliable

Every `battery_tuned` seed reached 100% evaluation success at some point in
training and then swung, so the run sometimes ended on a bad moment. Keeping
the **best checkpoint** (chosen on the training run's own random evaluation
episodes, never the test episodes) instead of the last one:

| Setting, 2M steps | Seeds 0 / 1 / 2 | Total |
|---|---|---|
| learning rate 3e-4, final policy (published) | 27 / 26 / 19 | 72/90 |
| learning rate 3e-4, **best checkpoint** | **30 / 27 / 25** | **82/90** - no camping on any seed |
| learning rate 1e-4, final policy | 6 / 30 / 0 | 36/90 - slower learning; seed 2 never learned |

Seed 0 now matches A* at 30/30. A lower learning rate was tried as a deeper fix
and made things worse at this budget, so it is reported, not adopted. Results:
`data/stability/summary.md`.

### Reward-weight ablation

PPO retrained on `dynamic_obstacles` with one reward term switched off at a
time, 3 seeds each, compared on task metrics (`data/ablation/summary.md`):

| Switched off | Success (of 90) | Collisions / episode |
|---|---|---|
| nothing (full reward) | 90 | 0.59 |
| progress shaping | **45** (and 3-7x longer paths) | 0.33 |
| collision penalty | 90 | **1.33** |
| energy / step / waiting | 90 each | 0.52 / 0.53 / 0.62 |

Progress shaping carries the learning; the collision penalty is what keeps
collisions down; the other three terms had no measurable effect on this
scenario.

Full tables, metric definitions and the threats to validity are in
[docs/experiments.md](docs/experiments.md).

---

## Quick start

```bash
git clone <this-repo>
cd ai-warehouse-automation-rl

python -m venv .venv
.venv\Scripts\activate           # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Watch A* solve a delivery in the terminal:

```bash
python -m baselines.evaluate_baselines --scenario default --episodes 5 --controllers astar
```

Train a policy (about 2 minutes on a laptop CPU for the default scenario):

```bash
python -m rl_agent.train --scenario default --timesteps 300000 --tag ppo_default
```

Compare it with the classical planners on identical episodes:

```bash
python -m rl_agent.evaluate --model rl_agent/models/ppo_default.zip --scenario default --compare astar bfs
```

Open the dashboard:

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

For live runs, PPO in the browser and the Live floor, also start the backend:

```bash
uvicorn backend.main:app --reload
```

The dashboard has four tabs:

- **Dashboard** - replays the recorded episodes and the measured comparison; this is where every reported number lives, with a per-step reward breakdown (every term the reward function can pay or charge, and which ones fired on the current step).
- **Live floor** - a warehouse that keeps running with several robots on a shared queue. Add a product's pickup and drop-off cell - as normal or urgent - and the nearest idle robot (by path distance through the aisles) takes it on the next tick. Robots charge, park at zone waiting points and are checked for deadlock. Runs on the 21x15 floor or a 46x30 robotic-fulfilment floor with one-way lanes. Driven by the classical controller; a demonstration, not a measured result.
- **Editor** - paint a warehouse by hand, pick a ruleset and a controller, and run it. With the backend this runs the real Python simulation, including PPO; without one, A* and BFS run in an in-browser sandbox whose numbers are never part of the reported results.
- **Info** - the MDP, the reward function walked step by step on a real recording, the algorithms, the reward-shaping bugs found and fixed, and the measured results read live from `data/results/`.

---

## Repository layout

```text
simulation/     the world: layout, robot, tasks, obstacles, engine, fleet, dispatch, zones
environment/    Gymnasium wrapper - the only place a reward is computed
rl_agent/       PPO config, training, evaluation
baselines/      A*, BFS and the battery-aware classical controller
analytics/      episode runner, metrics, report generation
backend/        FastAPI service (scenarios, replays, live runs)
frontend/       React + Vite dashboard (deployed to Vercel)
configs/        scenario YAML files - one file fully defines an experiment
data/           real experiment output: replays, results, training logs
tests/          276 tests: simulator, env, planners, fleet, analytics, API, frozen contract
docs/           architecture, RL formulation, experiments, deployment
scripts/        run_experiments.py (reproduce results), verify_*.py (pre-demo proof)
```

---

## How it works

```text
configs/*.yaml -> WarehouseSimulation -> WarehouseEnv -> PPO / A* / BFS
                       (dynamics)          (obs+reward)          |
                                                                 v
                                             analytics -> data/results + data/episodes
                                                                 |
                                                                 v
                                                    FastAPI  /  React dashboard
```

The simulation engine reports *facts* (moved, collided, delivered, energy
used); the environment turns those into a reward. That separation is what lets
the learned policy and the classical planners run on byte-identical dynamics -
see [docs/architecture.md](docs/architecture.md).

The MDP, the observation vector, the reward function and two reward-shaping
traps that were found and fixed are documented in
[docs/rl-formulation.md](docs/rl-formulation.md).

---

## Scenarios

| Scenario | Warehouse | Obstacles | Battery | Point of the experiment |
|---|---|---|---|---|
| `simple_static` | 15x11 | none | ample | Sanity check: can a controller work at all |
| `default` | 21x15 | none | ample | Baseline navigation task |
| `complex_static` | 31x21 | none | ample | Longer paths, denser shelving |
| `dynamic_obstacles` | 21x15 | 8 moving | ample | The shortest static path is no longer safe |
| `battery_constrained` | 21x15 | none | 45% start, 1.5%/move, 2 tasks | Charging detours become mandatory (original reward) |
| `battery_tuned` | 21x15 | none | same as above | Same world, corrected reward - the battery fix |

A scenario file inherits from `configs/default.yaml` with `extends:` and
overrides only what changes, so a difference in results always traces back to a
named parameter.

---

## Reproducing the experiments

```bash
python scripts/run_experiments.py                  # trains the policies, then evaluates everything
python scripts/run_experiments.py --skip-training  # evaluate with the committed models
python scripts/run_experiments.py --quick          # short training runs, for a smoke test
```

> **Warning:** evaluation writes straight into `data/results/` and
> `data/episodes/`, replacing the committed files. Run it on a clean `git status`
> so a short or accidental run can be undone with `git checkout -- data/`.

Reproducibility guarantees:

* the layout is a deterministic function of the config;
* `env.reset(seed=k)` rebuilds episode *k* exactly (task, obstacle placement,
  obstacle movement);
* every controller is scored on the same seed list through the same episode
  loop (`analytics/runner.py`).

---

## Testing

```bash
pytest                    # whole suite
pytest tests/test_env.py  # one file
```

Before a demo, two scripts prove the system end to end without writing
anything:

```bash
python scripts/verify_live_floor.py     # ~1 min: 8 Live-floor scenarios, every product checked
python scripts/verify_single_robot.py   # a few min: every scenario/controller + all 540 episodes
python scripts/stress_live_floor.py     # ~5 min: 60 stress runs, 3-8 robots (add `rmfs` for the large floor)
python scripts/run_test_cases.py        # backend running: 27 demo test cases, expected vs actual -> docs/test-cases.md
```

`verify_live_floor.py` drives the fleet API exactly as the browser does and
checks, against an independent BFS, that each product goes to the nearest idle
robot, that the robot physically stands on the pickup and drop-off cells, and
that no two robots ever share a cell. `verify_single_robot.py` confirms the demo
loads the same PPO model each published result used, and that all 540
committed evaluation episodes reproduce exactly with the current code.

The suite covers layout validity and connectivity, movement and collision
rules, pickup/delivery/charging, battery depletion and timeouts, planner
optimality (A* path length equals the BFS distance field), controller
behaviour, reward-shaping regressions, metric aggregation, the multi-robot
fleet (dispatch, zones, deadlock, one-way floor) and the HTTP API.

---

## Deployment

The dashboard is a static Vite build and deploys to Vercel from the repository
root (`vercel.json` handles the build). The FastAPI backend is optional and is
only needed for live runs from the browser. Details, including the
`VITE_API_BASE` and `ALLOWED_ORIGINS` settings, are in
[docs/deployment.md](docs/deployment.md).

---

## Academic integrity

This project follows one hard rule: **no fabricated results.**

* Every metric in `data/results/` was produced by an actual run and carries its
  seed list and timestamp.
* Every animation in the dashboard is a replay of a recorded episode
  (`data/episodes/*.json`), not a mock-up.
* Anything not yet measured is labelled *not implemented* or *not run yet* in
  the status table above, never estimated.
* Where a classical planner beats the learned policy, the table says so.

The classical controllers are deliberately given an advantage: they see the
full map and every obstacle position, while PPO only ever sees its
37-dimensional observation.

---

## Roadmap

Implemented: simulator, Gymnasium environment, A*/BFS baselines, PPO training
and evaluation, the battery reward fix, metrics and reports, backend, dashboard,
and a classically controlled multi-robot Live floor.

Known limitations, stated rather than hidden:

* **Three training seeds per policy, not five.** The main findings hold on all
  three; `simple_static` and the battery result vary between seeds (see
  [Across training seeds](#across-training-seeds)).
* **Shaping is invariant only for the undiscounted return.** The progress term
  uses shaping gamma 1.0 (so that loitering cannot pay) while PPO optimises a
  0.99-discounted return, and the potential is not zeroed at a failed terminal
  state. Ng et al.'s policy-invariance guarantee therefore holds exactly only
  for the undiscounted objective. Fixing it means retraining every policy; see
  [docs/rl-formulation.md](docs/rl-formulation.md).
* **PPO on the battery task is learnable, not better** than the planner.
* **The fleet is classical.** No learned policy has ever seen a second robot.

Next, in order:

1. Close the remaining battery gap (82 of 90 with the best checkpoint): longer
   runs and more seeds. A lower learning rate was tried and made it worse.
2. Obstacle-density sweep: at what traffic level does replanning A* degrade?
3. Multi-agent RL (MAPPO) on the fleet.

Explicitly out of scope: physical robots, ROS/ROS2, 3D simulation, computer
vision. Those are future scope, not part of this project.
