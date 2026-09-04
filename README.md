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
| PPO training pipeline (Stable-Baselines3) | Implemented, **run** - three policies trained and committed |
| Evaluation, metrics, reports | Implemented, results in `data/results/` |
| FastAPI backend | Implemented, tested, accepts a user-drawn custom layout |
| React dashboard (Dashboard / Editor / Info tabs) | Implemented, replays real recordings, live per-step reward breakdown |
| Battery-aware RL | **Trained but does not work yet** - PPO scores 0/30 on `battery_constrained`; reported honestly below |
| Multi-robot / MARL | **Not implemented** - future scope |
| Reward-weight ablation study | **Not run yet** - the weights in `configs/` are reasoned starting values, not tuned results |

Tests: `86 passed`. Every number in this README and in `data/` comes from a run
that actually happened; see [Academic integrity](#academic-integrity).

---

## Results

Numbers below are produced by `python scripts/run_experiments.py` and stored in
`data/results/`. 30 episodes per controller, identical episode seeds
(1000-1029) for every controller, so the comparison is paired.

### Static warehouse (`default`) — PPO matches optimal planning

| Controller | Success | Steps | Path efficiency | Collisions |
|---|---|---|---|---|
| PPO | 30/30 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 |
| A* | 30/30 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 |
| BFS | 30/30 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 |
| Random | 1/30 | 242.00 | 0.13 | 74.73 ± 6.86 |

PPO drives a shortest path on every one of the 30 episodes — identical to A*,
using only local observations. This is the ceiling for a static grid: A* cannot
be beaten there, only matched.

### Moving traffic (`dynamic_obstacles`) — a real trade-off appears

| Controller | Success | Steps | Path efficiency | Collisions |
|---|---|---|---|---|
| PPO | 30/30 | **28.93 ± 2.55** | 0.98 ± 0.01 | **0.47 ± 0.34** |
| A* | 30/30 | 30.50 ± 3.12 | 0.93 ± 0.03 | 0.00 ± 0.00 |
| BFS | 30/30 | 31.57 ± 3.59 | 0.91 ± 0.03 | 0.00 ± 0.00 |
| Random | 3/30 | 404.67 ± 101.59 | 0.11 | 85.17 ± 6.64 |

With eight moving obstacles PPO finishes deliveries in **~1.6 fewer steps than
A\*** (28.93 vs 30.50) because it squeezes past traffic instead of routing
around it — and pays for that with **0.47 collisions per episode**, where the
replanning A\* has zero. Faster but less safe: which one is "better" depends on
whether a bump costs more than the time saved.

### Battery-constrained (`battery_constrained`) — PPO fails, A* does not

| Controller | Success | Steps | Charging events | Energy used |
|---|---|---|---|---|
| PPO | **0/30** | n/a | 0.00 | 45.00 ± 0.00 |
| A* | 30/30 | 70.50 ± 4.63 | 9.17 ± 0.94 | 94.22 ± 6.02 |
| BFS | 30/30 | 70.50 ± 4.63 | 9.17 ± 0.94 | 94.22 ± 6.02 |
| Random | 0/30 | n/a | 6.27 ± 2.54 | 90.96 ± 14.50 |

Two tasks per episode, 45% starting charge, 1.5% per move. PPO never visits a
charger: it consumes exactly its starting 45% and dies every time, while the
battery-aware A* controller detours to charge and succeeds 30/30. This is
reported as-is rather than tuned away — it is the clearest limitation the
project has found so far, and the likely causes (no explicit reward for
charging, credit assignment across two tasks, 600k steps, one training seed, no
hyper-parameter search) are listed in
[docs/experiments.md](docs/experiments.md#threats-to-validity).

Full tables including `simple_static` and `complex_static`, the metric
definitions and the threats to validity are in
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
python -m rl_agent.evaluate --model rl_agent/models/ppo_default.zip --scenario default --compare astar bfs random
```

Open the dashboard:

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

The dashboard has three tabs:

- **Dashboard** — replays the recorded episodes and result tables above; this is where every reported number lives, including a live, per-step reward breakdown (every term the reward function can pay or charge, and which ones fired on the current step).
- **Editor** — paint a warehouse by hand (walls, shelves, storage points, packing and charging stations), pick a ruleset (default / dynamic obstacles / battery-constrained) and a controller (PPO, A*, BFS, Random), and run it. With a connected backend this runs the real Python simulation (`uvicorn backend.main:app --reload`, see [docs/deployment.md](docs/deployment.md)), including PPO and moving obstacles. Without one, A*, BFS and Random run in an in-browser sandbox for exploration — its numbers are never part of the reported results, only the Python pipeline's are.
- **Info** — an in-app explanation of the MDP, the reward function, the algorithms, the two reward-shaping bugs found and fixed, and the measured results, so the project is explainable from the deployed site alone.

---

## Repository layout

```text
simulation/     the world: layout, robot, tasks, obstacles, engine, renderer
environment/    Gymnasium wrapper - the only place a reward is computed
rl_agent/       PPO config, training, evaluation
baselines/      A*, BFS and the battery-aware classical controller
analytics/      episode runner, metrics, report generation
backend/        FastAPI service (scenarios, replays, live runs)
frontend/       React + Vite dashboard (deployed to Vercel)
configs/        scenario YAML files - one file fully defines an experiment
data/           real experiment output: replays, results, training logs
tests/          85 tests covering the simulator, env, planners, analytics, API
docs/           architecture, RL formulation, experiments, deployment
scripts/        run_experiments.py - reproduces every result with one command
```

---

## How it works

```text
configs/*.yaml -> WarehouseSimulation -> WarehouseEnv -> PPO / A* / BFS / random
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
| `battery_constrained` | 21x15 | none | 45% start, 1.5%/move, 2 tasks | Charging detours become mandatory |

A scenario file inherits from `configs/default.yaml` with `extends:` and
overrides only what changes, so a difference in results always traces back to a
named parameter.

---

## Reproducing the experiments

```bash
python scripts/run_experiments.py                  # trains 3 policies, then evaluates everything
python scripts/run_experiments.py --skip-training  # evaluate with the committed models
python scripts/run_experiments.py --quick          # 20k-step training runs, for a smoke test
```

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

The suite covers layout validity and connectivity, movement and collision
rules, pickup/delivery/charging, battery depletion and timeouts, planner
optimality (A* path length equals the BFS distance field), controller
behaviour, reward-shaping regressions, metric aggregation and the HTTP API.

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
and evaluation, metrics and reports, backend, dashboard.

Next, in order:

1. **Make battery-aware RL work.** Start by raising `battery_depleted_penalty`
   above `delivery_reward` (a policy that delivers once and dies currently
   still scores positively) and by giving charging its own signal.
2. Repeat each training run with 3-5 seeds so differences can be called
   significant rather than suggestive.
3. Reward-weight ablation: which terms actually matter?
4. Obstacle-density sweep: at what traffic level does replanning A* degrade?
5. Task queues with priorities, then multiple robots and multi-agent RL (MAPPO).

Explicitly out of scope: physical robots, ROS/ROS2, 3D simulation, computer
vision. Those are future scope, not part of this project.
