# Architecture

## Layering

The system is built as five layers. Each one only knows about the layer below
it, which is what makes it possible to swap the controller (PPO, A*, BFS,
random) without touching the world, and to run training with rendering off.

```text
              Dashboard (React)            <- replays recordings, shows metrics
                     |  HTTP (optional)
              FastAPI backend              <- scenarios, replays, live runs
                     |
   +-----------------+------------------+
   |                 |                  |
 Controllers    Analytics/runner    Recorder
 (PPO, A*, BFS)      |                  |
   |                 |                  |
   +----> Gymnasium environment <--------+   <- observation + reward only
                     |
          Warehouse simulation engine          <- all dynamics live here
                     |
   layout  |  robot  |  tasks  |  obstacles
```

### Who owns what

| Concern | Module | Notes |
|---|---|---|
| Static map (walls, shelves, stations) | `simulation/warehouse.py` | Immutable for the whole episode |
| Grid helpers, BFS distance fields | `simulation/navigation.py` | Shared by env, planners and analytics |
| Robot state and battery bookkeeping | `simulation/robot.py` | Holds state, never decides anything |
| Delivery tasks | `simulation/tasks.py` | Seeded generation, status transitions |
| Moving traffic | `simulation/obstacles.py` | Random walk or patrol |
| **All dynamics** (move, collide, pick up, deliver, charge, time out) | `simulation/engine.py` | Returns facts, not rewards |
| Observation vector and **reward** | `environment/warehouse_env.py` | The only place a reward exists |
| Classical planners | `baselines/astar.py`, `baselines/bfs.py` | Pure functions on a blocked grid |
| Classical controller (battery-aware, replanning) | `baselines/controller.py` | Drives the env like a policy |
| PPO training / evaluation | `rl_agent/` | Stable-Baselines3 |
| Episode loop, metrics, reports | `analytics/` | One loop shared by every controller |
| Text and JSON rendering | `simulation/renderer.py` | Read-only consumer of the state |
| HTTP API | `backend/` | Thin layer over `backend/services`; `POST /api/run` accepts an optional user-drawn `layout` grid (see `simulation.warehouse.layout_from_grid`) |
| Dashboard | `frontend/` | Three tabs: Dashboard (replays recorded JSON), Editor (paints a layout, runs it), Info (in-app explanation) |
| Offline sandbox | `frontend/src/lib/offlineSimulation.js` | Independent JS port of the engine + reward, used by the Editor's A*/BFS/Random controllers when no backend is connected. Not a source of reported results - see `docs/experiments.md` |

### Two deviations from the structure in the project memo

1. `simulation/engine.py` was split out of `simulation/warehouse.py`. Keeping
   the static map and the step logic in one file made it the largest module in
   the project; splitting them keeps each under ~300 lines and makes the
   "layout is immutable, engine is mutable" boundary explicit.
2. `analytics/runner.py` was added. Both `baselines/evaluate_baselines.py` and
   `rl_agent/evaluate.py` need to run episodes and collect metrics; a shared
   loop is what guarantees the two are measured identically, and it avoids the
   duplicated episode loop the memo warns against.

## Why the reward lives outside the engine

`WarehouseSimulation.step()` returns a `StepOutcome` - a record of *what
happened* (moved, collided with a wall, picked up, delivered, energy used,
distance before/after). It contains no reward.

The environment converts that record into a scalar. Two consequences:

* Reward-shaping experiments only ever touch `environment/warehouse_env.py`
  and `configs/*.yaml`; the physics cannot drift while you tune the reward.
* The classical controllers run on the same engine and are scored by the same
  metrics, so "PPO vs A*" compares controllers, not implementations.

## Data flow of an experiment

```text
configs/<scenario>.yaml
        |
        v
ScenarioConfig -> WarehouseSimulation -> WarehouseEnv
                                            |
                        +-------------------+-------------------+
                        |                                       |
                 rl_agent/train.py                    analytics/runner.py
                        |                                       |
             rl_agent/models/*.zip  --------------------->  EpisodeMetrics
                                                                |
                                                    analytics/metrics.summarise
                                                                |
                                        data/results/*.json + *.md, data/episodes/*.json
                                                                |
                                                    frontend (static copy) / API
```

Every artefact under `data/` is produced by a real run and carries the seeds
and the timestamp that produced it. `scripts/run_experiments.py` regenerates
all of them with one command.

## Reproducibility

An experiment is fully described by `(scenario file, first seed, number of
episodes)`:

* the layout is a deterministic function of the layout config;
* task sampling, obstacle placement and obstacle movement all draw from one
  `numpy.random.Generator` seeded per episode;
* `env.reset(seed=k)` therefore reconstructs episode *k* exactly;
* every controller is evaluated on the same seed list, so comparisons are
  paired.
