# Project status - phase by phase

Checked against the project memo's 10-week plan on 2026-10-30. `[x]` means built
and verified; `[ ]` means not done, with the reason. Numbers come from
`data/results/`, `data/seeds/`, `data/training/` or the verify scripts.

## Week 1 - Research and setup
- [x] Research question and scope fixed (software-only; PPO primary, A* main baseline)
- [x] Literature survey - 7 papers (Demo-1 and Demo-2 decks, slides 5-7)
- [x] Architecture designed and documented (`docs/architecture.md`)
- [x] Git repository, `requirements.txt`, `pyproject.toml`, CI (`.github/workflows/ci.yml`: pytest + dashboard build)

## Week 2 - Simulator
- [x] 2D grid warehouse: walls, shelves, aisles, storage/pick points, packing stations, charging stations
- [x] Procedural layouts driven by one YAML file per scenario (`configs/`)
- [x] Robot movement, collision detection, layout validation (reachability)
- [x] Terminal renderer (`simulation/renderer.py`)
- [x] Hand-drawn layouts (`layout_from_grid`, Editor tab)

## Week 3 - RL environment
- [x] Gymnasium environment, passes the API checker
- [x] 37-value local observation (5x5 patch + scalars), 5 discrete actions
- [x] Shaped reward with 10 terms, potential-based progress term
- [x] Two reward traps found and fixed (loitering bonus, delivery-step penalty), with regression tests
- [x] Shaping-invariance limit identified and documented (gamma 1.0 vs PPO 0.99)

## Week 4 - RL training
- [x] PPO (Stable-Baselines3) training pipeline, model save/load, checkpoints, best-model
- [x] Six policies trained, one per scenario
- [x] Learning curves exported from the training logs (3 runs with logs kept)
- [ ] Q-Learning / DQN validation - skipped deliberately (the memo allows it; PPO went straight in)

## Week 5 - Tasks
- [x] Pickup and delivery, task generation, task completion
- [x] Multiple tasks per episode (battery scenarios: 2)
- [x] Task queue (multi-robot Live floor)
- [x] Task priorities: normal/urgent; urgent dispatched first; per-priority turnaround (urgent 58.3 vs 187.1 ticks under load)

## Week 6 - Advanced navigation
- [x] Static obstacles
- [x] Dynamic obstacles (8 moving, random walk)
- [x] Battery: capacity, per-move and idle cost, charging stations, charging process
- [x] Battery-aware classical controller (detours to charge)
- [x] Battery-aware RL - diagnosed (dying paid under the original reward) and fixed: 0/30 -> 27/30

## Week 7 - Evaluation
- [x] A* and BFS baselines (full map, replanning every step)
- [x] Paired evaluation: 30 episodes, seeds 1000-1029, identical for every controller
- [x] Metrics: success, steps, path length/efficiency, collisions, energy, charging, idle, reward, 95% CIs
- [x] Six scenarios in complexity order
- [x] Three training seeds per policy (`data/seeds/summary.md`)
- [x] All 540 published episodes reproduce exactly with the current code
- [ ] Reward-weight ablation study - not run
- [ ] Dijkstra - not needed (uniform-cost grid: same paths as BFS)

## Week 8 - Dashboard
- [x] React portal, four tabs, flat monochrome design
- [x] Dashboard: replays, per-step reward breakdown, results chart (one metric at a time, 95% CI brackets), learning curves
- [x] Live floor: multiple robots, add a product, nearest-robot dispatch, live stream
- [x] Editor: paint a warehouse, run it (backend or in-browser sandbox)
- [x] Info: method explained, reward walked step by step on a real recording, results read live
- [x] One shared cell vocabulary (marks, not shades) across every map

## Week 9 - Integration and testing
- [x] FastAPI backend (16 endpoints), hostile-input hardened (fuzzed: no 5xx, no traceback leaks, no path traversal)
- [x] 276 automated tests, including a 50-test frozen-contract tripwire
- [x] `verify_live_floor.py` - 8 scenarios, 112/112 products, nearest robot, correct cells
- [x] `verify_single_robot.py` - every scenario x controller, same PPO models as published, 540/540 episodes
- [x] `stress_live_floor.py` - 120 runs (3-8 robots, 3 arrival patterns, 4 seeds, both floors): 120/120
- [x] Bug sweep: 13 defects found and fixed across fleet, engine, API and validators

## Week 10 - Documentation and demo
- [x] README with status, results, limitations
- [x] `docs/`: architecture, RL formulation, experiments, deployment, demo run sheet, this status
- [x] Demo-2 deck, 21 slides (`presentation/Second_Project_Demo.pptx`)
- [x] One-command demo launcher with port checks and `-Verify` (`start_demo.ps1`)
- [x] Viva question bank with measured answers (`docs/demo-script.md`)
- [ ] Final project report - not written yet
- [ ] Architecture diagrams as images for the report - only in slides and text

## Extensions beyond the plan
- [x] Multi-robot fleet (classical): shared queue, nearest-robot dispatch by path distance, charger zones and waiting points, deadlock detection, yielding, battery hysteresis
- [x] Realistic 46x30 robotic-fulfilment floor: 300 pods, 306 pick points, 8 stations, 8 chargers, one-way lanes (5.8x fewer blockings, 3% less throughput)
- [x] Deployed to Vercel from `main` (static; the Live floor needs the local backend)
- [ ] Multi-agent RL (MAPPO) on the fleet - future scope
- [ ] Battery policy reliability across seeds (27 / 26 / 19 of 30; one seed camps on a charger) - future scope

## Explicitly out of scope
Physical robots, ESP32/Arduino/Raspberry Pi, sensors, ROS/ROS2, 3D simulation,
computer vision. Software simulation only, by design.
