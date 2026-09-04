# Experiments

Everything in this document was produced by `python scripts/run_experiments.py`
on the machine described under [Setup](#setup). The raw files are in
`data/results/` (JSON with per-episode rows, plus a markdown table).

## Method

* **Episodes:** 30 per controller per scenario.
* **Seeds:** 1000-1029, the *same list for every controller*, so the comparison
  is paired - controller A and controller B face identical task instances,
  identical obstacle placements and identical obstacle movement.
* **Episode loop:** one shared implementation (`analytics/runner.py`), so no
  controller can benefit from a different stopping rule or metric definition.
* **Information asymmetry (deliberate):** A* and BFS receive the full map and
  every obstacle position each step; PPO receives only its 37-value
  observation (a 5x5 local patch plus task/battery scalars).

### Metrics

| Metric | Definition | Averaged over |
|---|---|---|
| Success rate | Episodes where every task was delivered | all episodes |
| Steps | Simulated time steps until the episode ended | successful episodes |
| Path length | Number of executed moves (waits and blocked moves excluded) | successful episodes |
| Path efficiency | Shortest possible path length / path actually driven | successful episodes |
| Collisions | Attempts to enter a wall, shelf or occupied cell | all episodes |
| Energy | Battery percentage points consumed | all episodes |
| Charging events | Steps spent taking charge at a station | all episodes |
| Reward | Total shaped reward of the episode | all episodes |

"Shortest possible path length" is computed with a BFS distance field on the
static map, ignoring dynamic obstacles - it is a lower bound no controller can
beat, which is why path efficiency can be below 1.0 but never above.

Numbers are reported as `mean ± half-width of the 95% confidence interval`.
Metrics restricted to successful episodes report `n/a` when nothing succeeded.

### Setup

| Item | Value |
|---|---|
| Python | 3.12.4 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.13.0 (CPU) |
| Gymnasium | 1.3.0 |
| Platform | Windows 11, CPU only |

## Training

| Scenario | Model | Timesteps | Wall clock |
|---|---|---|---|
| `default` | `ppo_default` | 300,000 | ~113 s |
| `dynamic_obstacles` | `ppo_dynamic` | 400,000 | ~150 s |
| `battery_constrained` | `ppo_battery` | 600,000 | ~230 s |

PPO hyper-parameters are in `rl_agent/config.py` (8 parallel environments,
rollout 512 steps each, batch 1024, 10 epochs, lr 3e-4, gamma 0.99, GAE lambda
0.95, clip 0.2, entropy bonus 0.01, MLP 128x128). They are Stable-Baselines3
defaults for discrete actions apart from the rollout length and the entropy
bonus; **no hyper-parameter search was run.**

## Results

All tables: 30 episodes per controller, seeds 1000-1029, `mean ± 95% CI`.
Source files are named in each heading.

### 1. `simple_static` — sanity check

`data/results/simple_static_baselines.md`

| Controller | Episodes | Success | Steps | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|---|
| astar | 30/30 | 1.00 | 19.60 ± 1.84 | 1.00 ± 0.00 | 0.00 | 42.42 ± 1.73 |
| bfs | 30/30 | 1.00 | 19.60 ± 1.84 | 1.00 ± 0.00 | 0.00 | 42.42 ± 1.73 |
| random | 2/30 | 0.07 | 114.00 ± 154.84 | 0.25 ± 0.32 | 43.53 ± 5.13 | -55.33 ± 7.54 |

Both planners drive shortest paths. A random policy solves 7% of episodes, so
the task is not trivially solvable by chance — the metric has room to
discriminate.

### 2. `complex_static` — longer paths

`data/results/complex_static_baselines.md`

| Controller | Episodes | Success | Steps | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|---|
| astar | 30/30 | 1.00 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 | 65.70 ± 2.99 |
| bfs | 30/30 | 1.00 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 | 65.70 ± 2.99 |
| random | 0/30 | 0.00 | n/a | n/a | 97.57 ± 14.37 | -127.03 ± 16.29 |

In a 31x21 warehouse random search never succeeds within the step budget.

### 3. `default` — PPO vs classical planning, static

`data/results/default_ppo_vs_baselines.md`

| Controller | Episodes | Success | Steps | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 | 50.19 ± 2.28 |
| astar | 30/30 | 1.00 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 | 50.19 ± 2.28 |
| bfs | 30/30 | 1.00 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 | 50.19 ± 2.28 |
| random | 1/30 | 0.03 | 242.00 | 0.13 | 74.73 ± 6.86 | -99.92 ± 9.08 |

**PPO is exactly optimal on all 30 episodes.** Every row for the three
controllers is identical because all three drove a shortest path in every
episode.

This is the expected ceiling, and it answers the first research question: a
policy trained from a 5x5 local view plus task/battery scalars can reproduce
full-information shortest-path planning on a static warehouse. It cannot do
better, because A* is already optimal here.

### 4. `dynamic_obstacles` — where the approaches diverge

`data/results/dynamic_obstacles_ppo_vs_baselines.md`

| Controller | Episodes | Success | Steps | Path length | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | **28.93 ± 2.55** | 28.47 ± 2.45 | 0.98 ± 0.01 | **0.47 ± 0.34** | 49.67 ± 2.22 |
| astar | 30/30 | 1.00 | 30.50 ± 3.12 | 30.33 ± 3.11 | 0.93 ± 0.03 | 0.00 ± 0.00 | 50.02 ± 2.26 |
| bfs | 30/30 | 1.00 | 31.57 ± 3.59 | 31.33 ± 3.58 | 0.91 ± 0.03 | 0.00 ± 0.00 | 49.95 ± 2.23 |
| random | 3/30 | 0.10 | 404.67 ± 101.59 | 226.33 ± 52.32 | 0.11 ± 0.01 | 85.17 ± 6.64 | -110.01 ± 8.70 |

The interesting result of the project:

* PPO completes deliveries in **1.6 fewer steps** than A* (28.93 vs 30.50) and
  stays closer to the shortest possible path (efficiency 0.98 vs 0.93).
* PPO pays for this with **0.47 collisions per episode**; A* has none.
* The mechanism is visible in the replays: the replanning A* controller routes
  *around* any cell an obstacle occupies, which costs a detour whenever traffic
  sits in a narrow aisle; PPO keeps pushing along the direct route and
  occasionally bumps into an obstacle that moved into its path.
* Total reward is statistically indistinguishable (49.67 vs 50.02): the time
  PPO saves is almost exactly cancelled by the collision penalty. That is the
  reward function working as designed, not a coincidence.

Note that A* also beats BFS here (30.50 vs 31.57 steps) even though both return
shortest paths: they break ties differently, and under replanning a different
tie-break leads into a different amount of traffic.

### 5. `battery_constrained` — the limitation

`data/results/battery_constrained_ppo_vs_baselines.md`

| Controller | Episodes | Success | Steps | Charging events | Energy used | Reward |
|---|---|---|---|---|---|---|
| ppo | **0/30** | 0.00 | n/a | 0.00 ± 0.00 | 45.00 ± 0.00 | 33.13 ± 3.44 |
| astar | 30/30 | 1.00 | 70.50 ± 4.63 | 9.17 ± 0.94 | 94.22 ± 6.02 | 97.50 ± 3.07 |
| bfs | 30/30 | 1.00 | 70.50 ± 4.63 | 9.17 ± 0.94 | 94.22 ± 6.02 | 97.50 ± 3.07 |
| random | 0/30 | 0.00 | n/a | 6.27 ± 2.54 | 90.96 ± 14.50 | -30.68 ± 5.20 |

PPO trained for 600,000 steps on this scenario and **never learned to charge**:
zero charging events, and an energy consumption of exactly 45.00 ± 0.00 - its
entire starting charge - in every single episode. It delivers the first package
(hence the positive reward of 33.13) and then runs the battery flat before
finishing the second.

The battery-aware A* controller succeeds 30/30 with an average of 9.17 charging
steps per episode.

Why this is being reported rather than tuned away: it is a real, reproducible
limitation, and it identifies the next piece of work precisely. Plausible
causes, in the order worth testing:

1. **No positive signal for charging.** Charging currently earns nothing; it is
   only worth doing because it avoids a later penalty ~40 steps away. That is a
   long credit-assignment gap for a 10-value discount horizon of gamma = 0.99.
2. **The depletion penalty is too cheap.** At -10 it is smaller than the
   delivery reward of +20, so a policy that delivers one package and dies still
   scores positively - which is exactly the behaviour observed.
3. **Two tasks per episode** double the horizon compared to every scenario PPO
   solved.
4. Single training seed, no hyper-parameter search, 600k steps.

A reward-shaping term based on "reachable charge margin", or simply raising
`battery_depleted_penalty` above `delivery_reward`, is the first experiment to
run. **No such experiment has been run yet, so no claim is made about whether it
would work.**

## Summary against the research question

| Question | Evidence |
|---|---|
| Can PPO match classical planning on static warehouse navigation? | Yes - identical to optimal A* on 30/30 episodes (`default`). |
| Does RL help when the warehouse is dynamic? | It is faster (28.93 vs 30.50 steps) but collides 0.47 times per episode where A* never does. No net reward advantage. |
| Does RL handle battery constraints? | Not with the current reward and budget: 0/30 vs 30/30 for A*. |
| Is the comparison fair? | It is biased *against* RL: the planners get the full map and every obstacle position; PPO gets a 5x5 patch. |

## Threats to validity

1. **One training seed per scenario.** Each policy was trained once (seed 0).
   RL results vary between training seeds; a rigorous claim needs 3-5 seeds per
   configuration. Treat differences of a fraction of a step as noise.
2. **Reward weights were not tuned.** The weights are reasoned starting values.
   A different weighting could change PPO's behaviour substantially.
3. **The obstacle model is benign.** Obstacles never step onto the robot, so
   collisions can only be caused by the controller. A harsher model (obstacles
   that do not yield) would change the collision metric for every controller.
4. **A* is given an advantage.** Full map, full obstacle visibility, replanning
   every single step. This is the strongest reasonable classical baseline, so
   the comparison is conservative with respect to the RL side.
5. **Single robot.** Nothing here says anything about congestion or deadlocks
   between multiple robots.
