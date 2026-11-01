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
| `simple_static` | `ppo_simple` | 300,000 | 136.5 s |
| `default` | `ppo_default` | 300,000 | ~113 s |
| `complex_static` | `ppo_complex` | 600,000 | 265.8 s |
| `dynamic_obstacles` | `ppo_dynamic` | 400,000 | ~150 s |
| `battery_constrained` | `ppo_battery` | 600,000 | ~230 s |
| `battery_tuned` | `ppo_battery_long_s0` | 2,000,000 | 915.7 s |

PPO hyper-parameters are in `rl_agent/config.py` (8 parallel environments,
rollout 512 steps each, batch 1024, 10 epochs, lr 3e-4, gamma 0.99, GAE lambda
0.95, clip 0.2, entropy bonus 0.01, MLP 128x128). They are Stable-Baselines3
defaults for discrete actions apart from the rollout length and the entropy
bonus; **no hyper-parameter search was run.** Every policy uses training seed 0.

## Results

All tables: 30 episodes per controller, seeds 1000-1029, `mean ± 95% CI`,
read from the named file in `data/results/`. Steps and path efficiency are
averaged over successful episodes only. Every committed episode reproduces
exactly with the current code (`python scripts/verify_single_robot.py`).

### 1. `simple_static` - sanity check

`data/results/simple_static_ppo_vs_baselines.md`

| Controller | Success | Steps | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|
| ppo | 30/30 | 19.60 ± 1.84 | 1.00 ± 0.00 | 0.00 | 42.42 ± 1.73 |
| astar | 30/30 | 19.60 ± 1.84 | 1.00 ± 0.00 | 0.00 | 42.42 ± 1.73 |
| bfs | 30/30 | 19.60 ± 1.84 | 1.00 ± 0.00 | 0.00 | 42.42 ± 1.73 |

### 2. `default` - PPO vs classical planning, static

`data/results/default_ppo_vs_baselines.md`

| Controller | Success | Steps | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|
| ppo | 30/30 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 | 50.19 ± 2.28 |
| astar | 30/30 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 | 50.19 ± 2.28 |
| bfs | 30/30 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 | 50.19 ± 2.28 |

### 3. `complex_static` - longer paths, denser shelving

`data/results/complex_static_ppo_vs_baselines.md`

| Controller | Success | Steps | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|
| ppo | 30/30 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 | 65.70 ± 2.99 |
| astar | 30/30 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 | 65.70 ± 2.99 |
| bfs | 30/30 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 | 65.70 ± 2.99 |

**On all three static floors PPO is exactly optimal on all 30 episodes.** The
three rows are identical to the last digit because all three controllers drove
a shortest path in every episode. This is the expected ceiling and answers the
first research question: a policy trained from a 5x5 local view plus
task/battery scalars reproduces full-information shortest-path planning on a
static warehouse. It cannot do better, because A* is already optimal there.

### 4. `dynamic_obstacles` - where the approaches diverge

`data/results/dynamic_obstacles_ppo_vs_baselines.md`

| Controller | Success | Steps | Path efficiency | Collisions | Reward |
|---|---|---|---|---|---|
| ppo | 30/30 | **28.93 ± 2.55** | **0.98 ± 0.01** | 0.47 ± 0.34 | 49.67 ± 2.22 |
| astar | 30/30 | 30.50 ± 3.12 | 0.93 ± 0.03 | **0.00** | 50.02 ± 2.26 |
| bfs | 30/30 | 31.57 ± 3.59 | 0.91 ± 0.03 | **0.00** | 49.95 ± 2.23 |

* PPO completes deliveries in **1.6 fewer steps** than A* (28.93 vs 30.50) and
  stays closer to the shortest possible path (efficiency 0.98 vs 0.93).
* PPO pays for this with **0.47 collisions per episode**; A* has none.
* The mechanism is visible in the replays: the replanning A* controller routes
  *around* any cell an obstacle occupies, which costs a detour whenever traffic
  sits in a narrow aisle; PPO keeps pushing along the direct route and
  occasionally bumps into an obstacle that moved into its path.
* Total reward is statistically indistinguishable (49.67 vs 50.02): the time
  PPO saves is almost exactly cancelled by the collision penalty. That is the
  reward function pricing the trade-off as designed.

A* also beats BFS here (30.50 vs 31.57 steps) even though both return shortest
paths: they break ties differently, and under replanning a different tie-break
leads into a different amount of traffic.

### 5. `battery_constrained` - the failure

`data/results/battery_constrained_ppo_vs_baselines.md`

| Controller | Success | Steps | Charging steps | Energy used | Reward |
|---|---|---|---|---|---|
| ppo | **0/30** | n/a | 0.00 ± 0.00 | 45.00 ± 0.00 | 33.13 ± 3.44 |
| astar | 30/30 | 70.50 ± 4.63 | 9.17 ± 0.94 | 94.22 ± 6.02 | 97.50 ± 3.07 |
| bfs | 30/30 | 70.50 ± 4.63 | 9.17 ± 0.94 | 94.22 ± 6.02 | 97.50 ± 3.07 |

Two tasks, 45% starting charge, 1.5% per move. PPO trained for 600,000 steps
**never charged**: zero charging steps, and exactly 45.00 ± 0.00 energy used -
its entire starting charge - in every episode. Every episode ended in
`battery_depleted`.

The diagnosis is in the reward, not the learner. Every one of those failing
episodes still scored **+33** on average: the -10 depletion penalty was cheaper
than the credit a policy could bank first (one pickup +5, one delivery +20, and
roughly 27 of accumulated progress shaping). Under that reward, delivering one
package and dying is a correct maximisation, not a failure to learn.

### 6. `battery_tuned` - the fix

`data/results/battery_tuned_ppo_vs_baselines.md`

The same world as `battery_constrained`; only the reward and training budget
change:

| Change | Value | Reason |
|---|---|---|
| `battery_depleted_penalty` | 10 -> **60** | above everything an episode can bank before dying, so dying never pays |
| `battery_reserve_weight` | 0 -> 10 (safety 2.0) | potential-based term: zero while a charger is comfortably in reach, negative as the margin runs out |
| `wait_penalty` | 0.1 -> 0.5 | stops camping on a charger from being the safe option |
| training budget | 600k -> **2M** steps | the corrected reward was given a longer training run |

| Controller | Success | Steps | Path efficiency | Charging steps | Reward |
|---|---|---|---|---|---|
| ppo | **27/30** | 87.89 ± 4.90 | 0.73 ± 0.02 | 15.43 ± 0.90 | 82.66 ± 9.88 |
| astar | 30/30 | 70.50 ± 4.63 | 0.91 ± 0.01 | 9.17 ± 0.94 | 94.21 ± 2.87 |
| bfs | 30/30 | 70.50 ± 4.63 | 0.91 ± 0.01 | 9.17 ± 0.94 | 94.21 ± 2.87 |

PPO now charges and delivers both packages in 27/30 episodes (3 still end in
`battery_depleted`). It is **not** a win over the planner: it charges more than
it needs (15.4 vs 9.2 steps), drives longer routes (0.73 vs 0.91 path
efficiency) and is slower (87.9 vs 70.5 steps). The claim is that the battery
task became learnable once the reward stopped paying for death, not that RL
beats a battery-aware planner on it.

## Across training seeds

Every result above uses one trained policy (training seed 0). Each policy was
retrained with seeds 1 and 2 - the identical command, only `--seed` changed -
and evaluated on the same 30 episodes. Source: `data/seeds/summary.md`,
produced by `python scripts/run_seed_study.py`, which writes only to
`data/seeds/` and checks that `data/results` and `data/episodes` are unchanged.

| Scenario | Seed 0 (published) | Seed 1 | Seed 2 |
|---|---|---|---|
| `simple_static` | 30/30, 19.60 steps | 30/30, 19.60 | **28/30**, 27.64 |
| `default` | 30/30, 27.87 | 30/30, 27.87 | 30/30, 27.87 |
| `complex_static` | 30/30, 44.37 | 30/30, 44.37 | 30/30, 44.37 |
| `dynamic_obstacles` | 30/30, 28.93 steps, 0.47 collisions | 30/30, 29.37, 0.77 | 30/30, 29.07, 0.53 |
| `battery_constrained` (original reward) | 0/30 | 0/30 | 6/30 |
| `battery_tuned` (corrected reward) | 27/30 | 26/30 | **19/30** |

What holds on every seed:

* PPO is identical to A* on `default` and `complex_static` (every episode).
* On `dynamic_obstacles` PPO is faster than A* (28.93 / 29.37 / 29.07 vs 30.50
  steps) and collides on every seed (0.47 / 0.77 / 0.53 per episode). The
  trade-off is a property of the method, not of one run.
* The original battery reward fails: 0, 0 and 6 of 30.

What does not:

* `simple_static` seed 2 delivers 28/30 at path efficiency 0.78; in the two
  failures the robot wanders for the full 200 steps and runs flat. So "exactly
  optimal on static floors" is true for 8 of the 9 static runs, not all 9.
* `battery_tuned` gives 27, 26 and 19 of 30. Seed 1's four failures deliver
  one package and then run flat. Seed 2's ten failures are the camping local
  optimum: the robot sits on a charger for all 600 steps (600 charging steps,
  nothing delivered) and times out. The corrected reward delivers in 72 of 90
  episodes against 6 of 90 under the original reward - the fix is real - but
  how well it works depends on the seed.

## Reward-weight ablation

Which reward terms actually matter? Each arm retrains PPO on
`dynamic_obstacles` - the scenario where every term is active - with exactly
one term switched off (`configs/ablation/no_*.yaml`; each file differs from the
scenario in that one weight). Same world, 400k steps, hyper-parameters and
training seeds 0/1/2; evaluated on episodes 1000-1029. Arms are compared on
task metrics only, never on reward, because each arm is scored by a different
reward function. Source: `data/ablation/summary.md`, produced by
`python scripts/run_reward_studies.py --study ablation`.

| Switched off | Success /30 (seeds 0/1/2) | Mean steps | Collisions / episode |
|---|---|---|---|
| nothing (full reward) | 30 / 30 / 30 | 28.93 / 29.37 / 29.07 | 0.47 / 0.77 / 0.53 |
| progress shaping | **19 / 7 / 19** | 81.42 / 204.43 / 122.42 | 0.33 / 0.00 / 0.67 |
| collision penalty | 30 / 30 / 30 | 30.07 / 29.03 / 29.03 | **1.93 / 1.10 / 0.97** |
| energy cost | 30 / 30 / 30 | 29.10 / 29.90 / 29.17 | 0.50 / 0.63 / 0.43 |
| step cost | 30 / 30 / 30 | 29.00 / 29.07 / 29.33 | 0.53 / 0.60 / 0.47 |
| waiting penalty | 30 / 30 / 30 | 29.27 / 29.47 / 28.93 | 0.80 / 0.47 / 0.60 |

* **Progress shaping carries the learning.** Without it PPO succeeds in 45 of
  90 episodes instead of 90 of 90, and the successful ones take 3-7x as many
  steps (path efficiency 0.19-0.57 against 0.98). The sparse pickup and
  delivery rewards alone are not enough within 400k steps.
* **The collision penalty is what keeps collisions down.** Removing it leaves
  success untouched but raises collisions from 0.59 to 1.33 per episode
  (averaged over the three seeds), about 2.3x.
* **Energy, step and waiting terms show no measurable effect here.** Success,
  steps and collisions stay within the spread between seeds. On this scenario
  they are redundant with the progress term, which already rewards short
  paths. This does not show they are useless everywhere: the waiting penalty
  was raised to stop charger camping on the battery scenario, which this
  ablation does not test.

## Making the battery policy reliable

The seed study left `battery_tuned` at 27, 26 and 19 of 30. The training logs
(`data/logs/*/eval/evaluations.npz`) show why: every seed reached **100%
evaluation success at some checkpoint**, then swung (seed 1: 100% at 1.4M,
20% at 1.6M, 100% at 1.8M). The policy was learned; the run simply ended on a
worse moment for some seeds. Two remedies were measured on the same 30 test
episodes. Source: `data/stability/summary.md`
(`python scripts/run_reward_studies.py --study battery-lr`).

| Setting (2M steps, seeds 0/1/2) | Success /30 | Total | Outcomes of the failures |
|---|---|---|---|
| learning rate 3e-4, final policy (published) | 27 / 26 / 19 | 72/90 | 8 ran flat; 10 camped on a charger until timeout |
| learning rate 3e-4, **best checkpoint** | **30 / 27 / 25** | **82/90** | 8 ran flat; no camping |
| learning rate 1e-4, final policy | 6 / 30 / 0 | 36/90 | 24 ran flat; 30 timed out without delivering |
| learning rate 1e-4, best checkpoint | 5 / 30 / 0 | 35/90 | 25 ran flat; 30 timed out without delivering |

* **Keeping the best checkpoint works.** The checkpoint is chosen by the
  training run's own evaluation environment, whose episodes are drawn at random
  and are not the test episodes 1000-1029, so the choice does not peek at the
  test set. It lifts the total from 72 to 82 of 90, brings seed 0 to 30/30 -
  equal to A* - and removes camping on every seed.
* **A lower learning rate made it worse.** At 1e-4 learning is slower, and
  2M steps is not enough: seed 2 never succeeded in a single evaluation, seed 0
  peaked at 50%. Seed 1, the one seed that did learn (from about 1.4M steps),
  then stayed at 95-100% - so a lower rate may be steadier once learned, but
  at this budget it costs far more than it gains. The hypothesis is rejected
  for this budget; the recommended setting is 3e-4 with best-checkpoint
  selection.
* **Still below the planner.** A* and BFS deliver 30/30 on every seed.

## Multi-robot Live floor (demonstration, not a measured result)

The Live floor runs several robots on a shared task queue with the classical
battery-aware controller. No learned policy drives it, and its numbers depend
on how long a page is left running, so it is not part of the comparison above.
Its correctness is checked instead: `python scripts/verify_live_floor.py`
drives the fleet API exactly as the browser does across 8 scenarios (21x15
floor with idle gaps and bursts, the 46x30 robotic-fulfilment floor on four
seeds, battery rules, moving traffic) and proves, against an independent BFS,
that every product is delivered by the nearest idle robot standing on the
pickup and drop-off cells, with no two robots ever on one cell, no deaths and
no deadlocks. Current result: 8/8 scenarios, 112/112 products.

A wider stress sweep, `python scripts/stress_live_floor.py` (and `... rmfs`),
runs 3, 4, 5, 6 and 8 robots against a burst of products, steady arrivals and
long idle gaps on four seeds - 60 runs per floor. It found that the original
controller deadlocked or left a product undelivered in **14 of 60** runs on the
21x15 floor, in three distinct ways: idle robots parked in a wall around the
charger and station corner and never moved for anyone; a robot bouncing on
the "enough charge to finish?" threshold occupied the charger while others ran
flat; and robots queued to enter a charger blocked a full one trying to leave
it. Three rules fixed them - an idle robot with charge to spare (or one
head-on with the robot it blocks) steps aside; a robot below the low-battery
threshold is not given new work; and a robot that decides to charge charges to
90% before resuming. Now: **60/60 on the 21x15 floor and 60/60 on the 46x30
floor.**

### Task priorities

A product can be queued as **normal** or **urgent** (`priority` on
`POST /api/fleet/task`; a Normal/Urgent choice on the Live-floor card). Dispatch
serves every waiting urgent request before any normal one, first come first
served within each level. Priority decides who is served next; it never takes a
package off a robot already carrying one. The card reports mean turnaround per
priority, and a waiting urgent product is drawn with a ring around its marker.

Measured on the 21x15 floor, 4 robots, a backlog of 20 normal products, then 5
more products added 10 ticks later - once marked urgent, once marked normal,
same products, seeds 0-4:

| The 5 late products marked | Their mean turnaround | The 20-product backlog's mean turnaround |
|---|---|---|
| urgent | **57.7 ticks** | 116.6 ticks |
| normal | 194.0 ticks | **90.6 ticks** |

Urgent products arrive about 3.4x faster; the backlog waits about 29% longer
to pay for it. Priority moves waiting time, it does not create capacity. All 25
products were delivered on every seed.

One-way lanes on the 46x30 floor, 200-product backlog, 400 ticks, seeds
1000-1004 (`data/fleet/one_way.json`):

| Floor | Robots | Robot-on-robot refusals (mean) | Products delivered (mean) |
|---|---|---|---|
| one-way lanes | 12 | 27.4 | 54.6 |
| two-way | 12 | **24.0** | **58.0** |
| one-way lanes | 20 | 102.0 | 86.4 |
| two-way | 20 | **75.2** | **91.8** |

**One-way lanes do not reduce blocking on this floor, and they deliver about 6%
less.** This replaces an earlier claim of 5.8x less blocking, which was wrong in
how it counted: on the one-way floor a boxed-in robot waited silently, while on
the two-way floor the same robot attempted the move and was refused, so only the
two-way floor recorded its congestion. With both counted alike the gap is gone.
Two earlier figures (a hundredfold cut, then 2.5x) were withdrawn for other
reasons. Each time the controller changed, the measurement was rerun rather than
kept. These are 400-tick runs; long-run reliability of the two-way floor has not
been measured.

### Traffic and battery on the 46x30 floor

The battery rules were first refused on this floor because a full charge covers
67 moves against a median 66-move cycle. That was true but it was not the whole
problem. Over 200-product runs on the 200-move rules, only 5 of 10 runs delivered
everything, and 31 of 33 dead robots had stood still for at least 40 ticks: they
were jammed, and draining flat where they stood. Four causes, each traced on a
real run and each with a rule and a test in `tests/test_fleet_traffic.py`:

| Trap | What happened | Rule |
|---|---|---|
| Head-on jam | Two robots with tasks wanted each other's cells, and only idle robots gave way | A boxed-in robot attempts the move, and a working robot steps aside if head-on or stuck for 4 ticks |
| Charger queue | Six robots queued for one charger while six of eight stood empty | Each robot claims the nearest free charger it can reach |
| Dodging in step | Two robots in a two-wide corridor dodged to the same side for ever, each still moving | A robot about to reverse waits one tick if a lower-numbered robot is close |
| Shadowed exit | A robot beside another on a parallel lane found every exit occupied and was carried past each one until it ran flat | If a detour is over 2 moves longer than the route ignoring peers, try that route and queue |

`DEADLOCK_STREAK` only fires when the whole fleet stops, so none of this was
reported. A new warning, `stalled_robots`, names any working robot that has got
no closer to its target for 60 ticks, and the Live floor shows it.

`configs/fleet/battery_large_floor.yaml` then adds battery rules for this floor:
`battery_constrained` with its move and idle costs divided by 2.64, the ratio of
the two floors' median cycles (66 moves against 25), so the battery is as tight
here as it is on the small floor. It is a fleet-only ruleset with no PPO policy.

Held-out test, `python scripts/run_fleet_studies.py reliability`. 5, 10 and 20
robots; a burst of 40, a burst of 200, and a steady flow of 200 at about 80% of
capacity; 10 runs of each, 90 per ruleset, 1,050 robots. Seeds 2000-2009 played
no part in tuning: seeds 0-4 and 1000-1009 did (`data/fleet/reliability.json`).

| Rules | Runs delivering everything | Robots flat | Runs stopped early |
|---|---|---|---|
| `battery_tuned` (67 moves, 45% start), still refused in the app | **0 / 90** | 1,021 / 1,050 | 90 |
| `battery_large_floor` (175 moves, 45% start) | **90 / 90** | 0 / 1,050 | 0 |
| `default` (200 moves, 100% start) | **90 / 90** | 1 / 1,050 | 0 |

The one flat robot was in a 20-robot steady run, which still delivered all 200.
It was not chased, because tuning against these seeds would end their use as a
test. Eight chargers were enough, so the floor itself did not change. The 1,050
is simulated robots across simulated runs: none of this has been compared with a
real warehouse.

## Summary against the research question

| Question | Evidence |
|---|---|
| Can PPO match classical planning on static warehouse navigation? | Yes - identical to optimal A* on 30/30 episodes on all three static floors. |
| Does RL help when the warehouse is dynamic? | It is faster (28.93 vs 30.50 steps) but collides 0.47 times per episode where A* never does. No net reward advantage. |
| Does RL handle battery constraints? | Not under the original reward (0/30: dying paid; 6 of 90 across three seeds). After correcting the depletion penalty and training for 2M steps, 27/30 - and 72 of 90 across three seeds (27, 26, 19). Keeping the best checkpoint raises it to 82 of 90 (30, 27, 25) and removes camping. Learnable, seed-dependent, and still below the planner's 30/30. |
| Which reward terms matter? | Progress shaping (45 of 90 without it, against 90 of 90) and the collision penalty (2.3x more collisions without it). Energy, step and waiting terms had no measurable effect on `dynamic_obstacles`. |
| Is the comparison fair? | It is biased *against* RL: the planners get the full map and every obstacle position; PPO gets a 5x5 patch. |

## Threats to validity

1. **Three training seeds per scenario.** The headline tables use seed 0;
   seeds 1 and 2 are reported above. Three seeds show which findings are
   robust and which vary, but are too few for formal significance tests.
2. **Reward weights were not searched.** The weights are reasoned starting
   values; only the battery depletion penalty was changed, and it was derived
   from the measured reward of the failing policy rather than tuned. The
   ablation switches terms off one at a time on one scenario; it does not
   search values, and it does not test interactions between terms.
3. **Shaping is policy-invariant only for the undiscounted return.** Progress
   shaping uses gamma 1.0 while PPO optimises a 0.99-discounted return, and the
   potential is not zeroed at a failed terminal state. Ng et al.'s guarantee
   therefore holds exactly only for the undiscounted objective; under the
   discounted one the shaping can favour some trajectories over others.
   Correcting it requires retraining every policy.
4. **The obstacle model is benign.** Obstacles never step onto the robot, so
   collisions can only be caused by the controller. A harsher model would
   change the collision metric for every controller.
5. **A* is given an advantage.** Full map, full obstacle visibility, replanning
   every single step. This is the strongest reasonable classical baseline, so
   the comparison is conservative with respect to the RL side.
6. **The multi-robot floor is classical and unmeasured as a comparison.** It
   says nothing about multi-agent RL.
