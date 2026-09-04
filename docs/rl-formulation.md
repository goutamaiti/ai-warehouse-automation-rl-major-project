# RL formulation

This document is the reference for the viva: it states the MDP precisely and
explains *why* each design choice was made.

## The MDP

| Element | Definition here |
|---|---|
| Agent | One virtual warehouse robot |
| Environment | A 2D grid warehouse with shelves, stations and moving traffic |
| State (internal) | Robot cell, battery, payload flag, task status, obstacle cells, step count |
| Observation | 37-dimensional vector (below) - strictly less than the full state |
| Action | 5 discrete actions: up, down, left, right, wait |
| Reward | Weighted sum of the terms below |
| Episode end | All tasks delivered (success), battery empty (failure), or step budget exhausted (truncation) |
| Discount | gamma = 0.99 during PPO training |

The problem is *partially observable*: the agent sees a 5x5 patch around itself
plus a handful of task/battery scalars, not the whole map. This is deliberate -
it is the assumption a real robot with local sensing would work under, and it
is what makes the comparison against a full-information A* planner meaningful.

## Observation vector (37 values)

| Index | Feature | Range | Why it is there |
|---|---|---|---|
| 0-1 | Robot row, column (normalised) | [0, 1] | Absolute position in the warehouse |
| 2-3 | Target row, column (normalised) | [0, 1] | Where the current leg ends |
| 4-5 | Target offset (target - robot) | [-1, 1] | Direction to drive, the most useful single cue |
| 6 | Shortest-path distance to target | [0, 1] | Distance *around* shelves, not through them |
| 7 | Battery level / capacity | [0, 1] | Needed for any battery-aware behaviour |
| 8 | Carrying a package | {0, 1} | Distinguishes the pickup leg from the delivery leg |
| 9-10 | Offset to the nearest charger | [-1, 1] | Lets the policy find a charger without a map |
| 11 | Shortest-path distance to that charger | [0, 1] | Together with 7, enough to decide "charge now?" |
| 12-36 | 5x5 local occupancy patch | {0, 0.5, 1} | 0 free, 0.5 dynamic obstacle, 1 wall/shelf/out of bounds |

`WarehouseEnv.observation_labels()` returns exactly this list, so any index can
be named at run time.

Everything is normalised into [-1, 1] so no feature dominates the input scale
of the network. The distance normaliser is a constant computed once from the
static map (the longest station-to-cell distance), never a per-episode value -
otherwise the same physical distance would mean different numbers in different
episodes.

## Action space

`0 up, 1 down, 2 left, 3 right, 4 wait`.

Pickup, delivery and charging happen **automatically** when the robot stands on
the right cell. Explicit actions for them were considered and rejected: they
would enlarge the action space by 60% without changing the navigation problem
that this project studies, and they introduce a failure mode ("robot on the
right cell that never presses the button") that is not interesting research.

`wait` is kept because it is genuinely useful - it is the correct action when a
moving obstacle blocks the only aisle, and it is how the robot charges.

## Reward function

```text
r = - step_penalty
    + progress_weight * (gamma_s * Phi(s') - Phi(s))     [potential shaping]
    + pickup_reward        if a package was picked up
    + delivery_reward      if a package was delivered
    - collision_penalty    if the robot drove into a wall, shelf or obstacle
    - wait_penalty         if the action was "wait"
    - energy_weight * energy_used_this_step
    - battery_depleted_penalty   if the battery hit zero
    - timeout_penalty            if the step budget ran out
```

Default weights (`configs/default.yaml`): step 0.05, progress 1.0, pickup 5,
delivery 20, collision 1.0, wait 0.1, energy 0.02, battery 10, timeout 5.

**These are starting values chosen by reasoning, not the output of a
hyper-parameter search.** Any claim that they are good must come from an
ablation experiment that is actually run.

### Potential-based shaping, and the trap in it

The progress term uses `Phi(s) = -shortest_path_distance(s, target)`. Ng,
Harada and Russell (1999) proved that a shaping term of the form
`F(s, s') = gamma * Phi(s') - Phi(s)` does not change the optimal policy, which
is why this form was chosen over an ad-hoc "reward for getting closer".

Two traps were found and fixed while building this, both worth being able to
explain:

1. **`gamma_s < 1` pays the robot to loiter.** If the robot does not move,
   `F = gamma*(-d) - (-d) = d * (1 - gamma)`. With `d = 20` and
   `gamma = 0.99` that is `+0.2` per step, which is *more* than the 0.05 step
   penalty: standing far away from the goal becomes profitable. The project
   therefore uses `shaping_gamma = 1.0`, so standing still is worth exactly 0
   and the step penalty dominates. Regression test:
   `tests/test_env.py::test_waiting_far_from_the_goal_is_never_profitable`.
2. **A goal that jumps breaks the telescoping sum.** When the target changes
   (pickup completed, or a new task is generated) the potentials of the old and
   the new goal are not comparable, so the shaping term is skipped on that
   step. For the same reason a finished task keeps reporting its drop-off as
   the target instead of falling back to the pickup point - otherwise the
   delivery step would be charged for "moving away" from the goal it just
   reached, cancelling the delivery reward. Regression test:
   `tests/test_env.py::test_delivery_step_is_not_punished_by_the_shaping_term`.

### Reward-hacking checklist

| Question | Answer in this design |
|---|---|
| Can the agent farm the progress term by oscillating? | No: shaping telescopes, a step towards and a step back sum to 0, and each step costs `step_penalty`. |
| Can it profit from standing still? | No, see trap 1 above. |
| Can it profit from colliding? | No: a collision costs reward and makes no progress. |
| Can it avoid the timeout penalty by dying early? | Battery depletion costs more (10) than a timeout (5). |
| Is charging exploitable? | Charging gives no reward of its own; it is only worth it because it prevents the depletion penalty. |

## Algorithms

| Algorithm | Role |
|---|---|
| **PPO** | The main learned policy. Actor-critic, clipped objective, MLP (128, 128) over the 37-value observation. |
| A* | Classical baseline with full map knowledge, replans every step around visible obstacles. |
| BFS | Uninformed baseline; same path length as A* on a uniform-cost grid, used to measure what the heuristic saves. |
| Random | Sanity floor: shows the task is not solvable by chance. |

PPO was chosen over DQN because the action space is small but the reward is
dense and shaped: PPO's on-policy updates are stable with dense rewards, need
no replay buffer, and Stable-Baselines3's implementation is well documented,
which matters for a project that has to be explained rather than just run.

## What PPO cannot be expected to beat

A* is optimal on a static grid with full knowledge. On the static scenarios the
honest expectation is that PPO *matches* it at best. The interesting question
is what happens when the world moves and the planner's assumptions weaken - see
`docs/experiments.md` for what was actually measured.
