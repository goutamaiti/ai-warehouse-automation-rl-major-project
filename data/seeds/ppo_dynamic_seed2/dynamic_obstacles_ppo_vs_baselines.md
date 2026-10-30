# Evaluation: dynamic_obstacles

- Generated: 2026-10-29T16:14:42+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.07 ± 2.58 | 28.53 ± 2.51 | 0.98 ± 0.02 | 0.53 ± 0.37 | 14.29 ± 1.26 | 0.00 ± 0.00 | 49.59 ± 2.25 |

## Notes

- PPO model: rl_agent\models\ppo_dynamic_seed2.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
