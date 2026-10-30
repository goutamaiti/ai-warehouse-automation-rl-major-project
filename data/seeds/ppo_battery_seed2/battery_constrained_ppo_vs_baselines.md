# Evaluation: battery_constrained

- Generated: 2026-10-29T16:17:52+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 6/30 | 0.20 | 69.50 ± 6.97 | 61.67 ± 5.42 | 0.80 ± 0.08 | 0.00 ± 0.00 | 54.66 ± 7.20 | 2.10 ± 1.58 | 42.66 ± 9.09 |

## Notes

- PPO model: rl_agent\models\ppo_battery_seed2.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
