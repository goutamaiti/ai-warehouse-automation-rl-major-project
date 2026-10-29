# Evaluation: battery_tuned

- Generated: 2026-10-29T21:39:43+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 79.23 ± 4.62 | 70.00 ± 4.13 | 0.82 ± 0.04 | 9.23 ± 0.55 | 105.92 ± 6.24 | 10.30 ± 0.52 | 88.42 ± 2.99 |

## Notes

- PPO model: rl_agent\models\ppo_battery_lr1e4_s1_best\best_model.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
