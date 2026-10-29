# Evaluation: battery_tuned

- Generated: 2026-10-29T21:27:55+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 5/30 | 0.17 | 75.80 ± 9.55 | 68.60 ± 8.53 | 0.74 ± 0.06 | 2.03 ± 3.99 | 57.77 ± 9.69 | 2.57 ± 1.94 | -12.50 ± 21.09 |

## Notes

- PPO model: rl_agent\models\ppo_battery_lr1e4_s0_best\best_model.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
