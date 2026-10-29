# Evaluation: battery_tuned

- Generated: 2026-10-29T21:52:50+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 0/30 | 0.00 | n/a | n/a | n/a | 0.00 ± 0.00 | 900.00 ± 0.00 | 249.90 ± 31.47 | -49.73 ± 1.81 |

## Notes

- PPO model: rl_agent\models\ppo_battery_lr1e4_s2_best\best_model.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
