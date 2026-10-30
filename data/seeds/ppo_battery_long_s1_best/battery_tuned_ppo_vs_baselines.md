# Evaluation: battery_tuned

- Generated: 2026-10-29T20:38:15+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 27/30 | 0.90 | 75.70 ± 5.09 | 65.59 ± 4.89 | 0.85 ± 0.04 | 0.00 ± 0.00 | 99.57 ± 6.60 | 12.63 ± 1.54 | 84.34 ± 7.92 |

## Notes

- PPO model: rl_agent/models/ppo_battery_long_s1_best/best_model.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
