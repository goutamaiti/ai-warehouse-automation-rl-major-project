# Evaluation: battery_tuned

- Generated: 2026-10-29T20:38:25+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 25/30 | 0.83 | 71.44 ± 4.54 | 63.72 ± 4.18 | 0.86 ± 0.04 | 0.00 ± 0.00 | 97.14 ± 5.27 | 10.10 ± 1.43 | 79.99 ± 9.79 |

## Notes

- PPO model: rl_agent/models/ppo_battery_long_s2_best/best_model.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
