# Evaluation: battery_tuned

- Generated: 2026-10-29T20:38:06+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 89.37 ± 4.50 | 78.73 ± 4.02 | 0.72 ± 0.02 | 0.00 ± 0.00 | 119.16 ± 6.06 | 15.93 ± 0.60 | 91.57 ± 2.83 |

## Notes

- PPO model: rl_agent/models/ppo_battery_long_s0_best/best_model.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
