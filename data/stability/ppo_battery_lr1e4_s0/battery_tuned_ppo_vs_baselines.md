# Evaluation: battery_tuned

- Generated: 2026-10-29T21:27:47+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 6/30 | 0.20 | 76.50 ± 7.69 | 69.00 ± 6.24 | 0.71 ± 0.08 | 2.03 ± 3.99 | 56.85 ± 8.80 | 2.67 ± 1.98 | -2.88 ± 17.30 |

## Notes

- PPO model: rl_agent\models\ppo_battery_lr1e4_s0.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
