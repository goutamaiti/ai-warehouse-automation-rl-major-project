# Evaluation: battery_tuned

- Generated: 2026-10-29T21:39:35+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 78.07 ± 4.47 | 68.93 ± 4.00 | 0.83 ± 0.04 | 9.13 ± 0.53 | 104.31 ± 6.05 | 10.23 ± 0.49 | 88.61 ± 2.96 |

## Notes

- PPO model: rl_agent\models\ppo_battery_lr1e4_s1.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
