# Evaluation: battery_tuned

- Generated: 2026-10-29T16:06:01+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 26/30 | 0.87 | 76.04 ± 5.28 | 66.00 ± 5.20 | 0.85 ± 0.04 | 0.00 ± 0.00 | 100.00 ± 6.75 | 12.23 ± 1.61 | 81.52 ± 9.91 |

## Notes

- PPO model: rl_agent\models\ppo_battery_long_s1.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
