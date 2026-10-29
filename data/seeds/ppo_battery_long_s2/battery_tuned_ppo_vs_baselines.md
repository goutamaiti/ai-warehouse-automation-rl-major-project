# Evaluation: battery_tuned

- Generated: 2026-10-29T16:30:26+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 19/30 | 0.63 | 65.95 ± 4.22 | 57.42 ± 3.44 | 0.90 ± 0.04 | 0.00 ± 0.00 | 78.46 ± 5.84 | 206.53 ± 101.27 | -54.74 ± 72.59 |

## Notes

- PPO model: rl_agent\models\ppo_battery_long_s2.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
