# Evaluation: battery_tuned

- Generated: 2026-10-17T08:39:04+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 27/30 | 0.90 | 87.89 ± 4.90 | 77.22 ± 4.41 | 0.73 ± 0.02 | 0.00 ± 0.00 | 119.85 ± 6.80 | 15.43 ± 0.90 | 82.66 ± 9.88 |
| astar | 30/30 | 1.00 | 70.50 ± 4.63 | 62.27 ± 3.97 | 0.91 ± 0.01 | 0.00 ± 0.00 | 94.22 ± 6.02 | 9.17 ± 0.94 | 94.21 ± 2.87 |
| bfs | 30/30 | 1.00 | 70.50 ± 4.63 | 62.27 ± 3.97 | 0.91 ± 0.01 | 0.00 ± 0.00 | 94.22 ± 6.02 | 9.17 ± 0.94 | 94.21 ± 2.87 |

## Notes

- PPO model: rl_agent/models/ppo_battery_long_s0.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
