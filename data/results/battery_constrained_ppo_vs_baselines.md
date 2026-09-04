# Evaluation: battery_constrained

- Generated: 2026-08-20T12:31:34+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 0/30 | 0.00 | n/a | n/a | n/a | 0.00 ± 0.00 | 45.00 ± 0.00 | 0.00 ± 0.00 | 33.13 ± 3.44 |
| astar | 30/30 | 1.00 | 70.50 ± 4.63 | 62.27 ± 3.97 | 0.91 ± 0.01 | 0.00 ± 0.00 | 94.22 ± 6.02 | 9.17 ± 0.94 | 97.50 ± 3.07 |
| bfs | 30/30 | 1.00 | 70.50 ± 4.63 | 62.27 ± 3.97 | 0.91 ± 0.01 | 0.00 ± 0.00 | 94.22 ± 6.02 | 9.17 ± 0.94 | 97.50 ± 3.07 |
| random | 0/30 | 0.00 | n/a | n/a | n/a | 16.43 ± 3.87 | 90.96 ± 14.50 | 6.27 ± 2.54 | -30.68 ± 5.20 |

## Notes

- PPO model: rl_agent/models/ppo_battery.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
