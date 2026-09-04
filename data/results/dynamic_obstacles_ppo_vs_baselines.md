# Evaluation: dynamic_obstacles

- Generated: 2026-08-20T12:31:27+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 28.93 ± 2.55 | 28.47 ± 2.45 | 0.98 ± 0.01 | 0.47 ± 0.34 | 14.26 ± 1.23 | 0.00 ± 0.00 | 49.67 ± 2.22 |
| astar | 30/30 | 1.00 | 30.50 ± 3.12 | 30.33 ± 3.11 | 0.93 ± 0.03 | 0.00 ± 0.00 | 15.18 ± 1.55 | 0.03 ± 0.07 | 50.02 ± 2.26 |
| bfs | 30/30 | 1.00 | 31.57 ± 3.59 | 31.33 ± 3.58 | 0.91 ± 0.03 | 0.00 ± 0.00 | 15.68 ± 1.79 | 0.03 ± 0.07 | 49.95 ± 2.23 |
| random | 3/30 | 0.10 | 404.67 ± 101.59 | 226.33 ± 52.32 | 0.11 ± 0.01 | 85.17 ± 6.64 | 124.49 ± 8.68 | 12.60 ± 4.20 | -110.01 ± 8.70 |

## Notes

- PPO model: rl_agent/models/ppo_dynamic.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
