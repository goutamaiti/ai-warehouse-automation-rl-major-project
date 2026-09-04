# Evaluation: default

- Generated: 2026-08-20T12:31:19+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 27.87 ± 2.43 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 ± 0.00 | 13.93 ± 1.21 | 0.00 ± 0.00 | 50.19 ± 2.28 |
| astar | 30/30 | 1.00 | 27.87 ± 2.43 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 ± 0.00 | 13.93 ± 1.21 | 0.00 ± 0.00 | 50.19 ± 2.28 |
| bfs | 30/30 | 1.00 | 27.87 ± 2.43 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 ± 0.00 | 13.93 ± 1.21 | 0.00 ± 0.00 | 50.19 ± 2.28 |
| random | 1/30 | 0.03 | 242.00 | 145.00 | 0.13 | 74.73 ± 6.86 | 112.16 ± 5.02 | 9.83 ± 3.60 | -99.92 ± 9.08 |

## Notes

- PPO model: rl_agent/models/ppo_default.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
