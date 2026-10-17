# Evaluation: default

- Generated: 2026-10-17T08:38:03+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 27.87 ± 2.43 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 ± 0.00 | 13.93 ± 1.21 | 0.00 ± 0.00 | 50.19 ± 2.28 |
| astar | 30/30 | 1.00 | 27.87 ± 2.43 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 ± 0.00 | 13.93 ± 1.21 | 0.00 ± 0.00 | 50.19 ± 2.28 |
| bfs | 30/30 | 1.00 | 27.87 ± 2.43 | 27.87 ± 2.43 | 1.00 ± 0.00 | 0.00 ± 0.00 | 13.93 ± 1.21 | 0.00 ± 0.00 | 50.19 ± 2.28 |

## Notes

- PPO model: rl_agent/models/ppo_default.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
