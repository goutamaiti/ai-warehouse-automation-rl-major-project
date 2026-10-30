# Evaluation: complex_static

- Generated: 2026-10-17T08:38:22+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 44.37 ± 3.18 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 ± 0.00 | 22.18 ± 1.59 | 0.00 ± 0.00 | 65.70 ± 2.99 |
| astar | 30/30 | 1.00 | 44.37 ± 3.18 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 ± 0.00 | 22.18 ± 1.59 | 0.00 ± 0.00 | 65.70 ± 2.99 |
| bfs | 30/30 | 1.00 | 44.37 ± 3.18 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 ± 0.00 | 22.18 ± 1.59 | 0.03 ± 0.07 | 65.70 ± 2.99 |

## Notes

- PPO model: rl_agent/models/ppo_complex.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
