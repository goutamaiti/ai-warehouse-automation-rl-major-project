# Evaluation: simple_static

- Generated: 2026-10-29T16:09:10+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 28/30 | 0.93 | 27.64 ± 4.56 | 27.64 ± 4.56 | 0.78 ± 0.07 | 0.00 ± 0.00 | 19.57 ± 8.11 | 0.00 ± 0.00 | 38.12 ± 5.43 |

## Notes

- PPO model: rl_agent\models\ppo_simple_seed2.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
