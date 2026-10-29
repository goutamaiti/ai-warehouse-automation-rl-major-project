# Evaluation: simple_static

- Generated: 2026-10-29T15:44:01+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 19.60 ± 1.84 | 19.60 ± 1.84 | 1.00 ± 0.00 | 0.00 ± 0.00 | 9.80 ± 0.92 | 0.00 ± 0.00 | 42.42 ± 1.73 |

## Notes

- PPO model: rl_agent\models\ppo_simple_seed1.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
