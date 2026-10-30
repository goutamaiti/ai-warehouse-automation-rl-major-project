# Evaluation: ablation_no_collision

- Generated: 2026-10-29T20:55:59+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.03 ± 2.49 | 28.07 ± 2.46 | 0.99 ± 0.01 | 0.97 ± 0.47 | 14.08 ± 1.23 | 0.07 ± 0.13 | 50.13 ± 2.28 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_collision_seed2.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
