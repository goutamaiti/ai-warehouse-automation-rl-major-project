# Evaluation: ablation_no_collision

- Generated: 2026-10-29T20:51:08+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 30.07 ± 3.03 | 28.13 ± 2.54 | 0.99 ± 0.01 | 1.93 ± 0.94 | 14.16 ± 1.29 | 0.07 ± 0.13 | 50.08 ± 2.26 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_collision_seed0.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
