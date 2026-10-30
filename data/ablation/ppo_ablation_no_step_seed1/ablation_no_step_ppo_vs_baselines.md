# Evaluation: ablation_no_step

- Generated: 2026-10-29T21:07:43+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.07 ± 2.67 | 28.47 ± 2.47 | 0.98 ± 0.01 | 0.60 ± 0.41 | 14.26 ± 1.24 | 0.00 ± 0.00 | 50.98 ± 2.24 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_step_seed1.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
