# Evaluation: ablation_no_step

- Generated: 2026-10-29T21:05:21+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.00 ± 2.57 | 28.47 ± 2.49 | 0.98 ± 0.01 | 0.53 ± 0.37 | 14.26 ± 1.25 | 0.00 ± 0.00 | 51.05 ± 2.37 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_step_seed0.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
