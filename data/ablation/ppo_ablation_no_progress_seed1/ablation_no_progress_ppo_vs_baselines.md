# Evaluation: ablation_no_progress

- Generated: 2026-10-29T20:46:02+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 7/30 | 0.23 | 204.43 ± 78.68 | 204.43 ± 78.68 | 0.19 ± 0.13 | 0.00 ± 0.00 | 215.52 ± 24.34 | 119.13 ± 27.76 | -23.70 ± 7.39 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_progress_seed1.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
