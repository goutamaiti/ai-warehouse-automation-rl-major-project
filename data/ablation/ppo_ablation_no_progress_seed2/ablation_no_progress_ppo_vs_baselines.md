# Evaluation: ablation_no_progress

- Generated: 2026-10-29T20:48:40+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 19/30 | 0.63 | 122.42 ± 37.16 | 121.74 ± 37.23 | 0.31 ± 0.09 | 0.67 ± 0.47 | 105.25 ± 27.84 | 33.70 ± 19.90 | 0.01 ± 8.35 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_progress_seed2.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
