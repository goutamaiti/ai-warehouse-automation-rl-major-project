# Evaluation: ablation_no_progress

- Generated: 2026-10-29T20:43:11+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 19/30 | 0.63 | 81.42 ± 32.55 | 81.11 ± 32.44 | 0.57 ± 0.16 | 0.33 ± 0.33 | 73.23 ± 22.00 | 15.93 ± 17.32 | 3.36 ± 8.04 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_progress_seed0.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
