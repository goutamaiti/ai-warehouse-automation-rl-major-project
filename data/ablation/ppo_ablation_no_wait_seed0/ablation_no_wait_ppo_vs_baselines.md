# Evaluation: ablation_no_wait

- Generated: 2026-10-29T21:12:31+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.27 ± 2.75 | 28.47 ± 2.58 | 0.98 ± 0.01 | 0.80 ± 0.43 | 14.27 ± 1.30 | 0.00 ± 0.00 | 49.32 ± 2.19 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_wait_seed0.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
