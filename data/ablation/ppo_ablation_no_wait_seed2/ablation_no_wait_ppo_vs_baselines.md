# Evaluation: ablation_no_wait

- Generated: 2026-10-29T21:17:23+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 28.93 ± 2.55 | 28.33 ± 2.49 | 0.98 ± 0.01 | 0.60 ± 0.38 | 14.20 ± 1.25 | 0.00 ± 0.00 | 49.54 ± 2.27 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_wait_seed2.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
