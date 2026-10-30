# Evaluation: ablation_no_energy

- Generated: 2026-10-29T21:02:59+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.17 ± 2.66 | 28.73 ± 2.63 | 0.97 ± 0.02 | 0.43 ± 0.36 | 14.39 ± 1.32 | 0.00 ± 0.00 | 49.98 ± 2.30 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_energy_seed2.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
