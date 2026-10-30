# Evaluation: ablation_no_energy

- Generated: 2026-10-29T21:00:36+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.90 ± 2.82 | 29.27 ± 2.65 | 0.96 ± 0.03 | 0.63 ± 0.43 | 14.66 ± 1.33 | 0.00 ± 0.00 | 49.74 ± 2.26 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_energy_seed1.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
