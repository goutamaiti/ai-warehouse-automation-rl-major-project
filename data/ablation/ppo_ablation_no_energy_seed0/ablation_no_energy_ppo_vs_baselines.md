# Evaluation: ablation_no_energy

- Generated: 2026-10-29T20:58:18+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.10 ± 2.56 | 28.60 ± 2.49 | 0.98 ± 0.02 | 0.50 ± 0.37 | 14.32 ± 1.25 | 0.00 ± 0.00 | 49.91 ± 2.28 |

## Notes

- PPO model: rl_agent\models\ppo_ablation_no_energy_seed0.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
