# Evaluation: dynamic_obstacles

- Generated: 2026-10-29T15:50:23+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 30/30 | 1.00 | 29.37 ± 2.69 | 28.60 ± 2.51 | 0.98 ± 0.02 | 0.77 ± 0.46 | 14.34 ± 1.26 | 0.00 ± 0.00 | 49.34 ± 2.20 |

## Notes

- PPO model: rl_agent\models\ppo_dynamic_seed1.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
