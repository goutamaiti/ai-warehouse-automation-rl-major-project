# Evaluation: battery_constrained

- Generated: 2026-10-29T15:53:44+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 0/30 | 0.00 | n/a | n/a | n/a | 0.00 ± 0.00 | 45.00 ± 0.00 | 0.00 ± 0.00 | 33.13 ± 3.44 |

## Notes

- PPO model: rl_agent\models\ppo_battery_seed1.zip (deterministic=True).
- All controllers ran on identical episode seeds.
- The classical controllers see the full map; PPO sees only its observation vector.
