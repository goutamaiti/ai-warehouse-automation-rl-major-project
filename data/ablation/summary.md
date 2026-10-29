# Reward-weight ablation (dynamic_obstacles)

Each arm retrains PPO with ONE reward term switched off (`configs/ablation/`),
same world, budget (400k steps) and hyper-parameters, training seeds 0/1/2,
evaluated on episodes 1000-1029. Values are per seed. Reward is not compared
across arms - each arm is scored by a different reward function.

| Arm | Seeds | Success /30 | Mean steps | Path efficiency | Collisions / episode |
|---|---|---|---|---|---|
| full reward (published + seed study) | 3 | 30 / 30 / 30 | 28.93 / 29.37 / 29.07 | 0.98 / 0.98 / 0.98 | 0.47 / 0.77 / 0.53 |
| without `progress` | 3 | 19 / 7 / 19 | 81.42 / 204.43 / 122.42 | 0.57 / 0.19 / 0.31 | 0.33 / 0.00 / 0.67 |
| without `collision` | 3 | 30 / 30 / 30 | 30.07 / 29.03 / 29.03 | 0.99 / 1.00 / 0.99 | 1.93 / 1.10 / 0.97 |
| without `energy` | 3 | 30 / 30 / 30 | 29.10 / 29.90 / 29.17 | 0.98 / 0.96 / 0.97 | 0.50 / 0.63 / 0.43 |
| without `step` | 3 | 30 / 30 / 30 | 29.00 / 29.07 / 29.33 | 0.98 / 0.98 / 0.97 | 0.53 / 0.60 / 0.47 |
| without `wait` | 3 | 30 / 30 / 30 | 29.27 / 29.47 / 28.93 | 0.98 / 0.97 / 0.98 | 0.80 / 0.47 / 0.60 |
