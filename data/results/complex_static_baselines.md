# Evaluation: complex_static

- Generated: 2026-08-20T12:31:12+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| astar | 30/30 | 1.00 | 44.37 ± 3.18 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 ± 0.00 | 22.18 ± 1.59 | 0.00 ± 0.00 | 65.70 ± 2.99 |
| bfs | 30/30 | 1.00 | 44.37 ± 3.18 | 44.37 ± 3.18 | 1.00 ± 0.00 | 0.00 ± 0.00 | 22.18 ± 1.59 | 0.03 ± 0.07 | 65.70 ± 2.99 |
| random | 0/30 | 0.00 | n/a | n/a | n/a | 97.57 ± 14.37 | 128.60 ± 11.13 | 12.83 ± 4.69 | -127.03 ± 16.29 |

## Notes

- Classical controllers see the full map and all obstacle positions; the RL policy only sees its observation vector.
- Path efficiency is optimal-path-length / driven-path-length, averaged over successful episodes only.
