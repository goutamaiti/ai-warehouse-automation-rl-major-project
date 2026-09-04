# Evaluation: simple_static

- Generated: 2026-08-20T12:31:09+00:00
- Episodes per controller: 30 (seeds 1000-1029)

| Controller | Episodes | Success rate | Steps | Path length | Path efficiency | Collisions | Energy % | Charging events | Reward |
|---|---|---|---|---|---|---|---|---|---|
| astar | 30/30 | 1.00 | 19.60 ± 1.84 | 19.60 ± 1.84 | 1.00 ± 0.00 | 0.00 ± 0.00 | 9.80 ± 0.92 | 0.00 ± 0.00 | 42.42 ± 1.73 |
| bfs | 30/30 | 1.00 | 19.60 ± 1.84 | 19.60 ± 1.84 | 1.00 ± 0.00 | 0.00 ± 0.00 | 9.80 ± 0.92 | 0.00 ± 0.00 | 42.42 ± 1.73 |
| random | 2/30 | 0.07 | 114.00 ± 154.84 | 69.00 ± 88.20 | 0.25 ± 0.32 | 43.53 ± 5.13 | 60.58 ± 3.84 | 10.63 ± 2.97 | -55.33 ± 7.54 |

## Notes

- Classical controllers see the full map and all obstacle positions; the RL policy only sees its observation vector.
- Path efficiency is optimal-path-length / driven-path-length, averaged over successful episodes only.
