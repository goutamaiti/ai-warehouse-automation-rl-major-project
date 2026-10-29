# Demo 2 - run sheet

Slides: `presentation/Second_Project_Demo.pptx` (21 slides). The live part is
slide 15's sequence, about 6 minutes. Every number below is in `data/results/`,
`data/training/` or the output of the two verify scripts.

## The night before

```bash
git status                                # must be clean - evaluation overwrites data/
python scripts/verify_live_floor.py       # expect: 8/8 scenarios passed
python scripts/verify_single_robot.py     # expect: 540/540 ... 0 PROBLEM(S)
```

Both write nothing. If either fails, do not demo that part; ask before changing
anything.

## On the day

1. Close every other copy of the project (`L:\MJ2\...`, the OneDrive copy) and
   any terminal running `uvicorn` or `npm run dev`.
2. Run `.\start_demo.ps1` from the repository root. It refuses to start if
   port 8000 or 5190 is busy and tells you which process to stop.
3. In the browser the badge at the top right must read **Backend connected**.
   If it reads anything else, the API is not reachable: stop both processes
   and run the launcher again.
4. Set browser zoom to 100%, and full-screen the window (F11).

## The sequence (slide 15)

| # | Who | Click | Say | If it goes wrong |
|---|---|---|---|---|
| 1 | Goutam | **Dashboard** -> scenario *2. Default* -> **PPO**, Play. Then **A\***. | Identical shortest path (31 steps on this seed; 27.87 averaged over 30); 30/30 for all three on every static floor. PPO sees a 5x5 patch, A* the whole map. | Recordings are static files; this works even with no backend. |
| 2 | Goutam | Scenario *4. Dynamic obstacles*, **PPO** then **A\***. | PPO 28.93 steps vs A* 30.50, but 0.47 collisions vs 0. Total reward tied - the reward prices the trade-off. | Same as 1. |
| 3 | Goutam / Prerana | **Info** tab -> reward walkthrough -> *5. Battery constrained*, drag the scrubber to the last step; then *5. Battery tuned*. | Before: the robot dies and still has **+21.60** cumulative reward - dying paid. After: penalty 60, it charges and delivers. 0/30 -> 27/30. | Static data; always works. |
| 4 | Prerana | **Live floor** -> floor *Robotic fulfilment 46x30* -> **Start the floor** -> **Play**. Wait a few seconds. Click a **Pick up** chip and a **Deliver to** chip, then **Add to the queue**. Add a few more, then one more with **Urgent** selected. | The nearest free robot by path distance, not straight line, takes it on the next tick. The urgent one jumps the queue - read *Urgent / normal* turnaround on the card. Watch it reach the pickup and the station. Classical control - no RL here. | If the floor shows *stopped*, press **Start the floor** again. If the badge is not *Backend connected*, skip to step 5. |
| 5 | Srushti | **Dashboard** -> scroll to *Measured results*, then *Learning curves*; hover the battery curve. | Battery stays under 50% success for over a million steps and first reaches 90% at 1.275M. A 600k-step run would have stopped on the plateau. | Static data; always works. |

## Questions the panel is likely to ask

**"Does PPO beat A*?"** On static floors it matches it exactly - A* is provably
optimal there, so matching is the ceiling. In traffic PPO is faster but collides;
reward is tied. On the battery task PPO learns (27/30) but A* is better (30/30,
fewer charging steps). We do not claim PPO wins.

**"Why did PPO fail at 0/30?"** The reward made dying profitable: the -10
depletion penalty was cheaper than the pickup, delivery and progress credit
banked first, so failing episodes still averaged +33. The policy was maximising
correctly. Raising the penalty to 60 fixed the incentive.

**"Your shaping is potential-based - is it really policy-invariant?"** Exactly
so only for the undiscounted return. We use shaping gamma 1.0 (gamma below 1 paid
the robot to loiter) while PPO discounts at 0.99, and the potential is not zeroed
when an episode fails. Measured example: 0, 20 or 60 waits give +38.0 shaping
undiscounted but 31.6 / 25.9 / 17.3 discounted. Fixing it means retraining every
policy - it is listed as future scope, not hidden (`docs/rl-formulation.md`).

**"Is the multi-robot floor reinforcement learning?"** No. It is classical
battery-aware A* with a shared queue and nearest-robot dispatch. No learned
policy has seen a second robot; multi-agent RL is future scope.

**"How do you know the Live floor works?"** `verify_live_floor.py`: 8 scenarios,
112/112 products delivered, each by the nearest idle robot (checked against an
independent BFS), each robot standing on its pickup and drop-off, no two robots
on one cell, no deaths, no deadlocks. Then `stress_live_floor.py`: 3-8 robots,
bursts, steady arrivals and idle gaps, four seeds - 60 runs on each floor, all
passing. Those checks found five real bugs before the demo (a false deadlock
after an idle spell, a charging loop, parked robots walling off a charger,
robots bouncing on the charge threshold, and a jammed charger doorway) - all
fixed and covered by regression tests. Before the fixes, 14 of 60 stress runs
on the 21x15 floor failed.

**"Do one-way lanes help?"** They cut robot-on-robot blocking about 5.8x
(17.6 vs 102.4 over 5 seeds) but deliver about 3% less (50.6 vs 52.4) because of the
detours. A trade-off. An earlier "hundredfold" figure was measured with the
charging-loop bug and was withdrawn.

**"Is this one lucky training run?"** No - every policy was retrained with two
more seeds (`data/seeds/summary.md`). The static ties with A* and the traffic
trade-off hold on all three. The battery fix holds on all three but varies: 27,
26 and 19 of 30 (72 of 90, against 6 of 90 under the original reward). Seed 2
fell into camping on a charger in 10 episodes - say so if asked.

**"Can the battery policy be made reliable?"** Partly. Every seed hit 100% at
some point in training and then swung, so we keep the best checkpoint (chosen
on random validation episodes, never the test ones): 72 -> 82 of 90, seed 0 at
30/30 like A*, no camping. A lower learning rate was tried and was worse
(36/90 - two seeds learned too slowly), so we report it and did not adopt it.

**"Which reward terms matter?"** We ran an ablation: retrain with one term off,
3 seeds each, compare on success and collisions. Without progress shaping PPO
succeeds in 45 of 90 instead of 90; without the collision penalty collisions go
up 2.3x. Energy, step and waiting terms made no measurable difference on the
moving-traffic scenario.

**"Where did Random go?"** Removed after Demo 1 to keep the comparison to the
controllers that matter. It solved almost nothing, so it carried no information.

**"Why PPO and not DQN?"** Dense, shaped reward and a small discrete action
space: PPO's clipped on-policy updates are stable there and need no replay
buffer. Q-Learning/DQN validation was planned and skipped deliberately.

**"How do priorities work?"** Two levels, normal and urgent. Every waiting
urgent product is dispatched before any normal one; within a level, first come
first served. A robot already carrying a package is never reassigned. Measured:
5 late products took 58.3 ticks marked urgent against 187.1 marked normal, and
the rest of the backlog waited about 42% longer to pay for it.
