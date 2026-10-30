# Final demo - checklist against the coordinator's guideline

Slides: `presentation/Final_Project_Demo.pptx` in the OneDrive copy of the
project, with a PDF beside it (6 slides, in the order the guideline asks for). Screenshots on slide 4 are real captures of the running
portal. The Demo-2 run sheet (`docs/demo-script.md`) still applies for the live
sequence and the viva questions.

## Presenters

| Slide | Content | Presenter |
|---|---|---|
| 1 | Title, team, USNs, guide | all three |
| 2 | Problem statement and objectives | Goutam A Iti |
| 3 | System design and architecture | Goutam A Iti |
| 4 | Results and discussion | Prerana C Patil |
| 5 | Conclusion | Srushti Bingikoppad |
| 6 | Work progress and report status | Srushti Bingikoppad |

## Part B - what to show for each point

| Guideline | What to do | Evidence |
|---|---|---|
| 1. All major modules | Dashboard (replay PPO, A*, BFS), Live floor (add products, one urgent), Editor (draw, run), Info (method) | four tabs of the portal |
| 2. End-to-end execution | Editor: draw a floor -> *Run episode* -> watch the robot deliver; Live floor: add a product -> robot assigned -> delivered | `POST /api/run`, `POST /api/fleet/task` |
| 3. Validation and testing | `python -m pytest` | 276 passed |
| 4. Test cases, expected vs actual | `python scripts/run_test_cases.py` (backend running) | `docs/test-cases.md`, 27/27 |
| 5. Input validation, error and exception handling | Show V1-V13 and E1-E4 in `docs/test-cases.md`; live: type a wall cell or a cell outside the floor on the Live-floor card and press *Add to the queue* | readable 400/404/422 messages, never a 500 |
| 6. Runs without interruption | `.\start_demo.ps1 -Verify` the night before | 8/8 live-floor scenarios, 540/540 episodes |
| 7. Source, data, software ready | GitHub repo, tags `demo1` and `demo2`; `requirements.txt`, `frontend/package.json`; results in `data/` | |

No database is used: results, replays and models are files in `data/` and
`rl_agent/models/`, committed to the repository.

## Before the demo

```bash
git status                              # must be clean
python -m pytest -q                     # 276 passed
.\start_demo.ps1 -Verify                # starts backend + portal after verifying
python scripts/run_test_cases.py        # with the backend running: 27/27
```

`run_test_cases.py` replaces the running Live-floor session; run it before
starting the live part of the demo, not during it.
