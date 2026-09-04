# data/

Output of real simulation runs. Nothing in this directory is written by hand.

| Path | Written by | Contents |
|---|---|---|
| `episodes/*.json` | `--record N` on either evaluation script | Full step-by-step replay: layout, one frame per step, summary. The dashboard animates these. |
| `results/*.json` | `analytics/reports.py` | Per-episode metrics plus aggregated summaries, with the seed list and a timestamp. |
| `results/*.md` | `analytics/reports.py` | The same summaries as a markdown comparison table. |
| `logs/` | `rl_agent/train.py` | Monitor CSVs, TensorBoard traces, `run_metadata.json`. Git-ignored (regenerate by training). |

Regenerate everything with:

```bash
python scripts/run_experiments.py
```

File names follow `<scenario>_<controller>_seed<seed>.json` for replays and
`<scenario>_<suite>.json` for result files.
