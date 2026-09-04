# Deployment

The dashboard and the Python side deploy separately, because they have very
different needs: the dashboard is a static site, while training needs a real
machine with a CPU and no request timeout.

## What runs where

| Part | Where | Why |
|---|---|---|
| Dashboard (`frontend/`) | Vercel (static) | Plain Vite build, no server needed |
| Recorded episodes and results | Bundled into the site at build time | The demo works with no backend at all |
| FastAPI backend (`backend/`) | Optional, any host that runs Python | Only needed for *live* runs from the browser |
| PPO training (`rl_agent/train.py`) | Local machine | Minutes of CPU per run; never triggered over HTTP |

## Vercel

The repository root contains `vercel.json`:

```json
{
  "installCommand": "npm install --prefix frontend",
  "buildCommand": "npm run build --prefix frontend",
  "outputDirectory": "frontend/dist"
}
```

Import the repository in Vercel and deploy - keep **Root Directory** at the
repository root so the build can read `data/` (the `prebuild` step copies the
episode replays and result files into `frontend/public/data`). No environment
variable is required.

If you set the Root Directory to `frontend` instead, the build cannot see
`data/` and the dashboard will come up empty.

### Optional: connect a live backend

Deploy `backend/` somewhere that runs Python (Render, Railway, Fly.io, a VM)
and then, in Vercel's project settings, add:

```text
VITE_API_BASE = https://your-backend-host
```

Redeploy. The dashboard will detect the backend, show "Backend connected" and
enable the "Run a new episode" form on the Dashboard tab plus the full Editor
tab: PPO, moving obstacles, and running a hand-drawn warehouse through the
real Python simulation (`POST /api/run` accepts an optional `layout` grid that
overrides the named scenario's own map). Without a backend, the Editor still
works for A*, BFS and Random through an in-browser sandbox - see
`frontend/src/lib/offlineSimulation.js` - but PPO and moving obstacles are
disabled there. On the backend set:

```text
ALLOWED_ORIGINS = https://your-dashboard.vercel.app
```

so CORS accepts the dashboard's origin.

The backend is *not* deployed as a Vercel serverless function on purpose: a
live episode plus a PPO forward pass per step does not fit the model of a short
serverless invocation, and the trained policy would have to be loaded on every
cold start.

## Running everything locally

```bash
# Python side
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# API on http://127.0.0.1:8000
uvicorn backend.main:app --reload

# Dashboard on http://localhost:5173
npm install --prefix frontend
npm run dev --prefix frontend
```

To let the local dashboard talk to the local API, create
`frontend/.env.local`:

```text
VITE_API_BASE=http://127.0.0.1:8000
```

## Regenerating the bundled data

The dashboard only ever shows files from `data/`. To refresh them:

```bash
python scripts/run_experiments.py          # full: trains 3 policies, then evaluates
python scripts/run_experiments.py --skip-training   # evaluate with existing models
```

Then rebuild the frontend (the `prebuild` hook re-copies the data).
