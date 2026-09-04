/**
 * Data access for the dashboard.
 *
 * Two sources, in this order:
 *  1. Static replays and result files copied out of ../data at build time.
 *     This is what the deployed (Vercel) site uses and it always works.
 *  2. The FastAPI backend, used only when VITE_API_BASE is configured. It adds
 *     live episode runs on top of the static data.
 */

const RAW_BASE = import.meta.env.VITE_API_BASE ?? '';
export const API_BASE = RAW_BASE.replace(/\/+$/, '');
export const backendConfigured = API_BASE.length > 0;

async function getJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${url}`);
  }
  return response.json();
}

/** Index of every replay and result file bundled with the site. */
export function loadIndex() {
  return getJson('data/index.json');
}

export function loadEpisode(id) {
  return getJson(`data/episodes/${encodeURIComponent(id)}.json`);
}

export function loadResult(id) {
  return getJson(`data/results/${encodeURIComponent(id)}.json`);
}

/** Returns the backend health payload, or null when no backend is reachable. */
export async function checkBackend() {
  if (!backendConfigured) return null;
  try {
    return await getJson(`${API_BASE}/api/health`);
  } catch {
    return null;
  }
}

export function loadScenarios() {
  return getJson(`${API_BASE}/api/scenarios`);
}

/**
 * Ask the backend to run one episode now and return its recording.
 * `layout`, when given, overrides the named scenario's own map with a
 * user-drawn grid of CellType integers - every other rule (battery, reward,
 * obstacles, step budget) still comes from `scenario`.
 */
export function runLiveEpisode({ scenario, controller, seed, layout }) {
  return getJson(`${API_BASE}/api/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario, controller, seed, layout: layout ?? null }),
  });
}
