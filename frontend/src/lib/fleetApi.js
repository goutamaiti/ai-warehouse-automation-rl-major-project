/**
 * Client for the live multi-robot warehouse (`/api/fleet/*`).
 *
 * Kept apart from lib/api.js on purpose. That module serves the *measured*
 * project - the recorded replays and the committed result tables, which work
 * with no backend at all because they are static files. These calls drive an
 * interactive floor and need a running backend, and the numbers they return
 * depend on how long a page was open and when someone added a product. Every
 * frame carries a `provenance` field saying so, and the UI shows it.
 */

import { API_BASE, backendConfigured } from './api';

export { backendConfigured };

async function request(path, options) {
  const response = await fetch(`${API_BASE}/api/fleet${path}`, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    // The backend phrases these for a person - "[5, 3] is a shelf cell; a robot
    // cannot stand there" - so surface its wording rather than a status code.
    throw new Error(payload?.detail ?? `${response.status} ${response.statusText}`);
  }
  return payload;
}

const json = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export function fleetHealth() {
  return request('/health');
}

/** Start a warehouse running. Replaces one already running. */
export function startFleet({ scenario, nRobots, controller, seed, floor }) {
  return request(
    '/start',
    json({
      scenario,
      n_robots: nRobots,
      controller,
      seed,
      floor,
    })
  );
}

export function stopFleet() {
  return request('/stop', { method: 'POST' });
}

export function fleetState() {
  return request('/state');
}

/** Queue a product. The nearest idle robot takes it on the next tick;
 *  an urgent one is served before every waiting normal one. */
export function addProduct({ pickup, dropoff, priority = 'normal' }) {
  return request('/task', json({ pickup, dropoff, priority }));
}

/** Step the warehouse on without a live clock - used by the Step button. */
export function advanceFleet(ticks = 1) {
  return request('/advance', json({ ticks }));
}

/**
 * Subscribe to the live frame stream.
 *
 * Returns a function that closes the connection. The stream *is* the
 * warehouse clock on the server, so closing it pauses the floor rather than
 * leaving a loop running behind the page.
 */
export function streamFleet({ tickSeconds = 0.1, onFrame, onError }) {
  const url = `${API_BASE}/api/fleet/stream?tick_seconds=${tickSeconds}`;
  const source = new EventSource(url);

  source.onmessage = (event) => {
    try {
      onFrame?.(JSON.parse(event.data));
    } catch (error) {
      onError?.(error);
    }
  };
  source.onerror = () => {
    // EventSource retries by itself, which would silently restart the
    // warehouse clock. Close it and let the page decide instead.
    source.close();
    onError?.(new Error('the live stream dropped; press Play to resume'));
  };

  return () => source.close();
}
