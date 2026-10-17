import { useCallback, useEffect, useRef, useState } from 'react';

import FleetCanvas from './FleetCanvas';
import {
  addProduct,
  advanceFleet,
  backendConfigured,
  fleetState,
  startFleet,
  stopFleet,
  streamFleet,
} from '../lib/fleetApi';

/**
 * The live warehouse floor: several robots, a shared queue, and a card for
 * adding a product by its coordinates.
 *
 * This page is explicitly a *demonstration*, not a measurement, and says so -
 * its numbers depend on how long it has been left running and when someone
 * added a product. The measured results live on the other tab and come from
 * fixed seeds through the offline pipeline.
 */

const FLOORS = [
  { id: 'default', label: 'Scenario floor · 21×15' },
  { id: 'rmfs', label: 'Robotic fulfilment floor · 46×30' },
];

const SCENARIOS = [
  { id: 'default', label: 'Default floor' },
  { id: 'dynamic_obstacles', label: 'With moving traffic' },
  { id: 'battery_tuned', label: 'Battery constrained' },
];

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {hint ? <span className="stat-hint">{hint}</span> : null}
    </div>
  );
}

export default function FleetPage() {
  const [layout, setLayout] = useState(null);
  const [frame, setFrame] = useState(null);
  const [running, setRunning] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const [nRobots, setNRobots] = useState(5);
  const [scenario, setScenario] = useState('default');
  const [floor, setFloor] = useState('default');
  const [pickup, setPickup] = useState({ row: '', col: '' });
  const [dropoff, setDropoff] = useState({ row: '', col: '' });
  const [lastAdded, setLastAdded] = useState(null);

  const closeStream = useRef(null);

  const stopStream = useCallback(() => {
    closeStream.current?.();
    closeStream.current = null;
    setPlaying(false);
  }, []);

  useEffect(() => stopStream, [stopStream]);

  const guard = async (work) => {
    setBusy(true);
    setError(null);
    try {
      return await work();
    } catch (problem) {
      setError(problem.message);
      return null;
    } finally {
      setBusy(false);
    }
  };

  const onStart = () =>
    guard(async () => {
      stopStream();
      const payload = await startFleet({
        scenario,
        nRobots: Number(nRobots),
        controller: 'astar',
        seed: 1000,
        floor,
      });
      setLayout(payload.layout);
      setFrame(payload.state);
      setRunning(true);
      setLastAdded(null);
    });

  const onStop = () =>
    guard(async () => {
      stopStream();
      await stopFleet();
      setRunning(false);
      setFrame(null);
      setLayout(null);
    });

  const onPlay = () => {
    if (playing) {
      stopStream();
      return;
    }
    setError(null);
    setPlaying(true);
    closeStream.current = streamFleet({
      tickSeconds: 0.12,
      onFrame: setFrame,
      onError: (problem) => {
        setError(problem.message);
        stopStream();
      },
    });
  };

  const onStep = (ticks) =>
    guard(async () => {
      const next = await advanceFleet(ticks);
      setFrame(next);
    });

  const onAddProduct = (event) => {
    event.preventDefault();
    return guard(async () => {
      const result = await addProduct({
        pickup: [Number(pickup.row), Number(pickup.col)],
        dropoff: [Number(dropoff.row), Number(dropoff.col)],
      });
      setLastAdded(result.request);
      // Show it immediately rather than waiting for the next frame, so adding
      // a product visibly does something even while paused.
      const next = await fleetState();
      setFrame(next);
    });
  };

  const coordsFilled =
    pickup.row !== '' && pickup.col !== '' && dropoff.row !== '' && dropoff.col !== '';
  const highlight =
    pickup.row !== '' && pickup.col !== ''
      ? [Number(pickup.row), Number(pickup.col)]
      : null;

  const fleet = frame?.fleet;
  const queue = frame?.queue;

  if (!backendConfigured) {
    return (
      <section className="floor">
        <div className="floor-empty">
          <h2>The live floor needs the backend</h2>
          <p>
            This page drives a warehouse that is actually running, so it cannot work
            from the static replays the results tab uses. Start the API with{' '}
            <code>uvicorn backend.main:app --reload</code> and set{' '}
            <code>VITE_API_BASE</code> in <code>frontend/.env.local</code>.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="floor">
      <header className="floor-header">
        <div>
          <h2>Live floor</h2>
          <p className="floor-sub">
            Several robots working a shared queue. Add a product and the nearest free
            robot is assigned on the next tick.
          </p>
        </div>
        <span className="floor-badge">demonstration, not a measured result</span>
      </header>

      <div className="floor-controls">
        <label>
          Floor
          <select
            value={floor}
            onChange={(event) => setFloor(event.target.value)}
            disabled={running}
          >
            {FLOORS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Rules
          <select
            value={scenario}
            onChange={(event) => setScenario(event.target.value)}
            disabled={running}
          >
            {SCENARIOS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Robots
          <input
            type="number"
            min="1"
            value={nRobots}
            onChange={(event) => setNRobots(event.target.value)}
            disabled={running}
          />
        </label>
        {running ? (
          <>
            <button type="button" className="primary" onClick={onPlay}>
              {playing ? 'Pause' : 'Play'}
            </button>
            <button type="button" onClick={() => onStep(1)} disabled={playing || busy}>
              Step
            </button>
            <button type="button" onClick={() => onStep(25)} disabled={playing || busy}>
              Step 25
            </button>
            <button type="button" onClick={onStop} disabled={busy}>
              Stop
            </button>
          </>
        ) : (
          <button type="button" className="primary" onClick={onStart} disabled={busy}>
            Start the floor
          </button>
        )}
      </div>

      {error ? <p className="floor-error">{error}</p> : null}
      {frame?.stopped ? (
        <p className="floor-error">
          The floor stopped: {frame.stopped_reason}. Press Start to run it again.
        </p>
      ) : null}

      {running && layout ? (
        <div className="floor-body">
          <div className="floor-main">
            <FleetCanvas layout={layout} frame={frame} highlight={highlight} />
            <ul className="floor-legend">
              <li><span className="dot" style={{ background: '#1d4ed8' }} /> robot</li>
              <li><span className="dot" style={{ background: '#b45309' }} /> carrying</li>
              <li><span className="dot" style={{ background: '#9ca3af' }} /> out of charge</li>
              <li><span className="dot" style={{ background: '#0891b2' }} /> product waiting</li>
              <li><span className="dot" style={{ background: '#be123c' }} /> traffic</li>
            </ul>
          </div>

          <aside className="floor-side">
            <form className="card" onSubmit={onAddProduct}>
              <h3>Add a product</h3>
              <p className="card-hint">
                Read a cell off the rulers along the top and left of the map, or
                click one below. Blue cells are storage points, green ones packing
                stations.
              </p>

              <div className="coord-row">
                <span className="coord-label">Pick up</span>
                <input
                  type="number" min="0" placeholder="row" aria-label="pickup row"
                  value={pickup.row}
                  onChange={(event) => setPickup({ ...pickup, row: event.target.value })}
                />
                <input
                  type="number" min="0" placeholder="col" aria-label="pickup column"
                  value={pickup.col}
                  onChange={(event) => setPickup({ ...pickup, col: event.target.value })}
                />
              </div>

              <div className="coord-row">
                <span className="coord-label">Deliver to</span>
                <input
                  type="number" min="0" placeholder="row" aria-label="dropoff row"
                  value={dropoff.row}
                  onChange={(event) => setDropoff({ ...dropoff, row: event.target.value })}
                />
                <input
                  type="number" min="0" placeholder="col" aria-label="dropoff column"
                  value={dropoff.col}
                  onChange={(event) => setDropoff({ ...dropoff, col: event.target.value })}
                />
              </div>

              <button type="submit" className="primary" disabled={!coordsFilled || busy}>
                Add to the queue
              </button>

              {lastAdded ? (
                <p className="card-note">
                  Request #{lastAdded.id} queued from [{lastAdded.pickup.join(', ')}] to [
                  {lastAdded.dropoff.join(', ')}].
                </p>
              ) : null}

              {/* Always visible, never behind a disclosure. The coordinates
                  are the one thing a person cannot guess, so hiding the valid
                  ones made the card unusable - click to fill, or read them off
                  the rulers on the map. */}
              <div className="card-choices">
                <p className="choice-group">
                  <strong>
                    Pick up from
                    {layout.pickup_total > (layout.pickup_choices ?? []).length
                      ? ` · ${(layout.pickup_choices ?? []).length} of ${layout.pickup_total}`
                      : ''}
                  </strong>
                  {(layout.pickup_choices ?? []).map((cell) => {
                    const active =
                      String(cell[0]) === pickup.row && String(cell[1]) === pickup.col;
                    return (
                      <button
                        key={cell.join(',')}
                        type="button"
                        className={`chip${active ? ' chip-active' : ''}`}
                        onClick={() =>
                          setPickup({ row: String(cell[0]), col: String(cell[1]) })
                        }
                      >
                        {cell[0]}, {cell[1]}
                      </button>
                    );
                  })}
                </p>
                <p className="choice-group">
                  <strong>Deliver to</strong>
                  {(layout.dropoff_choices ?? []).map((cell) => {
                    const active =
                      String(cell[0]) === dropoff.row && String(cell[1]) === dropoff.col;
                    return (
                      <button
                        key={cell.join(',')}
                        type="button"
                        className={`chip${active ? ' chip-active' : ''}`}
                        onClick={() =>
                          setDropoff({ row: String(cell[0]), col: String(cell[1]) })
                        }
                      >
                        {cell[0]}, {cell[1]}
                      </button>
                    );
                  })}
                </p>
              </div>
            </form>

            <div className="card">
              <h3>The floor right now</h3>
              <div className="stat-grid">
                <Stat label="Tick" value={frame?.ticks ?? 0} />
                <Stat
                  label="Robots"
                  value={`${fleet?.robots_alive ?? 0}/${frame?.n_robots ?? 0}`}
                  hint="still charged"
                />
                <Stat label="Delivered" value={fleet?.tasks_delivered ?? 0} />
                <Stat label="Waiting" value={queue?.waiting ?? 0} hint="unassigned" />
                <Stat
                  label="Turnaround"
                  value={queue?.mean_turnaround ?? '—'}
                  hint="ticks, request to delivery"
                />
                <Stat
                  label="Blocked"
                  value={fleet?.blocked_by_robot ?? 0}
                  hint="refused by a peer"
                />
              </div>
              {frame?.idle ? (
                <p className="card-note">
                  Caught up — every request delivered. The floor is idle, not stuck.
                </p>
              ) : null}
            </div>

            {frame?.dispatch?.length ? (
              <div className="card">
                <h3>Just assigned</h3>
                <ul className="assign-list">
                  {frame.dispatch.map((decision) => (
                    <li key={decision.request_id}>
                      request #{decision.request_id} → robot {decision.robot_id}
                      <span className="muted"> ({decision.distance} cells away)</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </aside>
        </div>
      ) : (
        !error && (
          <div className="floor-empty">
            <h2>Nothing is running yet</h2>
            <p>
              Choose a floor and a fleet size, then start it. The warehouse keeps running
              once started — a delivery does not reset it.
            </p>
          </div>
        )
      )}
    </section>
  );
}
