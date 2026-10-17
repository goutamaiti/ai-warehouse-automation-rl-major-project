import { useState } from 'react';

import { controllerRank, scenarioAdds, scenarioRank, scenarioTier } from '../lib/complexity';
import { controllerLabel, titleCase } from '../lib/format';

/**
 * The measured comparison: one metric at a time, every scenario at once.
 *
 * What this replaces, and why
 * ---------------------------
 * Six stacked cards, each an eight-column bordered table, is a data dump: 144
 * numbers with no indication which of them carries an argument. Worse, the
 * "±" that makes a difference meaningful or meaningless was hidden inside a
 * cell, so a reader could not see that A* and PPO overlap on one scenario and
 * separate on another.
 *
 * So: pick a metric, see all six scenarios as bars on a shared scale, with the
 * 95% interval drawn as a bracket. The full numeric grid is still one click
 * away - nothing is removed, it is just no longer the first thing shown.
 *
 * Three-way ties are stated, not drawn
 * ------------------------------------
 * On the three static scenarios all three controllers are numerically
 * identical - 19.60, 27.87 and 44.37 steps, the same to the last digit -
 * because a policy on a 5x5 local view reproduces full-map optimal planning
 * there. Drawn as three coincident bars that reads as a broken chart; it is in
 * fact the project's strongest result, so it is collapsed into one statement.
 */

const METRICS = [
  { key: 'path_efficiency', label: 'Path efficiency', better: 'high', fmt: (v) => v.toFixed(2) },
  { key: 'steps', label: 'Steps', better: 'low', fmt: (v) => v.toFixed(1) },
  { key: 'success_rate', label: 'Success', better: 'high', fmt: (v) => v.toFixed(2), flat: true },
  { key: 'collisions', label: 'Collisions', better: 'low', fmt: (v) => v.toFixed(2) },
  { key: 'energy_consumed', label: 'Energy', better: 'low', fmt: (v) => v.toFixed(1) },
  { key: 'charging_events', label: 'Charging', better: 'low', fmt: (v) => v.toFixed(2) },
  { key: 'total_reward', label: 'Reward', better: 'high', fmt: (v) => v.toFixed(1) },
];

/** Controllers are told apart by tint, never by hue - there is no second colour. */
const TINT = { astar: 'var(--s4)', bfs: 'var(--s3)', ppo: 'var(--s6)' };

function read(summary, metric) {
  if (!summary) return null;
  if (metric.flat) {
    const value = summary[metric.key];
    return typeof value === 'number' ? { mean: value, ci95: 0, n: summary.n_episodes } : null;
  }
  const entry = summary[metric.key];
  if (!entry || typeof entry.mean !== 'number') return null;
  return { mean: entry.mean, ci95: entry.ci95 ?? 0, n: entry.n ?? 0 };
}

export default function ResultsTable({ results }) {
  const [metricKey, setMetricKey] = useState(METRICS[0].key);
  const [showGrid, setShowGrid] = useState(false);

  if (!results?.length) {
    return (
      <section className="results">
        <h2>Measured results</h2>
        <p className="muted">
          No result files found. Generate them with{' '}
          <code>python scripts/run_experiments.py --skip-training</code>.
        </p>
      </section>
    );
  }

  const metric = METRICS.find((m) => m.key === metricKey) ?? METRICS[0];
  const rows = [...results].sort((a, b) => scenarioRank(a.scenario) - scenarioRank(b.scenario));

  // One scale across every scenario, so bar lengths are comparable down the
  // column rather than only within a row.
  let peak = 0;
  rows.forEach((row) => {
    (row.summaries ?? []).forEach((summary) => {
      const value = read(summary, metric);
      if (value) peak = Math.max(peak, value.mean + value.ci95);
    });
  });
  if (peak <= 0) peak = 1;

  return (
    <section className="results">
      <div className="results-head">
        <h2>Measured results</h2>
        <p className="muted small">
          30 episodes per controller, seeds 1000–1029, every controller on the same
          episodes. Shaded bar is the mean; the bracket is the 95% interval.
        </p>
      </div>

      <div className="metric-pick">
        {METRICS.map((item) => (
          <button
            key={item.key}
            type="button"
            className={`pick${item.key === metric.key ? ' active' : ''}`}
            onClick={() => setMetricKey(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="compare">
        {rows.map((row) => {
          const summaries = [...(row.summaries ?? [])].sort(
            (a, b) => controllerRank(a.controller) - controllerRank(b.controller)
          );
          const values = summaries.map((s) => ({ controller: s.controller, v: read(s, metric) }));
          const present = values.filter((x) => x.v);
          // Collapse only when EVERY controller is present and agrees. A row
          // where one has no qualifying episode must stay as bars: on
          // battery_constrained A* and BFS tie while PPO is n/a because it
          // delivered 0 of 30, and folding that into "A* = BFS" would delete
          // the project's single most important battery finding from the view.
          const tied =
            present.length === values.length &&
            present.length > 2 &&
            present.every((x) => Math.abs(x.v.mean - present[0].v.mean) < 1e-9);

          return (
            <article className="compare-row" key={row.id}>
              <header>
                <h3>
                  {scenarioTier(row.scenario) ? `${scenarioTier(row.scenario)}. ` : ''}
                  {titleCase(row.scenario)}
                </h3>
                <p className="muted small">{scenarioAdds(row.scenario)}</p>
              </header>

              {tied ? (
                /* The tie IS the result. Three coincident bars would read as a
                   broken chart; one statement reads as a finding. */
                <div className="tied">
                  <span className="tied-bar" style={{ width: `${(present[0].v.mean / peak) * 100}%` }} />
                  <b className="num">{metric.fmt(present[0].v.mean)}</b>
                  <em>
                    {present.map((x) => controllerLabel(x.controller).split(' ')[0]).join(' = ')} —
                    identical on {metric.label.toLowerCase()}
                  </em>
                </div>
              ) : (
                <div className="bars">
                  {values.map(({ controller, v }) => (
                    <div className="bar-row" key={controller}>
                      <span className="bar-name">
                        {controller === 'astar' ? 'A*' : controller === 'bfs' ? 'BFS' : 'PPO'}
                      </span>
                      <span className="bar-track">
                        {v ? (
                          <>
                            <i style={{ width: `${(v.mean / peak) * 100}%`, background: TINT[controller] }} />
                            {v.ci95 > 0 && (
                              <u
                                style={{
                                  left: `${(Math.max(v.mean - v.ci95, 0) / peak) * 100}%`,
                                  width: `${(Math.min(2 * v.ci95, peak) / peak) * 100}%`,
                                }}
                              />
                            )}
                          </>
                        ) : (
                          /* No qualifying episode. A hatch rather than an empty
                             row, because "nothing succeeded" is a result. */
                          <i className="none" />
                        )}
                      </span>
                      <span className="bar-value num">
                        {v ? metric.fmt(v.mean) : 'n/a'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </article>
          );
        })}
      </div>

      <button type="button" className="pick reveal" onClick={() => setShowGrid((v) => !v)}>
        {showGrid ? 'Hide the full numbers' : 'Show the full numbers'}
      </button>

      {showGrid && (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Controller</th>
                <th>Episodes</th>
                <th>{metric.label}</th>
                <th>± 95%</th>
              </tr>
            </thead>
            <tbody>
              {rows.flatMap((row) =>
                [...(row.summaries ?? [])]
                  .sort((a, b) => controllerRank(a.controller) - controllerRank(b.controller))
                  .map((summary) => {
                    const v = read(summary, metric);
                    return (
                      <tr key={`${row.id}-${summary.controller}`}>
                        <td>{titleCase(row.scenario)}</td>
                        <td>{controllerLabel(summary.controller)}</td>
                        <td>
                          {summary.n_successful}/{summary.n_episodes}
                        </td>
                        <td>{v ? metric.fmt(v.mean) : 'n/a'}</td>
                        <td>{v && v.ci95 ? v.ci95.toFixed(2) : '—'}</td>
                      </tr>
                    );
                  })
              )}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
