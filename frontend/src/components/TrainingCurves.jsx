import { useState } from 'react';

import { scenarioRank, scenarioTier } from '../lib/complexity';
import { titleCase } from '../lib/format';

/**
 * PPO learning curves, from the training runs whose logs were kept.
 *
 * Every point is one evaluation Stable-Baselines3 actually ran during training:
 * the EvalCallback stops every 25,000 steps and plays 20 deterministic
 * episodes on a separate evaluation environment. scripts/export_training_curves.py
 * copies those evaluations out of the git-ignored data/logs/ unchanged - no
 * smoothing, no resampling - so the line goes exactly where the measurements are.
 *
 * Two charts per run, never one chart with two y-axes: success rate and reward
 * are different quantities on different scales, and a shared plot with a second
 * axis invites reading the crossing point as meaningful when it is an artefact
 * of where the two scales were pinned.
 *
 * Three policies have no curve because their logs were not retained when they
 * were trained. That is said, not papered over.
 */

const W = 320;
const H = 150;
const PAD = { top: 12, right: 12, bottom: 26, left: 40 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

/** Policies that were trained and evaluated but whose training logs are gone. */
const NO_LOG = [
  { scenario: 'default', tag: 'ppo_default' },
  { scenario: 'dynamic_obstacles', tag: 'ppo_dynamic' },
  { scenario: 'battery_constrained', tag: 'ppo_battery' },
];

function compact(steps) {
  if (steps === 0) return '0';
  if (steps >= 1_000_000) return `${(steps / 1_000_000).toFixed(steps % 1_000_000 ? 1 : 0)}M`;
  return `${Math.round(steps / 1000)}k`;
}

function niceTicks(min, max, count = 4) {
  const span = max - min || 1;
  const raw = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((m) => span / m <= count) ?? raw;
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 0.001; v += step) ticks.push(Number(v.toFixed(6)));
  // Only ticks inside the plotted range: one below it was drawn under the
  // x-axis labels.
  return ticks.filter((v) => v >= min - step * 0.001 && v <= max + step * 0.001);
}

function Chart({ title, points, value, band, domain, format, unit }) {
  const [hover, setHover] = useState(null);
  const maxX = points[points.length - 1].timesteps;
  const x = (t) => PAD.left + (t / maxX) * PLOT_W;
  const [lo, hi] = domain;
  const y = (v) => PAD.top + PLOT_H - ((v - lo) / (hi - lo || 1)) * PLOT_H;
  const clampY = (v) => y(Math.max(lo, Math.min(hi, v)));

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.timesteps).toFixed(1)},${clampY(value(p)).toFixed(1)}`).join('');
  const area = band
    ? points.map((p, i) => `${i ? 'L' : 'M'}${x(p.timesteps).toFixed(1)},${clampY(band(p)[1]).toFixed(1)}`).join('') +
      [...points].reverse().map((p) => `L${x(p.timesteps).toFixed(1)},${clampY(band(p)[0]).toFixed(1)}`).join('') +
      'Z'
    : null;
  const yTicks = niceTicks(lo, hi);
  const xTicks = [0, maxX / 2, maxX];

  const onMove = (event) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const t = ((px - PAD.left) / PLOT_W) * maxX;
    let best = 0;
    points.forEach((p, i) => {
      if (Math.abs(p.timesteps - t) < Math.abs(points[best].timesteps - t)) best = i;
    });
    setHover(best);
  };

  const hp = hover === null ? null : points[hover];

  return (
    <figure className="curve">
      <figcaption>{title}</figcaption>
      <div className="curve-plot">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`${title}: ${format(value(points[0]))} at ${compact(points[0].timesteps)} steps to ${format(value(points[points.length - 1]))} at ${compact(maxX)} steps`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {yTicks.map((v) => (
            <g key={v}>
              <line className="curve-grid" x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)} />
              <text className="curve-tick" x={PAD.left - 6} y={y(v)} textAnchor="end" dominantBaseline="middle">
                {Number.isInteger(v) ? v : format(v)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} className="curve-tick" x={x(t)} y={H - 8} textAnchor={t === 0 ? 'start' : t === maxX ? 'end' : 'middle'}>
              {compact(t)}
            </text>
          ))}
          {area && <path className="curve-band" d={area} />}
          <path className="curve-line" d={line} />
          {hp && (
            <g>
              <line className="curve-cross" x1={x(hp.timesteps)} x2={x(hp.timesteps)} y1={PAD.top} y2={PAD.top + PLOT_H} />
              <circle className="curve-dot" cx={x(hp.timesteps)} cy={clampY(value(hp))} r="4" />
            </g>
          )}
          {/* full-height hit area so the crosshair answers anywhere in the plot */}
          <rect x={PAD.left} y={PAD.top} width={PLOT_W} height={PLOT_H} fill="transparent" />
        </svg>
        {hp && (
          <div
            className="curve-tip"
            style={{ left: `${(x(hp.timesteps) / W) * 100}%` }}
          >
            <b className="num">{format(value(hp))}{unit}</b>
            <span>{hp.timesteps.toLocaleString()} steps</span>
            {band && <span>± {format(hp.reward_std)} (1 SD)</span>}
          </div>
        )}
      </div>
    </figure>
  );
}

export default function TrainingCurves({ training }) {
  const [showTable, setShowTable] = useState(false);
  const runs = [...(training ?? [])].sort((a, b) => scenarioRank(a.scenario) - scenarioRank(b.scenario));
  const logged = new Set(runs.map((r) => r.scenario));
  const missing = NO_LOG.filter((m) => !logged.has(m.scenario));

  if (!runs.length) return null;

  return (
    <section className="training">
      <div className="results-head">
        <h2>Learning curves</h2>
        <p className="muted small">
          Each point is a real evaluation taken during training: every 25,000 steps the policy
          played 20 deterministic episodes on a separate evaluation environment. Nothing is
          smoothed. These are the policies' progress <em>while learning</em>; the comparison
          above uses the final policy on the fixed seeds 1000–1029, so the end of a curve and
          the table need not match exactly.
        </p>
      </div>

      {runs.map((run) => {
        const rewards = run.points.flatMap((p) => [p.reward_mean - p.reward_std, p.reward_mean + p.reward_std]);
        const rLo = Math.min(...rewards);
        const rHi = Math.max(...rewards);
        const hasSuccess = run.points.every((p) => typeof p.success_rate === 'number');
        const first = run.points[0];
        const last = run.points[run.points.length - 1];
        // First evaluation at 90% success or better - read off the data, so the
        // battery run's long plateau is stated as a number rather than implied.
        const reached = hasSuccess ? run.points.find((p) => p.success_rate >= 0.9) : null;
        return (
          <article className="curve-row" key={run.tag}>
            <header>
              <h3>
                {scenarioTier(run.scenario) ? `${scenarioTier(run.scenario)}. ` : ''}
                {titleCase(run.scenario)}
              </h3>
              <p className="muted small">
                <code>{run.tag}</code> · {compact(run.total_timesteps)} steps
                {run.wall_clock_seconds ? ` · ${Math.round(run.wall_clock_seconds)} s on CPU` : ''}
                {hasSuccess
                  ? ` · success ${Math.round(first.success_rate * 100)}% → ${Math.round(last.success_rate * 100)}%`
                  : ''}
                {reached ? ` · first ≥90% at ${compact(reached.timesteps)} steps` : ''}
              </p>
            </header>
            <div className="curve-pair">
              {hasSuccess && (
                <Chart
                  title="Evaluation success rate"
                  points={run.points}
                  value={(p) => p.success_rate * 100}
                  domain={[0, 100]}
                  format={(v) => `${Math.round(v)}`}
                  unit="%"
                />
              )}
              <Chart
                title="Mean evaluation reward (band: ±1 SD)"
                points={run.points}
                value={(p) => p.reward_mean}
                band={(p) => [p.reward_mean - p.reward_std, p.reward_mean + p.reward_std]}
                domain={[rLo, rHi]}
                format={(v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1))}
                unit=""
              />
            </div>
          </article>
        );
      })}

      {missing.length > 0 && (
        <p className="muted small">
          No curve for {missing.map((m) => <code key={m.tag}>{m.tag}</code>).reduce((a, b) => [a, ', ', b])}:
          those policies were trained and evaluated (their results are in the table above), but
          their training logs were not retained, so there is no learning curve to show rather
          than one reconstructed after the fact.
        </p>
      )}

      <button type="button" className="pick reveal" onClick={() => setShowTable((v) => !v)}>
        {showTable ? 'Hide the evaluation numbers' : 'Show the evaluation numbers'}
      </button>
      {showTable && (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Policy</th>
                <th>Steps</th>
                <th>Success</th>
                <th>Mean reward</th>
                <th>SD</th>
                <th>Mean length</th>
              </tr>
            </thead>
            <tbody>
              {runs.flatMap((run) =>
                run.points.map((p) => (
                  <tr key={`${run.tag}-${p.timesteps}`}>
                    <td>{run.tag}</td>
                    <td>{p.timesteps.toLocaleString()}</td>
                    <td>{typeof p.success_rate === 'number' ? `${Math.round(p.success_rate * 100)}%` : 'n/a'}</td>
                    <td>{p.reward_mean.toFixed(2)}</td>
                    <td>{p.reward_std.toFixed(2)}</td>
                    <td>{p.length_mean.toFixed(1)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
