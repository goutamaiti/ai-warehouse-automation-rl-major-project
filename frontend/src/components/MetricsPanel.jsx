import {
  controllerLabel,
  episodeSucceeded,
  formatPercent,
  outcomeLabel,
  pathEfficiency,
  titleCase,
} from '../lib/format';
import { scenarioLabel } from '../lib/complexity';

/**
 * Live state of the current frame plus the summary of the finished episode.
 * Every value shown here comes straight from the recording file.
 */
export default function MetricsPanel({ meta, frame, summary, frameIndex, totalFrames }) {
  if (!frame) {
    return (
      <aside className="panel">
        <h2>This run</h2>
        <p className="muted">Nothing has run yet.</p>
      </aside>
    );
  }

  const robot = frame.robot ?? {};
  const task = frame.task ?? {};
  const efficiency = pathEfficiency(summary);
  const succeeded = episodeSucceeded(summary);

  return (
    <aside className="panel">
      <h2>This run</h2>
      <dl className="kv">
        <div><dt>Driving method</dt><dd>{controllerLabel(meta.controller)}</dd></div>
        <div><dt>Warehouse setup</dt><dd>{scenarioLabel(meta.scenario)}</dd></div>
      </dl>

      <h2>Robot now</h2>
      <dl className="kv">
        <div>
          <dt>Step</dt>
          <dd>{frameIndex} / {Math.max(totalFrames - 1, 0)}</dd>
        </div>
        <div>
          <dt>Battery</dt>
          <dd>
            <span className={`badge battery-${robot.battery_state ?? 'ok'}`}>
              {(robot.battery ?? 0).toFixed(1)}%
            </span>
          </dd>
        </div>
        <div><dt>Carrying</dt><dd>{robot.carrying ? 'yes' : 'nothing'}</dd></div>
        <div><dt>Task</dt><dd>{titleCase(task.status)}</dd></div>
        <div><dt>Distance to target</dt><dd>{frame.distance_to_target ?? 'n/a'}</dd></div>
      </dl>

      {frame.events?.length > 0 && (
        <div className="events">
          {frame.events.map((event) => (
            <span key={event} className={`badge event-${event}`}>{event.replace(/_/g, ' ')}</span>
          ))}
        </div>
      )}

      <h2>How it ended</h2>
      {summary ? (
        <dl className="kv">
          <div>
            <dt>Outcome</dt>
            <dd>
              <span className={`badge ${succeeded ? 'ok' : 'fail'}`}>
                {outcomeLabel(summary.termination_reason)}
              </span>
            </dd>
          </div>
          <div><dt>Steps</dt><dd>{summary.steps}</dd></div>
          <div><dt>Path length</dt><dd>{summary.moves} moves</dd></div>
          <div><dt>Shortest possible</dt><dd>{summary.optimal_path_length} moves</dd></div>
          <div>
            <dt>Route quality</dt>
            <dd>{efficiency === null ? 'n/a' : formatPercent(efficiency)}</dd>
          </div>
          <div><dt>Collisions</dt><dd>{summary.collisions}</dd></div>
          <div><dt>Energy used</dt><dd>{summary.energy_consumed}%</dd></div>
          <div><dt>Steps spent charging</dt><dd>{summary.charging_events}</dd></div>
          <div><dt>Steps spent waiting</dt><dd>{summary.idle_steps}</dd></div>
          <div><dt>AI training score</dt><dd>{summary.total_reward ?? 'n/a'}</dd></div>
        </dl>
      ) : (
        <p className="muted">This recording has no summary.</p>
      )}
    </aside>
  );
}
