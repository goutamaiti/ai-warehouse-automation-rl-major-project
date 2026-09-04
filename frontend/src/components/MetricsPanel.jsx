import {
  controllerLabel,
  episodeSucceeded,
  formatPercent,
  pathEfficiency,
  titleCase,
} from '../lib/format';

/**
 * Live state of the current frame plus the summary of the finished episode.
 * Every value shown here comes straight from the recording file.
 */
export default function MetricsPanel({ meta, frame, summary, frameIndex, totalFrames }) {
  if (!frame) {
    return (
      <aside className="panel">
        <h2>Episode</h2>
        <p className="muted">No episode loaded.</p>
      </aside>
    );
  }

  const robot = frame.robot ?? {};
  const task = frame.task ?? {};
  const efficiency = pathEfficiency(summary);
  const succeeded = episodeSucceeded(summary);

  return (
    <aside className="panel">
      <h2>Episode</h2>
      <dl className="kv">
        <div><dt>Controller</dt><dd>{controllerLabel(meta.controller)}</dd></div>
        <div><dt>Scenario</dt><dd>{titleCase(meta.scenario)}</dd></div>
        <div><dt>Seed</dt><dd>{meta.seed}</dd></div>
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
        <div><dt>Payload</dt><dd>{robot.carrying ? 'carrying package' : 'empty'}</dd></div>
        <div><dt>Task</dt><dd>{titleCase(task.status)}</dd></div>
        <div><dt>Target</dt><dd>{task.target ? `(${task.target[0]}, ${task.target[1]})` : 'n/a'}</dd></div>
        <div><dt>Distance to target</dt><dd>{frame.distance_to_target ?? 'n/a'}</dd></div>
      </dl>

      {frame.events?.length > 0 && (
        <div className="events">
          {frame.events.map((event) => (
            <span key={event} className={`badge event-${event}`}>{event.replace(/_/g, ' ')}</span>
          ))}
        </div>
      )}

      <h2>Episode result</h2>
      {summary ? (
        <dl className="kv">
          <div>
            <dt>Outcome</dt>
            <dd>
              <span className={`badge ${succeeded ? 'ok' : 'fail'}`}>
                {titleCase(summary.termination_reason)}
              </span>
            </dd>
          </div>
          <div><dt>Steps</dt><dd>{summary.steps}</dd></div>
          <div><dt>Path length</dt><dd>{summary.moves} moves</dd></div>
          <div><dt>Shortest possible</dt><dd>{summary.optimal_path_length} moves</dd></div>
          <div>
            <dt>Path efficiency</dt>
            <dd>{efficiency === null ? 'n/a' : formatPercent(efficiency)}</dd>
          </div>
          <div><dt>Collisions</dt><dd>{summary.collisions}</dd></div>
          <div><dt>Energy used</dt><dd>{summary.energy_consumed}%</dd></div>
          <div><dt>Charging events</dt><dd>{summary.charging_events}</dd></div>
          <div><dt>Idle steps</dt><dd>{summary.idle_steps}</dd></div>
          <div><dt>Total reward</dt><dd>{summary.total_reward ?? 'n/a'}</dd></div>
        </dl>
      ) : (
        <p className="muted">This recording has no summary block.</p>
      )}
    </aside>
  );
}
