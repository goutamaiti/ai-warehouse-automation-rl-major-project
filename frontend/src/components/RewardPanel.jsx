import { REWARD_TERMS, RULESETS } from '../lib/rulesets';

const FALLBACK_WEIGHTS = RULESETS.default.reward;

function formatSigned(value) {
  if (value === 0 || value === undefined || value === null) return '0.00';
  const sign = value > 0 ? '+' : '−';
  return `${sign}${Math.abs(value).toFixed(2)}`;
}

/**
 * The full reward "rulebook" (every term the environment can pay or charge,
 * with its configured weight) plus, when a frame is playing, which of those
 * terms actually fired on this step and the running total so far.
 *
 * `rewardConfig` should be the `reward_config` block that travelled with the
 * episode (real weights the run actually used); it falls back to the default
 * ruleset's weights so the legend still renders before anything is loaded.
 */
export default function RewardPanel({ rewardConfig, components, reward, cumulative, stepIndex }) {
  const weights = rewardConfig ?? FALLBACK_WEIGHTS;
  const hasStep = Boolean(components);

  return (
    <section className="panel reward-panel">
      <div className="reward-panel-header">
        <h2>Reward &amp; penalties</h2>
        {hasStep && (
          <span className={`badge ${reward >= 0 ? 'ok' : 'fail'}`}>
            step {stepIndex}: {formatSigned(reward)}
          </span>
        )}
      </div>

      <div className="reward-terms">
        {REWARD_TERMS.map((term) => {
          const value = hasStep ? components[term.key] : undefined;
          const active = hasStep && Math.abs(value ?? 0) > 1e-9;
          return (
            <div key={term.key} className={`reward-term tone-${term.tone} ${active ? 'active' : ''}`}>
              <div className="reward-term-top">
                <span className="reward-term-label">{term.label}</span>
                {hasStep ? (
                  <span className="reward-term-value">{active ? formatSigned(value) : '—'}</span>
                ) : (
                  <span className="reward-term-sign">{term.sign}</span>
                )}
              </div>
              <p className="reward-term-rule">{term.describe(weights)}</p>
              <p className="reward-term-why">{term.why}</p>
            </div>
          );
        })}
      </div>

      {hasStep && (
        <div className="reward-totals">
          <div>
            <dt>This step</dt>
            <dd>{formatSigned(reward)}</dd>
          </div>
          <div>
            <dt>Cumulative so far</dt>
            <dd>{formatSigned(cumulative)}</dd>
          </div>
        </div>
      )}
    </section>
  );
}
