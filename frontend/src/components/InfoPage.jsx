import { useEffect, useMemo, useState } from 'react';

import { loadEpisode, loadIndex, loadResult } from '../lib/api';
import { byComplexity, scenarioAdds, scenarioRank, scenarioTier } from '../lib/complexity';
import { controllerLabel, titleCase } from '../lib/format';
import { REWARD_TERMS, RULESETS } from '../lib/rulesets';

/**
 * The method, explained — and demonstrated on a real episode.
 *
 * The prose is the point of this page: it is what makes the deployed site
 * explainable without the repository. But a reward function described in a
 * table is a claim, and this project has the data to show it instead, so the
 * reward section is driven by an actual recording: step through it and each
 * term shows what it paid or charged on that step, next to the explanation of
 * why the term exists.
 *
 * Nothing here is hand-written any more. The results tables used to be a
 * hardcoded array, which had quietly gone stale - it still reported PPO failing
 * 0/30 on the battery scenario, a result since fixed, and a Random controller
 * that no longer exists - while the footer claimed every number came from a run
 * that actually happened. Reading them from data/results/ makes that claim true
 * rather than aspirational.
 *
 * Weights come from the recording, not from a table
 * -------------------------------------------------
 * Each term's rule text is rendered from the chosen episode's own
 * `reward_config`, which every replay file carries. Taken from a single ruleset
 * instead, the page would describe a -60 depletion penalty while walking an
 * episode that was actually run at -10 - which is precisely the distinction the
 * battery section is about, and would be wrong by a factor of six.
 */

function Section({ title, children }) {
  return (
    <section className="info-section">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function fmt(entry, digits = 2) {
  if (entry === null || entry === undefined) return 'n/a';
  if (typeof entry === 'number') return entry.toFixed(digits);
  if (typeof entry.mean !== 'number') return 'n/a';
  return entry.mean.toFixed(digits);
}

/** A term's value for this step. Two decimals is right for everything except
 *  the energy term, whose idle charge is 0.001 and printed as "-0.00" - which
 *  reads as a bug rather than as a very small number. */
function signed(value) {
  const digits = Math.abs(value) < 0.01 ? 3 : 2;
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}`;
}

/**
 * What each measured scenario actually shows.
 *
 * The page used to carry a hand-written takeaway per scenario, and they were
 * the most useful writing on it - so they are kept, but as functions of the
 * loaded summaries rather than as sentences with numbers typed into them. The
 * old ones had gone stale silently ("PPO never learns to charge" survived the
 * fix that made it charge); these cannot, because every figure they quote is
 * read from the same file as the table directly above them.
 *
 * `by` is a lookup of that scenario's summaries by controller name.
 */
const TAKEAWAYS = {
  simple_static: (by) =>
    `All three controllers are identical to the last digit - ${fmt(by.ppo?.steps)} steps at a path
     efficiency of ${fmt(by.ppo?.path_efficiency)}. On a static floor A* is provably optimal, so
     this is the ceiling: it can be matched, never beaten. PPO matches it seeing only a 5x5 patch.`,
  default: (by) =>
    `The same exact tie over a longer route - ${fmt(by.ppo?.steps)} steps for every controller.
     Scaling the floor up does not open a gap, which is the point: the learned policy is not
     approximately optimal here, it is optimal.`,
  complex_static: (by) =>
    `Densest shelving, longest paths, and still a three-way tie at ${fmt(by.ppo?.steps)} steps.
     Three static floors, three exact ties - the static case is settled, and the interesting
     question is what happens when the shortest path stops being safe.`,
  dynamic_obstacles: (by) =>
    `Here they separate. PPO finishes ${fmt(
      (by.astar?.steps?.mean ?? 0) - (by.ppo?.steps?.mean ?? 0),
      1
    )} steps sooner than A* by pushing through gaps instead of detouring round them, and pays
     ${fmt(by.ppo?.collisions)} collisions an episode for it where A* has none. Total reward comes
     out statistically tied (${fmt(by.ppo?.total_reward)} ± ${fmt(
       by.ppo?.total_reward?.ci95
     )} against ${fmt(by.astar?.total_reward)} ± ${fmt(
       by.astar?.total_reward?.ci95
     )}): the reward function is pricing that trade-off exactly as intended, which is the result,
     not a draw.`,
  battery_constrained: (by) =>
    `The failure. PPO delivers ${by.ppo?.n_successful}/${by.ppo?.n_episodes} and records
     ${fmt(by.ppo?.charging_events)} charging events - it never charges once, in thirty episodes -
     while the battery-aware planner takes ${by.astar?.n_successful}/${by.astar?.n_episodes}. The
     tell is that PPO still scores ${fmt(by.ppo?.total_reward)}: a positive return for an episode
     that ended in a dead robot. It was not failing to learn. It had learnt that dying paid.`,
  battery_tuned: (by) =>
    `The same warehouse with the depletion penalty corrected. PPO now delivers
     ${by.ppo?.n_successful}/${by.ppo?.n_episodes} and charges on ${fmt(
       by.ppo?.charging_events
     )} steps an episode. It is not a clean win: at ${fmt(by.ppo?.steps)} steps against
     ${fmt(by.astar?.steps)} and a path efficiency of ${fmt(
       by.ppo?.path_efficiency
     )} against ${fmt(
       by.astar?.path_efficiency
     )}, the policy charges more than it needs to and takes a longer route than the planner. The
     claim is that the battery task became learnable, not that it was solved better.`,
};

/** What a recording ended as, in words, taken from its own summary. */
function outcome(summary) {
  if (!summary) return 'outcome not recorded';
  const parts = [`${summary.tasks_delivered ?? 0} delivered`];
  if (summary.tasks_failed) parts.push(`${summary.tasks_failed} failed`);
  if (summary.charging_events) parts.push(`${summary.charging_events} charging steps`);
  parts.push(`${summary.steps} steps`);
  return parts.join(' · ');
}

export default function InfoPage() {
  const [results, setResults] = useState([]);
  const [walkable, setWalkable] = useState([]);
  const [episodeId, setEpisodeId] = useState(null);
  const [episode, setEpisode] = useState(null);
  const [step, setStep] = useState(0);

  useEffect(() => {
    let cancelled = false;
    loadIndex()
      .then(async (data) => {
        const loaded = await Promise.all(
          (data.results ?? []).map(async (entry) => {
            const full = await loadResult(entry.id).catch(() => null);
            return full ? { ...entry, summaries: full.summaries ?? [] } : null;
          })
        );
        if (cancelled) return;
        setResults(loaded.filter(Boolean).sort(
          (a, b) => scenarioRank(a.scenario) - scenarioRank(b.scenario)
        ));

        // The learned policy only. This section is about the reward function
        // the policy was trained against; a classical controller's recording
        // would illustrate the arithmetic just as well while inviting the
        // reader to think A* is being rewarded for anything at all.
        const policy = (data.episodes ?? [])
          .filter((e) => e.controller === 'ppo')
          .sort(byComplexity);
        setWalkable(policy);

        // Open on the battery fix: it is the recording where the most terms
        // fire - the step cost and energy throughout, progress on nearly every
        // step, a pickup, two deliveries, and the waiting that charging is made
        // of - and it is the project's headline result.
        const first =
          policy.find((e) => e.scenario === 'battery_tuned') ??
          policy.find((e) => e.scenario === 'default') ??
          policy[0];
        if (first) setEpisodeId(first.id);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!episodeId) return undefined;
    let cancelled = false;
    loadEpisode(episodeId)
      .then((full) => {
        if (cancelled) return;
        setEpisode({ ...full, id: episodeId });
        setStep(0);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [episodeId]);

  /* The weights this episode was actually run at, so the rule text and the
     value printed beside it cannot disagree. */
  const weights = episode?.reward_config ?? RULESETS.battery_tuned.reward;

  const frames = useMemo(
    () => (episode?.frames ?? []).filter((f) => f.reward_components),
    [episode]
  );
  const frame = frames[Math.min(step, Math.max(frames.length - 1, 0))] ?? null;
  const components = frame?.reward_components ?? {};

  /** Steps on which each term was non-zero, for the little activity strip. */
  const activity = useMemo(() => {
    const map = {};
    REWARD_TERMS.forEach((term) => {
      map[term.key] = frames.map((f) => {
        const v = f.reward_components?.[term.key] ?? 0;
        return v === 0 ? 0 : v > 0 ? 1 : -1;
      });
    });
    return map;
  }, [frames]);

  const cumulative = useMemo(
    () => frames.slice(0, step + 1).reduce((total, f) => total + (f.reward ?? 0), 0),
    [frames, step]
  );

  return (
    <div className="info-page">
      <header className="info-hero">
        <h1>AI-Based Warehouse Automation System</h1>
        <p className="subtitle">
          Reinforcement Learning for Intelligent Robot Navigation — what this project is, what was
          built, and what the numbers actually show.
        </p>
      </header>

      <Section title="Research question">
        <p>
          Can a Reinforcement Learning navigation policy match or improve on conventional path
          planning for warehouse delivery tasks under simulated warehouse conditions — and where
          does each approach break down? The project is a 100% software simulation: no physical
          robots, no ROS, no hardware of any kind.
        </p>
      </Section>

      <Section title="How a warehouse is built">
        <p>
          A warehouse is a 2D grid of six cell types: <strong>walls</strong> and{' '}
          <strong>racking</strong> block movement; <strong>pick points</strong> are where a package
          is collected (<code>storage_points</code> in the simulation code);{' '}
          <strong>packing stations</strong> are delivery targets; <strong>chargers</strong> are
          where the robot starts and can recharge. The Editor tab paints with exactly these six, and
          draws each as its own mark rather than its own shade — a filled square for a pick point, a
          hollow one for packing, a dashed one for a charger — so a floor looks the same while you
          are drawing it as it does when a robot runs on it. A layout is only accepted if every
          point of interest is walkable and all of them can reach each other — the same check runs
          whether
          the map was generated procedurally (the six measured scenarios), drawn by hand on the
          Editor tab, or generated as the larger robotic-fulfilment floor the Live floor tab offers.
        </p>
      </Section>

      <Section title="The MDP">
        <table className="info-table">
          <tbody>
            <tr><th>Agent</th><td>One virtual warehouse robot</td></tr>
            <tr><th>Observation</th><td>37 values: robot/target position, distance to target, battery %, carrying flag, offset and distance to the nearest charger, and a 5×5 local occupancy patch. The robot never sees the full map.</td></tr>
            <tr><th>Actions</th><td>5 discrete: up, down, left, right, wait. Pickup / delivery / charging happen automatically on the right cell.</td></tr>
            <tr><th>Episode ends</th><td>Success (every task delivered), failure (battery hits 0%), or truncation (step budget exhausted)</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="Reward function, on a real episode">
        <p>
          Every action earns or costs reward from the terms below. Rather than describe them, this
          walks a real recording: pick a scenario, step through the learned policy's episode on it,
          and each term shows what it actually paid or charged on that step — at the weights that
          episode was run at, read from the replay file rather than written here. The strip beside
          each term marks every step of the episode on which it fired.
        </p>

        {walkable.length > 1 && (
          <div className="walk-pick">
            {walkable.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`pick${item.id === episodeId ? ' active' : ''}`}
                onClick={() => setEpisodeId(item.id)}
              >
                {scenarioTier(item.scenario) ? `${scenarioTier(item.scenario)}. ` : ''}
                {titleCase(item.scenario)}
              </button>
            ))}
          </div>
        )}

        {frames.length > 0 ? (
          <>
            <div className="walk-controls">
              <button
                type="button"
                className="pick"
                onClick={() => setStep((s) => Math.max(0, s - 1))}
                disabled={step === 0}
              >
                Previous step
              </button>
              <input
                type="range"
                min="0"
                max={frames.length - 1}
                value={Math.min(step, frames.length - 1)}
                onChange={(event) => setStep(Number(event.target.value))}
                aria-label="episode step"
              />
              <button
                type="button"
                className="pick"
                onClick={() => setStep((s) => Math.min(frames.length - 1, s + 1))}
                disabled={step >= frames.length - 1}
              >
                Next step
              </button>
            </div>

            <div className="walk-head">
              <span>
                Step <b className="num">{frame?.step ?? 0}</b> of {frames.length}
              </span>
              <span>
                This step <b className="num">{(frame?.reward ?? 0).toFixed(2)}</b>
              </span>
              <span>
                Cumulative <b className="num">{cumulative.toFixed(2)}</b>
              </span>
              {frame?.events?.length ? (
                <span className="walk-event">{frame.events.join(' · ')}</span>
              ) : null}
            </div>

            <div className="terms">
              {REWARD_TERMS.map((term) => {
                const value = components[term.key] ?? 0;
                const fired = value !== 0;
                return (
                  <article key={term.key} className={`term${fired ? ' fired' : ''}`}>
                    <header>
                      <h3>{term.label}</h3>
                      <b className="num">{fired ? signed(value) : '—'}</b>
                    </header>
                    <p className="term-rule">{term.describe(weights)}</p>
                    {/* The prose stays: the point of this page is explaining
                        why each term exists, not only what it did. */}
                    <p className="term-why">{term.why}</p>
                    {/* Every step of the episode, one tick each: dark where the
                        term paid, mid where it charged, hairline where it was
                        silent. The line is where the scrubber currently sits,
                        so "fires constantly" and "fires once" are visible
                        without stepping through the whole recording. */}
                    <span className="term-strip" aria-hidden="true">
                      {(activity[term.key] ?? []).map((state, i) => (
                        <i
                          key={i}
                          className={
                            state === 0 ? 'off' : state > 0 ? 'credit' : 'charge'
                          }
                        />
                      ))}
                      <u
                        style={{
                          left: `${(Math.min(step, frames.length - 1) / Math.max(frames.length - 1, 1)) * 100}%`,
                        }}
                      />
                    </span>
                  </article>
                );
              })}
            </div>

            <p className="muted small">
              Walking <code>{episode?.id}</code> — {outcome(episode?.summary)}. A term with an empty
              strip never fired in this recording; that is a fact about the episode, not a gap in the
              implementation. <strong>Battery reserve</strong> is the clearest case: it is switched
              on in <code>battery_tuned</code> at weight 10, and still reads zero on every step,
              because it is defined to stay flat while a charger is comfortably in reach and this
              policy never let the charge fall far enough to leave that region.
            </p>
          </>
        ) : (
          <p className="muted small">
            Loading a recording… if this persists, the bundled data under{' '}
            <code>frontend/public/data</code> has not been generated — run{' '}
            <code>npm run copy-data</code>.
          </p>
        )}

        <p className="muted small">
          These are reasoned starting values (see the source repository's{' '}
          <code>docs/rl-formulation.md</code>), not the output of a hyper-parameter search — with one
          exception: the battery scenario's depletion penalty was derived from the measured reward of
          the policy that failed, described below.
        </p>
      </Section>

      <Section title="Two bugs a reward function can hide">
        <p>
          The progress term uses potential-based shaping (Ng, Harada &amp; Russell, 1999):{' '}
          <code>F = γ·Φ(s′) − Φ(s)</code> with <code>Φ(s) = −distance(s, target)</code>. Proven not to
          change the optimal policy — in theory. Two ways it broke in practice, both caught before
          the final training runs and covered by a regression test:
        </p>
        <ol>
          <li>
            <strong>Standing still paid a bonus.</strong> With shaping discount γ below 1.0, waiting
            earns <code>distance × (1 − γ)</code> every step — with a far target that exceeded the
            0.05 step penalty, i.e. the agent was paid to loiter. Fixed by fixing γ at exactly 1.0 for
            this term.
          </li>
          <li>
            <strong>The delivery step was punished.</strong> A finished task's target briefly fell
            back to the pickup point, so the distance jumped by the width of the warehouse on the
            very last step — cancelling the +20 delivery reward with a huge negative shaping term.
            Fixed by keeping a completed task's target at its last real destination.
          </li>
        </ol>
        <p>
          <strong>What is not fixed, and is stated rather than hidden.</strong> The guarantee
          needs the shaping discount to equal the learner's. Here it is 1.0 (that is the fix for
          bug 1) while PPO discounts at 0.99, and the potential is not reset to zero when an
          episode fails. So the shaping is exactly policy-invariant for the undiscounted return,
          and only approximately for the discounted one PPO optimises: three A* runs with the same
          start and end, differing only in 0, 20 or 60 waits at the start, all collect +38.0 of
          shaping undiscounted, but 31.6, 25.9 and 17.3 discounted. Matching the two would bring
          back bug 1 unless the step penalty rose about fivefold, and would mean retraining every
          policy, so every number on this site was measured with the reward exactly as described
          here.
        </p>
      </Section>

      <Section title="Algorithms">
        <table className="info-table">
          <tbody>
            <tr><th>PPO</th><td>The learned policy. Actor-critic, clipped objective, a 128×128 MLP over the 37-value observation. Trained with Stable-Baselines3.</td></tr>
            <tr><th>A*</th><td>Classical baseline with the <em>full map</em> and every obstacle position, replanning every step. Also decides when to detour to a charger. This is a deliberate advantage over PPO, which only ever sees its local observation.</td></tr>
            <tr><th>BFS</th><td>Uninformed search — same path length as A* on this uniform-cost grid, kept to show what the Manhattan-distance heuristic actually saves in nodes expanded.</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="Measured results">
        <p className="muted small">
          30 episodes per controller, identical seeds across controllers (paired comparison), read
          live from the committed result files. Full tables and methodology are in the source
          repository's <code>docs/experiments.md</code>.
        </p>
        {results.length === 0 ? (
          <p className="muted small">Loading the committed results…</p>
        ) : (
          results.map((block) => (
            <div key={block.id} className="info-result-block">
              <h3>
                {scenarioTier(block.scenario) ? `${scenarioTier(block.scenario)}. ` : ''}
                {titleCase(block.scenario)}
              </h3>
              <p className="muted small">{scenarioAdds(block.scenario)}</p>
              <div className="table-scroll">
                <table className="info-table">
                  <thead>
                    <tr>
                      <th>Controller</th><th>Success</th><th>Steps</th>
                      <th>Path efficiency</th><th>Collisions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {block.summaries.map((s) => (
                      <tr key={s.controller}>
                        <td>{controllerLabel(s.controller)}</td>
                        <td>{s.n_successful}/{s.n_episodes}</td>
                        <td>{fmt(s.steps, 2)}</td>
                        <td>{fmt(s.path_efficiency, 2)}</td>
                        <td>{fmt(s.collisions, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {TAKEAWAYS[block.scenario] && (
                <p className="takeaway">
                  {TAKEAWAYS[block.scenario](
                    Object.fromEntries(block.summaries.map((s) => [s.controller, s]))
                  )}
                </p>
              )}
            </div>
          ))
        )}
      </Section>

      <Section title="The battery result, and how it was fixed">
        <p>
          On <code>battery_constrained</code> the learned policy fails completely: 0 of 30 episodes,
          with <strong>zero charging events ever recorded</strong>. It was not under-trained. It was
          correctly maximising a reward function that paid it to die — every one of those failing
          episodes still scored positively, because the −10 depletion penalty was cheaper than the
          credit a policy could bank first: one pickup (+5), one delivery (+20) and roughly 27 of
          accumulated progress shaping. Select <strong>Battery constrained</strong> in the
          walkthrough above and drag to the final step to read it off the recording: the episode
          ends on <code>battery_depleted</code>, the term charges −10, and the cumulative reward is
          still <b className="num">+21.60</b>. Nothing was broken. Dying was simply the better deal.
        </p>
        <p>
          <code>battery_tuned</code> is the same warehouse with the depletion penalty raised to 60 —
          above everything an episode can bank — plus a potential-based battery-reserve term and a
          higher price on waiting. The policy now charges and delivers. Both scenarios are kept
          above so the before and after sit side by side; the fix is one number, and the number was
          derived from the measured reward of the policy that failed rather than guessed. Both
          recordings are in the walkthrough above: switch between them and watch{' '}
          <strong>Battery depleted</strong> charge −10 on the final step of one and never fire at all
          in the other.
        </p>
      </Section>

      <Section title="What does not work yet">
        <p>
          Multi-robot operation runs, but no learned policy drives it: the trained policies have
          never seen another robot, and the Live floor tab is driven by the classical controller
          only, so nothing on it is a reinforcement-learning result. Above roughly eight robots some
          still run out of charge under charger contention. A reward-weight ablation study, multiple
          training seeds per configuration, and multi-agent RL are all unimplemented — see the
          repository README for the roadmap.
        </p>
      </Section>

      <Section title="Dashboard tabs">
        <table className="info-table">
          <tbody>
            <tr><th>Dashboard</th><td>Replays real recorded episodes (<code>data/episodes/</code>) and shows the aggregated result tables (<code>data/results/</code>) — this is where the official, reported numbers live.</td></tr>
            <tr><th>Live floor</th><td>A warehouse that keeps running, with several robots working a shared queue. Add a product and the nearest free robot is assigned on the next tick. A demonstration, not a measured result: its numbers depend on how long the page was left open.</td></tr>
            <tr><th>Editor</th><td>Paint a custom warehouse, pick a ruleset and a controller, and run it. With a connected backend this runs the real Python simulation, including PPO; without one, the classical controllers run in an in-browser sandbox for exploration only — its numbers are never part of the reported results.</td></tr>
            <tr><th>Info</th><td>This page.</td></tr>
          </tbody>
        </table>
      </Section>

      <footer className="info-footer">
        <p>
          Every number on this page comes from a simulation run that actually happened — the tables
          and the reward walkthrough above are read from the committed result and replay files, not
          written by hand. Where something has not been measured, it says so.
        </p>
      </footer>
    </div>
  );
}
