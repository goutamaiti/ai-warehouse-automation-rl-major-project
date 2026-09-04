import { REWARD_TERMS, RULESETS } from '../lib/rulesets';

const weights = RULESETS.default.reward;

const RESULTS = [
  {
    scenario: 'Static warehouse (default)',
    rows: [
      ['PPO', '30/30', '27.87', '1.00', '0.00'],
      ['A*', '30/30', '27.87', '1.00', '0.00'],
      ['Random', '1/30', '242.00', '0.13', '74.73'],
    ],
    takeaway:
      'PPO drives an exactly optimal path on every episode — identical to A*, using only a 5×5 local view. This is the ceiling: A* cannot be beaten on a static grid, only matched.',
  },
  {
    scenario: 'Moving obstacles (dynamic_obstacles)',
    rows: [
      ['PPO', '30/30', '28.93', '0.98', '0.47'],
      ['A*', '30/30', '30.50', '0.93', '0.00'],
      ['Random', '3/30', '404.67', '0.11', '85.17'],
    ],
    takeaway:
      'PPO finishes ~1.6 steps faster than A* by pushing through gaps instead of detouring — and pays for it with 0.47 collisions per episode where A* has none. Total reward ends up statistically tied: the reward function is pricing the trade-off as intended.',
  },
  {
    scenario: 'Battery-constrained (battery_constrained)',
    rows: [
      ['PPO', '0/30', 'n/a', 'n/a', '0.00'],
      ['A*', '30/30', '70.50', '0.91', '0.00'],
      ['Random', '0/30', 'n/a', 'n/a', '16.43'],
    ],
    takeaway:
      'PPO never learns to charge: it spends exactly its 45% starting battery every episode and fails. The battery-aware A* controller succeeds 30/30. Reported as-is — see "What does not work yet" below.',
  },
];

function Section({ title, children }) {
  return (
    <section className="info-section">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export default function InfoPage() {
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
          A warehouse is a 2D grid of six cell types: <strong>wall</strong> and <strong>shelf</strong>{' '}
          block movement; <strong>storage points</strong> are where a package can be picked up;{' '}
          <strong>packing stations</strong> are delivery targets; <strong>charging stations</strong>{' '}
          are where the robot starts and can recharge. A layout is only accepted if every point of
          interest is walkable and all of them can reach each other — the same check runs whether
          the map was generated procedurally (the five built-in scenarios) or drawn by hand on the
          Editor tab.
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

      <Section title="Reward function">
        <p>
          Every action earns or costs reward from the terms below. The same weights are shown live,
          per step, in the Reward panel on the Dashboard and Editor tabs.
        </p>
        <table className="info-table reward-table">
          <thead>
            <tr><th>Term</th><th>Rule</th><th>Why</th></tr>
          </thead>
          <tbody>
            {REWARD_TERMS.map((term) => (
              <tr key={term.key}>
                <td>{term.label}</td>
                <td>{term.describe(weights)}</td>
                <td>{term.why}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          These are reasoned starting values (see the source repository's{' '}
          <code>docs/rl-formulation.md</code>), not the output of a hyper-parameter search.
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
      </Section>

      <Section title="Algorithms">
        <table className="info-table">
          <tbody>
            <tr><th>PPO</th><td>The learned policy. Actor-critic, clipped objective, a 128×128 MLP over the 37-value observation. Trained with Stable-Baselines3.</td></tr>
            <tr><th>A*</th><td>Classical baseline with the <em>full map</em> and every obstacle position, replanning every step. Also decides when to detour to a charger. This is a deliberate advantage over PPO, which only ever sees its local observation.</td></tr>
            <tr><th>BFS</th><td>Uninformed search — same path length as A* on this uniform-cost grid, kept to show what the Manhattan-distance heuristic actually saves in nodes expanded.</td></tr>
            <tr><th>Random</th><td>Uniformly random actions — the floor every other controller must clear.</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="Measured results">
        <p className="muted small">
          30 episodes per controller, identical seeds across controllers (paired comparison). Full
          tables and methodology are in the source repository's <code>docs/experiments.md</code>.
        </p>
        {RESULTS.map((block) => (
          <div key={block.scenario} className="info-result-block">
            <h3>{block.scenario}</h3>
            <div className="table-scroll">
              <table className="info-table">
                <thead>
                  <tr><th>Controller</th><th>Success</th><th>Steps</th><th>Path efficiency</th><th>Collisions</th></tr>
                </thead>
                <tbody>
                  {block.rows.map((row) => (
                    <tr key={row[0]}>
                      {row.map((cell, i) => <td key={i}>{cell}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>{block.takeaway}</p>
          </div>
        ))}
      </Section>

      <Section title="What does not work yet">
        <p>
          PPO fails completely on the battery-constrained scenario (0/30, see above). It is reported
          rather than tuned away. The likely cause: charging pays no reward of its own, so a policy
          only learns to use it if it can connect a charge now to avoiding a penalty roughly 40 steps
          later — a long gap for training that never explicitly rewards the intermediate step, and
          the depletion penalty (−10) is smaller than one delivery reward (+20), so "deliver once and
          die" still scores positively. Multi-robot coordination, moving obstacles in the Editor's
          offline sandbox, and a reward-weight ablation study are also not implemented — see the
          repository README for the full roadmap.
        </p>
      </Section>

      <Section title="Dashboard tabs">
        <table className="info-table">
          <tbody>
            <tr><th>Dashboard</th><td>Replays real recorded episodes (data/episodes/) and shows the aggregated result tables (data/results/) — this is where the official, reported numbers live.</td></tr>
            <tr><th>Editor</th><td>Paint a custom warehouse, pick a ruleset and a controller, and run it. With a connected backend this runs the real Python simulation (including PPO and moving obstacles); without one, A*, BFS and Random run in an in-browser sandbox reimplementation for exploration only — its numbers are never part of the reported results.</td></tr>
            <tr><th>Info</th><td>This page.</td></tr>
          </tbody>
        </table>
      </Section>

      <footer className="info-footer">
        <p>
          Every number on this page and in the Dashboard tab comes from a simulation run that
          actually happened. Nothing here is estimated, mocked, or hand-written.
        </p>
      </footer>
    </div>
  );
}
