import { useEffect, useMemo, useState } from 'react';

import FleetCanvas from './FleetCanvas';
import { backendConfigured, loadFleetStudy } from '../lib/api';
import { MAP_THEMES } from '../lib/mapTheme';
import { BRAND } from '../lib/site';

/**
 * What this project is, for someone who has never heard of it.
 *
 * Every figure on this page is read from a file a real run wrote: the moving
 * floor from data/fleet/home_floor.json and the proof from
 * data/fleet/reliability.json. If either file is missing its section is left
 * out, because a typed-in fallback would be a number nobody measured.
 */

const TICK_MS = 110;
const REPLAY_END_PAUSE_TICKS = 25;

//: The two battery plans the proof compares. Same floor, fleets and orders.
const FAILS = 'battery_tuned';
const WORKS = 'battery_large_floor';

const QUESTIONS = [
  {
    title: 'Will the batteries last?',
    body: 'See which robots run flat, and when, before a battery is chosen.',
  },
  {
    title: 'Will the robots jam?',
    body: 'See where robots queue, block each other or get stuck.',
  },
  {
    title: 'Is the fleet big enough?',
    body: 'Change the number of robots and see whether the orders still get delivered.',
  },
];

const STEPS = [
  {
    title: 'Set up the plan',
    body: 'Choose a floor, the number of robots and the battery, or draw your own layout.',
  },
  {
    title: 'Run it',
    body: 'Orders arrive and the simulator drives every robot, move by move, including traffic and charging.',
  },
  {
    title: 'Read the result',
    body: 'See what was delivered, which robots ran flat and which ones got stuck.',
  },
];

function expandFrame(frame) {
  return {
    robots: frame.robots.map(([row, col, carrying, battery, alive], id) => ({
      id,
      position: [row, col],
      carrying: Boolean(carrying),
      battery,
      alive: Boolean(alive),
    })),
    requests: frame.waiting.map((pickup) => ({ pickup, status: 'waiting' })),
  };
}

/** A recorded run playing on a loop. It needs no backend, so it always works. */
function RecordedFloor({ recording }) {
  const [tick, setTick] = useState(0);
  const last = recording.frames.length - 1;

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setTick(Math.floor(last / 2));
      return undefined;
    }
    const timer = setInterval(
      () => setTick((t) => (t >= last + REPLAY_END_PAUSE_TICKS ? 0 : t + 1)),
      TICK_MS
    );
    return () => clearInterval(timer);
  }, [last]);

  const shown = recording.frames[Math.min(tick, last)];
  const frame = useMemo(() => expandFrame(shown), [shown]);
  const charged = shown.robots.filter((robot) => robot[4]).length;

  return (
    <figure className="hero-floor">
      <FleetCanvas layout={recording.layout} frame={frame} palette={MAP_THEMES.dark} plain />
      <figcaption>
        <div className="hero-stats">
          <span>
            <b className="num">{shown.delivered}</b> of {recording.products} orders delivered
          </span>
          <span>
            <b className="num">{charged}</b> of {shown.robots.length} robots still charged
          </span>
        </div>
        <ul className="hero-key">
          <li><i style={{ background: MAP_THEMES.dark.robot }} /> empty robot</li>
          <li><i style={{ background: MAP_THEMES.dark.robotCarrying }} /> carrying</li>
          <li><i className="square" style={{ background: MAP_THEMES.dark.activePickup }} /> order waiting</li>
        </ul>
        <p>A recorded simulation, not a video of real robots.</p>
      </figcaption>
    </figure>
  );
}

function tally(summary, rules) {
  const rows = summary.filter((row) => row.rules === rules);
  const sum = (read) => rows.reduce((total, row) => total + read(row), 0);
  return {
    runs: sum((row) => row.runs),
    finished: sum((row) => row.all_delivered),
    flat: sum((row) => row.robots_dead),
    robots: sum((row) => row.robots * row.runs),
  };
}

function PlanResult({ name, moves, result }) {
  const works = result.finished === result.runs;
  return (
    <article className={`plan-card ${works ? 'works' : 'fails'}`}>
      <p className="plan-name">{name}</p>
      <p className="plan-battery">{moves} moves on a full charge</p>
      <p className="plan-score num">
        {result.finished} of {result.runs}
      </p>
      <p className="plan-label">test runs delivered every order</p>
      <p className="plan-detail">
        {result.flat.toLocaleString('en-US')} of {result.robots.toLocaleString('en-US')} robots ran flat
      </p>
    </article>
  );
}

function Proof({ recording, summary }) {
  const fleets = [...new Set(summary.map((row) => row.robots))];
  const fleetWords = `${fleets.slice(0, -1).join(', ')} and ${fleets[fleets.length - 1]}`;
  return (
    <section className="home-section">
      <div className="home-inner">
        <h2>Same warehouse. One change to the battery.</h2>
        <p className="home-lead">
          Two plans for the same large floor, each tested with {fleetWords} robots. Plan A can
          travel a typical round trip on a full charge, so it looks reasonable on paper. The
          simulator shows it cannot work.
        </p>
        <div className="plan-grid">
          <PlanResult
            name="Plan A"
            moves={recording.moves_on_full_charge[FAILS]}
            result={tally(summary, FAILS)}
          />
          <PlanResult
            name="Plan B"
            moves={recording.moves_on_full_charge[WORKS]}
            result={tally(summary, WORKS)}
          />
        </div>
        <p className="home-note">
          Simulated runs, on test conditions kept apart from tuning.{' '}
          <a href="#/research">See how the project is measured</a>
        </p>
      </div>
    </section>
  );
}

export default function HomePage() {
  const [recording, setRecording] = useState(null);
  const [reliability, setReliability] = useState(null);

  useEffect(() => {
    let cancelled = false;
    loadFleetStudy('home_floor')
      .then((data) => !cancelled && setRecording(data))
      .catch(() => {});
    loadFleetStudy('reliability')
      .then((data) => !cancelled && setReliability(data))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="home">
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-copy">
            <h1>Find out if your robot plan works before you buy the robots.</h1>
            <p>
              {BRAND} tests a warehouse robot plan in software: the floor, the fleet, the batteries
              and the orders. You see whether the work gets done, and where it breaks.
            </p>
            <div className="hero-actions">
              {/* The live floor needs the simulator, which the public site
                  does not host, so it only leads where it will work. */}
              <a className="button primary" href={backendConfigured ? '#/watch' : '#/test'}>
                {backendConfigured ? 'Watch it run' : 'Test a plan'}
              </a>
              <a className="button ghost" href={backendConfigured ? '#/test' : '#/watch'}>
                {backendConfigured ? 'Test a plan' : 'Watch it run'}
              </a>
            </div>
          </div>
          {recording ? (
            <RecordedFloor recording={recording} />
          ) : (
            <div className="hero-floor is-empty" />
          )}
        </div>
      </section>

      <section className="home-section">
        <div className="home-inner">
          <h2>Find the problems while they are still cheap</h2>
          <p className="home-lead">
            A robot plan that looks fine on paper can still jam or run flat on the floor. {BRAND} is
            for the people making that decision: warehouse planners and the companies that install
            robots.
          </p>
          <div className="question-grid">
            {QUESTIONS.map((question) => (
              <article key={question.title}>
                <h3>{question.title}</h3>
                <p>{question.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="home-section tinted">
        <div className="home-inner">
          <h2>How it works</h2>
          <ol className="step-grid">
            {STEPS.map((step, index) => (
              <li key={step.title}>
                <span className="step-number num">{index + 1}</span>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {recording && reliability ? (
        <Proof recording={recording} summary={reliability.summary} />
      ) : null}

      <section className="home-section tinted">
        <div className="home-inner home-split">
          <div>
            <h2>Where the AI comes in</h2>
            <p>
              This began as a research question: can a robot that learns by trial and error
              (reinforcement learning) drive as well as a classic route planner? That work is still
              here, with every result, and it is the academic heart of the project.
            </p>
            <a className="text-link" href="#/research">Read the findings</a>
          </div>
          <div>
            <h2>What this is not</h2>
            <p>
              {BRAND} is a software simulation from a final-year engineering project. It uses no
              physical robots and has not yet been compared with a real warehouse, so its results
              show how a plan behaves in the model. They are not a guarantee.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
