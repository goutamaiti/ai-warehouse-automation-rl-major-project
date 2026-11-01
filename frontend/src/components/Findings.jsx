/**
 * The three things the research found, in plain words.
 *
 * The sentences are functions of the loaded summaries, like the takeaways on
 * the method page, so a rerun of the experiments cannot leave them stale.
 */

const QUIET_FLOORS = ['simple_static', 'default', 'complex_static'];

function byController(results, scenario) {
  const block = results.find((entry) => entry.scenario === scenario);
  return Object.fromEntries((block?.summaries ?? []).map((s) => [s.controller, s]));
}

const mean = (entry) => entry?.mean ?? null;

export default function Findings({ results }) {
  const loaded = results.some((entry) => entry.summaries?.length);
  if (!loaded) return null;

  const ties = QUIET_FLOORS.filter((scenario) => {
    const { ppo, astar } = byController(results, scenario);
    return ppo && astar && mean(ppo.steps) === mean(astar.steps);
  }).length;

  const traffic = byController(results, 'dynamic_obstacles');
  const quicker = mean(traffic.astar?.steps) - mean(traffic.ppo?.steps);

  const before = byController(results, 'battery_constrained').ppo;
  const after = byController(results, 'battery_tuned').ppo;

  return (
    <section className="findings">
      <article>
        <h3>Quiet floors: a tie</h3>
        <p className="finding-score num">{ties} of {QUIET_FLOORS.length}</p>
        <p>
          floors ended in an exact tie. The trained AI drove routes as short as the route planner,
          while seeing only the 5×5 cells around it.
        </p>
      </article>
      <article>
        <h3>Moving traffic: quicker, but riskier</h3>
        <p className="finding-score num">{quicker.toFixed(1)} steps</p>
        <p>
          quicker than the route planner on average. It paid with{' '}
          {mean(traffic.ppo?.collisions)?.toFixed(2)}{' '}
          collisions a run, against {mean(traffic.astar?.collisions)?.toFixed(2)}.
        </p>
      </article>
      <article>
        <h3>Tight battery: learned after a fix</h3>
        <p className="finding-score num">
          {before?.n_successful} → {after?.n_successful} of {after?.n_episodes}
        </p>
        <p>
          runs finished before and after one correction. The first version earned more by running
          flat than by charging, so that is what it learned to do.
        </p>
      </article>
    </section>
  );
}
