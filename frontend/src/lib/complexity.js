/**
 * Scenario ordering by complexity, for every list the dashboard renders.
 *
 * The ordering is not cosmetic - it is the argument the project makes about
 * its own experiments, so each step adds exactly one new demand on a
 * controller, and each claim below traces to a measured file in data/results.
 *
 * Axes, in the order they start to bite:
 *
 *   1-3. Scale.        Bigger grid, longer shortest path. A planner needs
 *                      nothing new, only more search. A* stays optimal
 *                      throughout (path efficiency 1.00 on all three).
 *     4. Reactivity.   Moving obstacles mean the static shortest path is no
 *                      longer safe. This is the first scenario where A* stops
 *                      being optimal (efficiency drops 1.00 -> 0.93).
 *     5. Resource.     Two tasks need ~59.8 moves but the starting charge only
 *                      buys 30, so charging is mandatory rather than optional.
 *                      The first scenario a controller can fail purely by
 *                      running out, independent of navigation skill - and the
 *                      only one where PPO ever scored 0/30. *
 * The two battery_* entries share one identical *world* and differ only in the
 * agent's reward, which is why they carry the same `tier`: battery_constrained
 * is the original reward the policy could not solve, battery_tuned the
 * corrected one. Ranking them against each other would compare objectives,
 * not tasks.
 */

/** Ascending complexity. `tier` is the world; equal tiers are equally hard. */
export const SCENARIO_COMPLEXITY = [
  { name: 'simple_static', tier: 1, adds: 'Smallest grid, no obstacles, ample battery' },
  { name: 'default', tier: 2, adds: 'Baseline grid, longer paths' },
  { name: 'complex_static', tier: 3, adds: 'Largest static grid, densest shelving' },
  { name: 'dynamic_obstacles', tier: 4, adds: 'Moving traffic - the static shortest path stops being safe' },
  { name: 'battery_constrained', tier: 5, adds: 'Two tasks on a charge that covers half of one - charging is mandatory' },
  { name: 'battery_tuned', tier: 5, adds: 'Same world, corrected reward - the battery fix' },
];

const RANK = new Map(SCENARIO_COMPLEXITY.map((s, i) => [s.name, i]));

/** Classical first, then the learned policy, so each table reads as a comparison. */
const CONTROLLER_RANK = { astar: 0, bfs: 1, ppo: 2, ppo_charger: 3, random: 4 };

/** Rank of a scenario; unknown scenarios sort to the end rather than vanish. */
export function scenarioRank(name) {
  return RANK.has(name) ? RANK.get(name) : RANK.size;
}

export function scenarioTier(name) {
  return SCENARIO_COMPLEXITY.find((s) => s.name === name)?.tier ?? null;
}

export function scenarioAdds(name) {
  return SCENARIO_COMPLEXITY.find((s) => s.name === name)?.adds ?? '';
}

export function controllerRank(name) {
  return CONTROLLER_RANK[name] ?? 99;
}

/** Sort episode/result rows by scenario complexity, then by controller. */
export function byComplexity(a, b) {
  const scenario = scenarioRank(a.scenario) - scenarioRank(b.scenario);
  if (scenario !== 0) return scenario;
  const controller = controllerRank(a.controller) - controllerRank(b.controller);
  if (controller !== 0) return controller;
  return (a.seed ?? 0) - (b.seed ?? 0);
}
