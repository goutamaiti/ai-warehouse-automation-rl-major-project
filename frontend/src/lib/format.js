/** Small formatting helpers shared by the panels and tables. */

/** Format a summary entry of the shape { mean, ci95, n } produced by analytics. */
export function formatStat(value) {
  if (value === null || value === undefined) return 'n/a';
  if (typeof value === 'number') return value.toFixed(2);
  if (typeof value === 'object') {
    const { mean, ci95, n } = value;
    if (mean === null || mean === undefined || !n) return 'n/a';
    return n > 1 ? `${mean.toFixed(2)} ± ${ci95.toFixed(2)}` : mean.toFixed(2);
  }
  return String(value);
}

export function formatPercent(fraction) {
  if (fraction === null || fraction === undefined) return 'n/a';
  return `${(fraction * 100).toFixed(0)}%`;
}

/** True when every task of the episode was delivered. */
export function episodeSucceeded(summary) {
  return summary?.termination_reason === 'all_tasks_delivered';
}

/**
 * Path efficiency = shortest possible path / path actually driven.
 *
 * Only defined for successful episodes: a robot that gave up half way through
 * has driven fewer moves than the shortest complete route, which would produce
 * a nonsensical efficiency above 100%.
 */
export function pathEfficiency(summary) {
  if (!summary || !episodeSucceeded(summary)) return null;
  const optimal = summary.optimal_path_length;
  const driven = summary.moves;
  if (!optimal || !driven) return null;
  return optimal / driven;
}

export function titleCase(text) {
  if (!text) return '';
  return text.charAt(0).toUpperCase() + text.slice(1).replace(/_/g, ' ');
}

export const CONTROLLER_LABELS = {
  ppo: 'PPO (learned)',
  astar: 'A* (classical)',
  bfs: 'BFS (classical)',
  random: 'Random',
};

export function controllerLabel(name) {
  return CONTROLLER_LABELS[name] ?? titleCase(name);
}
