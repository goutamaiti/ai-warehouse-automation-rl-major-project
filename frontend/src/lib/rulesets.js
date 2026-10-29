/**
 * Non-layout rules (battery, reward weights, tasks) for the sandbox editor.
 *
 * These mirror configs/default.yaml and configs/scenarios/battery_constrained.yaml
 * exactly - they are not invented numbers. Keep them in sync if those files
 * change. Dynamic obstacles are not offered here because the editor paints a
 * static map only; a scenario with moving traffic still runs for real through
 * the backend's "Run a new episode" panel on the Dashboard tab.
 */

export const RULESETS = {
  default: {
    label: 'Default',
    source: 'configs/default.yaml',
    maxSteps: 400,
    battery: {
      capacity: 100.0,
      startLevel: 100.0,
      moveCost: 0.5,
      idleCost: 0.05,
      chargeRate: 8.0,
      lowThreshold: 30.0,
      criticalThreshold: 10.0,
    },
    tasks: { tasksPerEpisode: 1, minSeparation: 6 },
    reward: {
      step_penalty: 0.05,
      progress_weight: 1.0,
      shaping_gamma: 1.0,
      pickup_reward: 5.0,
      delivery_reward: 20.0,
      collision_penalty: 1.0,
      wait_penalty: 0.1,
      energy_weight: 0.02,
      battery_depleted_penalty: 10.0,
      timeout_penalty: 5.0,
    },
  },
  battery_constrained: {
    label: 'Battery constrained',
    source: 'configs/scenarios/battery_constrained.yaml',
    maxSteps: 600,
    battery: {
      capacity: 100.0,
      startLevel: 45.0,
      moveCost: 1.5,
      idleCost: 0.1,
      chargeRate: 10.0,
      lowThreshold: 35.0,
      criticalThreshold: 15.0,
    },
    tasks: { tasksPerEpisode: 2, minSeparation: 6 },
    // Reward weights are inherited unchanged from default.yaml.
    reward: {
      step_penalty: 0.05,
      progress_weight: 1.0,
      shaping_gamma: 1.0,
      pickup_reward: 5.0,
      delivery_reward: 20.0,
      collision_penalty: 1.0,
      wait_penalty: 0.1,
      energy_weight: 0.02,
      battery_depleted_penalty: 10.0,
      timeout_penalty: 5.0,
    },
  },
  battery_tuned: {
    label: 'Battery constrained (fixed reward)',
    source: 'configs/scenarios/battery_tuned.yaml',
    maxSteps: 600,
    // Identical world to battery_constrained - only the reward differs.
    battery: {
      capacity: 100.0,
      startLevel: 45.0,
      moveCost: 1.5,
      idleCost: 0.1,
      chargeRate: 10.0,
      lowThreshold: 35.0,
      criticalThreshold: 15.0,
    },
    tasks: { tasksPerEpisode: 2, minSeparation: 6 },
    reward: {
      step_penalty: 0.05,
      progress_weight: 1.0,
      shaping_gamma: 1.0,
      pickup_reward: 5.0,
      delivery_reward: 20.0,
      collision_penalty: 1.0,
      // Raised from 0.1: a 600-step camp on a charger has to cost more than
      // the depletion penalty it avoids, or stalling becomes the safe option.
      wait_penalty: 0.5,
      energy_weight: 0.02,
      // Raised from 10: a dying episode could bank up to 54.6 beforehand, so
      // at 10 the agent was paid to deliver one package and expire.
      battery_depleted_penalty: 60.0,
      timeout_penalty: 5.0,
      battery_reserve_weight: 10.0,
      battery_reserve_safety: 2.0,
    },
  },
};

export const DEFAULT_RULESET = 'default';

/** Plain-language description of every reward/penalty term, for the legend. */
export const REWARD_TERMS = [
  { key: 'step_penalty', label: 'Step cost', sign: '−', tone: 'penalty', describe: (w) => `−${w.step_penalty} every step`, why: 'Discourages wasted movement; the shortest solution earns the most.' },
  { key: 'progress', label: 'Progress', sign: '±', tone: 'shaping', describe: () => 'proportional to distance closed toward the target', why: 'Potential-based shaping (Ng et al., 1999): rewards moving closer to the pickup or drop-off. Policy-invariant for the undiscounted return; PPO discounts at 0.99, so only approximately for what it actually optimises (see the Info tab).' },
  { key: 'pickup', label: 'Pickup', sign: '+', tone: 'reward', describe: (w) => `+${w.pickup_reward} once, on reaching the storage point`, why: 'Marks the first half of a delivery as complete.' },
  { key: 'delivery', label: 'Delivery', sign: '+', tone: 'reward', describe: (w) => `+${w.delivery_reward} once, on reaching the packing station`, why: 'The main objective signal - this is what the task is actually for.' },
  { key: 'collision', label: 'Collision', sign: '−', tone: 'penalty', describe: (w) => `−${w.collision_penalty} per bump`, why: 'Driving into a wall, shelf or obstacle. No movement happens either.' },
  { key: 'wait', label: 'Wait', sign: '−', tone: 'penalty', describe: (w) => `−${w.wait_penalty} per step spent waiting`, why: 'Small penalty so the agent only waits when it has a real reason to (e.g. an obstacle blocking the only aisle, or charging).' },
  { key: 'energy', label: 'Energy', sign: '−', tone: 'penalty', describe: (w) => `−${w.energy_weight} × battery % used this step`, why: 'Ties reward to energy efficiency, not just step count.' },
  { key: 'battery_reserve', label: 'Battery reserve', sign: '±', tone: 'shaping', describe: (w) => (w.battery_reserve_weight ? `weight ${w.battery_reserve_weight}, safety ×${w.battery_reserve_safety}` : 'not active in this ruleset'), why: 'A second potential-based term: the charge the robot is short of what it needs to still reach a charger. Zero while a charger is comfortably in reach, so it only speaks up in the danger zone. Being potential-based it changes how fast a good policy is found, not which one is optimal - exactly so for the undiscounted return, approximately under the 0.99 discount PPO uses.' },
  { key: 'battery_depleted', label: 'Battery depleted', sign: '−', tone: 'penalty', describe: (w) => `−${w.battery_depleted_penalty}, episode ends in failure`, why: 'The battery hit 0% before every task was delivered. Must cost more than everything an episode can bank beforehand, or dying becomes profitable - which is exactly what happened at the original −10.' },
  { key: 'timeout', label: 'Timeout', sign: '−', tone: 'penalty', describe: (w) => `−${w.timeout_penalty}, episode ends in failure`, why: 'The step budget ran out before every task was delivered.' },
];
