/**
 * Standalone (no-backend) simulator for the warehouse editor's "sandbox" mode.
 *
 * This is an independent JavaScript port of the physics in simulation/engine.py
 * and the reward in environment/warehouse_env.py, restricted to what the editor
 * needs: a single robot, a static hand-drawn map (no dynamic obstacles yet),
 * and the A*, BFS and Random controllers (PPO needs the trained network and
 * only runs through the backend - see App.jsx).
 *
 * IMPORTANT: this file produces a *sandbox* episode for exploring an edited
 * warehouse interactively. It is not part of the reported project results -
 * every number in data/results/ comes from the Python pipeline, seeded with
 * numpy's RNG, which this file does not attempt to reproduce bit-for-bit (it
 * uses its own small PRNG so a seed is still reproducible *within the
 * sandbox*, just not identical to a Python run of the same seed).
 */

import {
  CellType,
  DIRECTIONS,
  astarPath,
  bfsDistanceField,
  bfsPath,
  isWalkable,
  manhattan,
  pointsOfType,
} from './grid.js';
import { DEFAULT_RULESET, RULESETS } from './rulesets.js';

export const ACTION = { UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3, WAIT: 4 };
export const OFFLINE_CONTROLLERS = ['astar', 'bfs', 'random'];

const round2 = (x) => Math.round(x * 100) / 100;
const round3 = (x) => Math.round(x * 1000) / 1000;
const round4 = (x) => Math.round(x * 10000) / 10000;
const posKey = (p) => `${p[0]},${p[1]}`;
const samePos = (a, b) => a[0] === b[0] && a[1] === b[1];

function mulberry32(seed) {
  let t = seed >>> 0;
  return function next() {
    t |= 0;
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function buildLayoutDict(grid) {
  return {
    height: grid.length,
    width: grid[0].length,
    grid,
    legend: { empty: 0, wall: 1, shelf: 2, storage: 3, packing: 4, charging: 5 },
    storage_points: pointsOfType(grid, CellType.STORAGE),
    packing_stations: pointsOfType(grid, CellType.PACKING),
    charging_stations: pointsOfType(grid, CellType.CHARGING),
  };
}

// --- task ------------------------------------------------------------

function taskTarget(task) {
  if (task.status === 'pending') return task.pickup;
  if (task.status === 'picked_up') return task.dropoff;
  return task.pickedStep !== null ? task.dropoff : task.pickup;
}

function makeTaskGenerator(layoutDict, taskConfig, rng) {
  const { storage_points: storage, packing_stations: packing } = layoutDict;
  let nextId = 0;
  return function generate(step) {
    let pairs = [];
    for (const pickup of storage) {
      for (const dropoff of packing) {
        if (manhattan(pickup, dropoff) >= taskConfig.minSeparation) pairs.push([pickup, dropoff]);
      }
    }
    if (pairs.length === 0) {
      let best = [storage[0], packing[0]];
      let bestDistance = -1;
      for (const pickup of storage) {
        for (const dropoff of packing) {
          const distance = manhattan(pickup, dropoff);
          if (distance > bestDistance) {
            bestDistance = distance;
            best = [pickup, dropoff];
          }
        }
      }
      pairs = [best];
    }
    const [pickup, dropoff] = pairs[Math.floor(rng() * pairs.length)];
    const task = {
      id: nextId,
      pickup,
      dropoff,
      status: 'pending',
      createdStep: step,
      pickedStep: null,
      deliveredStep: null,
    };
    nextId += 1;
    return task;
  };
}

// --- robot -------------------------------------------------------------

function consume(robot, amount) {
  const drawn = Math.min(amount, robot.battery);
  robot.battery -= drawn;
  robot.energyConsumed += drawn;
}

function chargeRobot(robot, batteryCfg) {
  const gained = Math.min(batteryCfg.chargeRate, batteryCfg.capacity - robot.battery);
  robot.battery += gained;
  return gained;
}

function batteryState(robot, batteryCfg) {
  if (robot.battery <= batteryCfg.criticalThreshold) return 'critical';
  if (robot.battery <= batteryCfg.lowThreshold) return 'low';
  return 'ok';
}

// --- episode state -------------------------------------------------------

function optimalLegs(grid, start, task) {
  const toPickup = bfsDistanceField(grid, task.pickup)[start[0]][start[1]];
  const toDropoff = bfsDistanceField(grid, task.dropoff)[task.pickup[0]][task.pickup[1]];
  return Math.max(toPickup, 0) + Math.max(toDropoff, 0);
}

function createState(grid, ruleset, seed) {
  const layoutDict = buildLayoutDict(grid);
  const rng = mulberry32(seed);
  const chargingSorted = [...layoutDict.charging_stations].sort(
    (a, b) => a[0] - b[0] || a[1] - b[1]
  );
  const start = chargingSorted[0];

  const robot = {
    position: start,
    battery: Math.min(ruleset.battery.startLevel, ruleset.battery.capacity),
    carrying: false,
    energyConsumed: 0,
  };
  const genTask = makeTaskGenerator(layoutDict, ruleset.tasks, rng);
  const task = genTask(0);

  const state = {
    grid,
    layoutDict,
    ruleset,
    robot,
    task,
    tasksRemaining: ruleset.tasks.tasksPerEpisode - 1,
    genTask,
    done: false,
    counters: {
      steps: 0,
      moves: 0,
      collisions: 0,
      staticCollisions: 0,
      dynamicCollisions: 0,
      idleSteps: 0,
      energyConsumed: 0,
      chargingEvents: 0,
      tasksDelivered: 0,
      tasksFailed: 0,
      optimalPathLength: optimalLegs(grid, start, task),
      terminationReason: 'running',
    },
  };
  state.targetCache = taskTarget(task);
  state.targetField = bfsDistanceField(grid, state.targetCache);
  return state;
}

function distanceToTarget(state) {
  return state.targetField[state.robot.position[0]][state.robot.position[1]];
}

function refreshTargetField(state) {
  const target = taskTarget(state.task);
  if (!samePos(target, state.targetCache)) {
    state.targetField = bfsDistanceField(state.grid, target);
    state.targetCache = target;
  }
}

// --- one simulated step, mirrors simulation/engine.py::WarehouseSimulation.step ---

function step(state, action) {
  if (state.done) throw new Error('episode already finished');
  const battery = state.ruleset.battery;
  const outcome = {
    moved: false,
    blockedStatic: false,
    blockedObstacle: false,
    waited: false,
    pickedUp: false,
    delivered: false,
    charged: 0,
    energyUsed: 0,
    distanceBefore: distanceToTarget(state),
    distanceAfter: 0,
    batteryDepleted: false,
    allTasksDone: false,
    timedOut: false,
    events: [],
  };

  if (action === ACTION.WAIT) {
    outcome.waited = true;
    outcome.energyUsed = battery.idleCost;
    state.counters.idleSteps += 1;
  } else {
    const [dr, dc] = DIRECTIONS[action];
    const target = [state.robot.position[0] + dr, state.robot.position[1] + dc];
    if (!isWalkable(state.grid, target[0], target[1])) {
      outcome.blockedStatic = true;
      outcome.energyUsed = battery.idleCost;
      outcome.events.push('collision_static');
      state.counters.staticCollisions += 1;
    } else {
      state.robot.position = target;
      outcome.moved = true;
      outcome.energyUsed = battery.moveCost;
      state.counters.moves += 1;
    }
  }
  consume(state.robot, outcome.energyUsed);
  state.counters.energyConsumed += outcome.energyUsed;
  state.counters.collisions = state.counters.staticCollisions + state.counters.dynamicCollisions;

  const position = state.robot.position;
  if (state.task.status === 'pending' && samePos(position, state.task.pickup)) {
    state.task.status = 'picked_up';
    state.task.pickedStep = state.counters.steps;
    state.robot.carrying = true;
    outcome.pickedUp = true;
    outcome.events.push('pickup');
  } else if (state.task.status === 'picked_up' && samePos(position, state.task.dropoff)) {
    state.task.status = 'delivered';
    state.task.deliveredStep = state.counters.steps;
    state.robot.carrying = false;
    outcome.delivered = true;
    outcome.events.push('delivery');
    state.counters.tasksDelivered += 1;
    if (state.tasksRemaining > 0) {
      state.tasksRemaining -= 1;
      const nextTask = state.genTask(state.counters.steps);
      state.counters.optimalPathLength += optimalLegs(state.grid, position, nextTask);
      state.task = nextTask;
      outcome.events.push('new_task');
    } else {
      outcome.allTasksDone = true;
      state.counters.terminationReason = 'all_tasks_delivered';
    }
  }

  if (state.grid[position[0]][position[1]] === CellType.CHARGING) {
    const gained = chargeRobot(state.robot, battery);
    if (gained > 0) {
      outcome.charged = gained;
      outcome.events.push('charging');
      state.counters.chargingEvents += 1;
    }
  }

  state.counters.steps += 1;
  if (state.robot.battery <= 0) {
    outcome.batteryDepleted = true;
    outcome.events.push('battery_depleted');
    if (state.task.status === 'pending' || state.task.status === 'picked_up') {
      state.task.status = 'failed';
    }
    state.counters.tasksFailed += 1;
    state.counters.terminationReason = 'battery_depleted';
  } else if (state.counters.steps >= state.ruleset.maxSteps && !outcome.allTasksDone) {
    outcome.timedOut = true;
    outcome.events.push('timeout');
    if (state.task.status === 'pending' || state.task.status === 'picked_up') {
      state.task.status = 'failed';
    }
    state.counters.tasksFailed += 1;
    state.counters.terminationReason = 'timeout';
  }

  refreshTargetField(state);
  outcome.distanceAfter = distanceToTarget(state);
  outcome.terminated = outcome.batteryDepleted || outcome.allTasksDone;
  outcome.truncated = outcome.timedOut && !outcome.terminated;
  outcome.collided = outcome.blockedStatic || outcome.blockedObstacle;
  state.done = outcome.terminated || outcome.truncated;
  return outcome;
}

// --- reward, mirrors environment/warehouse_env.py::WarehouseEnv._reward ---

function computeReward(outcome, rewardCfg) {
  const targetChanged = outcome.pickedUp || outcome.events.includes('new_task');
  let shaping = 0;
  if (!targetChanged && outcome.distanceBefore >= 0 && outcome.distanceAfter >= 0) {
    const phiBefore = -outcome.distanceBefore;
    const phiAfter = -outcome.distanceAfter;
    shaping = rewardCfg.progress_weight * (rewardCfg.shaping_gamma * phiAfter - phiBefore);
  }
  const components = {
    step_penalty: -rewardCfg.step_penalty,
    progress: shaping,
    pickup: outcome.pickedUp ? rewardCfg.pickup_reward : 0,
    delivery: outcome.delivered ? rewardCfg.delivery_reward : 0,
    collision: outcome.collided ? -rewardCfg.collision_penalty : 0,
    wait: outcome.waited ? -rewardCfg.wait_penalty : 0,
    energy: -rewardCfg.energy_weight * outcome.energyUsed,
    battery_depleted: outcome.batteryDepleted ? -rewardCfg.battery_depleted_penalty : 0,
    timeout: outcome.truncated ? -rewardCfg.timeout_penalty : 0,
  };
  const total = Object.values(components).reduce((sum, value) => sum + value, 0);
  return { total, components };
}

// --- controllers ---------------------------------------------------------

function actionTowards([r, c], [nr, nc]) {
  const dr = nr - r;
  const dc = nc - c;
  const index = DIRECTIONS.findIndex(([ddr, ddc]) => ddr === dr && ddc === dc);
  if (index === -1) throw new Error('planner produced a non-adjacent move');
  return index;
}

function makePlannerController(name, grid, layoutDict, ruleset) {
  const planFn = name === 'bfs' ? bfsPath : astarPath;
  const chargerFields = layoutDict.charging_stations.map((station) => ({
    station,
    field: bfsDistanceField(grid, station),
  }));
  const safetyFactor = 1.2;
  const chargeUntil = 0.9;
  let charging = false;

  function nearestCharger(position) {
    let best = chargerFields[0].station;
    let bestDistance = Infinity;
    for (const { station, field } of chargerFields) {
      const distance = field[position[0]][position[1]];
      if (distance >= 0 && distance < bestDistance) {
        bestDistance = distance;
        best = station;
      }
    }
    return best;
  }

  function selectGoal(state) {
    const batteryCfg = state.ruleset.battery;
    const level = state.robot.battery;
    const target = taskTarget(state.task);

    if (charging) {
      if (level >= batteryCfg.capacity * chargeUntil) charging = false;
      else return nearestCharger(state.robot.position);
    }

    const distanceToGoal = distanceToTarget(state);
    let distanceGoalToCharger = Infinity;
    for (const { field } of chargerFields) {
      const distance = field[target[0]][target[1]];
      if (distance >= 0) distanceGoalToCharger = Math.min(distanceGoalToCharger, distance);
    }
    const required =
      (distanceToGoal + distanceGoalToCharger) * batteryCfg.moveCost * safetyFactor;
    const nearlyFull = level >= batteryCfg.capacity * chargeUntil;
    if (!nearlyFull && (level <= required || level <= batteryCfg.criticalThreshold)) {
      charging = true;
      return nearestCharger(state.robot.position);
    }
    return target;
  }

  return {
    name,
    act(state) {
      const goal = selectGoal(state);
      const position = state.robot.position;
      if (samePos(position, goal)) return ACTION.WAIT;
      const plan = planFn(grid, position, goal);
      if (!plan.found || plan.path.length < 2) return ACTION.WAIT;
      return actionTowards(position, plan.path[1]);
    },
  };
}

function makeRandomController(seed) {
  // A stream independent of the task generator's, mirroring how
  // baselines/controller.py::RandomPolicy owns its own RNG.
  const rng = mulberry32((seed ^ 0x9e3779b9) >>> 0);
  return { name: 'random', act: () => Math.floor(rng() * 5) };
}

function makeController(name, grid, layoutDict, ruleset, seed) {
  if (name === 'random') return makeRandomController(seed);
  if (name === 'astar' || name === 'bfs') return makePlannerController(name, grid, layoutDict, ruleset);
  throw new Error(`the offline sandbox does not support controller '${name}'`);
}

// --- frame / summary serialisation, matching EpisodeRecorder.to_dict() ---

function snapshotFrame(state, events = [], reward = null, components = null) {
  const frame = {
    step: state.counters.steps,
    robot: {
      id: 0,
      position: state.robot.position,
      battery: round2(state.robot.battery),
      battery_state: batteryState(state.robot, state.ruleset.battery),
      carrying: state.robot.carrying,
      energy_consumed: round2(state.robot.energyConsumed),
    },
    task: {
      id: state.task.id,
      pickup: state.task.pickup,
      dropoff: state.task.dropoff,
      status: state.task.status,
      target: taskTarget(state.task),
      created_step: state.task.createdStep,
      picked_step: state.task.pickedStep,
      delivered_step: state.task.deliveredStep,
    },
    obstacles: [],
    distance_to_target: distanceToTarget(state),
    events,
  };
  if (reward !== null) frame.reward = round4(reward);
  if (components !== null) {
    frame.reward_components = Object.fromEntries(
      Object.entries(components).map(([k, v]) => [k, round4(v)])
    );
  }
  return frame;
}

function countersToDict(counters) {
  return {
    steps: counters.steps,
    moves: counters.moves,
    collisions: counters.collisions,
    static_collisions: counters.staticCollisions,
    dynamic_collisions: counters.dynamicCollisions,
    idle_steps: counters.idleSteps,
    energy_consumed: round3(counters.energyConsumed),
    charging_events: counters.chargingEvents,
    tasks_delivered: counters.tasksDelivered,
    tasks_failed: counters.tasksFailed,
    optimal_path_length: counters.optimalPathLength,
    termination_reason: counters.terminationReason,
  };
}

/**
 * Run one full episode in the browser and return a replay object shaped
 * exactly like `EpisodeRecorder.to_dict()` from the Python side, so it can be
 * fed into the same WarehouseCanvas / PlaybackControls / MetricsPanel /
 * RewardPanel components used for real recordings.
 */
export function runOfflineEpisode({ grid, rulesetName = DEFAULT_RULESET, controllerName, seed = 0 }) {
  if (!OFFLINE_CONTROLLERS.includes(controllerName)) {
    throw new Error(`controller ${controllerName} is not available offline (needs a backend)`);
  }
  const ruleset = RULESETS[rulesetName] ?? RULESETS[DEFAULT_RULESET];
  const state = createState(grid, ruleset, seed);
  const controller = makeController(controllerName, grid, state.layoutDict, ruleset, seed);

  const frames = [snapshotFrame(state)];
  let totalReward = 0;
  const maxIterations = ruleset.maxSteps + 5; // safety net, should never be hit
  let iterations = 0;
  while (!state.done && iterations < maxIterations) {
    const action = controller.act(state);
    const outcome = step(state, action);
    const { total, components } = computeReward(outcome, ruleset.reward);
    totalReward += total;
    frames.push(snapshotFrame(state, outcome.events, total, components));
    iterations += 1;
  }

  const scenarioLabel = `sandbox: ${ruleset.label}`;
  return {
    controller: controllerName,
    scenario: scenarioLabel,
    seed,
    layout: state.layoutDict,
    reward_config: ruleset.reward,
    frames,
    summary: {
      controller: controllerName,
      scenario: scenarioLabel,
      seed,
      ...countersToDict(state.counters),
      total_reward: round3(totalReward),
    },
  };
}
