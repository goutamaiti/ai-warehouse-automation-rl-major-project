/**
 * Grid primitives shared by the warehouse editor and the offline (no-backend)
 * simulator: cell types, BFS distance fields, pathfinding (A-star and BFS),
 * and layout validation.
 *
 * This mirrors simulation/navigation.py and simulation/warehouse.py closely
 * enough that the same warehouse behaves the same way conceptually in both
 * places, but it is an independent implementation - the numbers reported as
 * the project's official results always come from the Python pipeline in
 * data/results/, never from this file. See lib/offlineSimulation.js.
 */

export const CellType = {
  EMPTY: 0,
  WALL: 1,
  SHELF: 2,
  STORAGE: 3,
  PACKING: 4,
  CHARGING: 5,
};

// CELL_NAMES and CELL_COLORS used to live here. Both are gone: naming and
// drawing a cell are now one job, done by lib/cellArt, because a colour map
// could only ever say what tint a cell is - and tint stopped telling these six
// apart the moment the site went to a single hue. cellArt gives each type a
// mark instead, and carries the label beside it so the two cannot drift.

/** Mirrors simulation/warehouse.py::MIN_CUSTOM_SIZE and MAX_CUSTOM_SIZE.
 *  Exported so the editor's resize clamp reads them from here rather than
 *  keeping its own third copy of the same two numbers. */
export const MIN_GRID_SIZE = 5;
export const MAX_GRID_SIZE = 40;

const BLOCKING = new Set([CellType.WALL, CellType.SHELF]);

// Up, down, left, right - same order as the discrete action space (0-3),
// action 4 is "wait" and has no direction.
export const DIRECTIONS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

export function inBounds(grid, row, col) {
  return row >= 0 && row < grid.length && col >= 0 && col < grid[0].length;
}

export function isWalkable(grid, row, col) {
  return inBounds(grid, row, col) && !BLOCKING.has(grid[row][col]);
}

export function neighbours(grid, [row, col]) {
  const result = [];
  for (const [dr, dc] of DIRECTIONS) {
    const r = row + dr;
    const c = col + dc;
    if (isWalkable(grid, r, c)) result.push([r, c]);
  }
  return result;
}

const key = (r, c) => `${r},${c}`;

/** Shortest-path distance (in moves) from every walkable cell to `goal`. -1 = unreachable. */
export function bfsDistanceField(grid, goal) {
  const height = grid.length;
  const width = grid[0].length;
  const field = Array.from({ length: height }, () => new Array(width).fill(-1));
  const [goalRow, goalCol] = goal;
  if (!isWalkable(grid, goalRow, goalCol)) return field;

  field[goalRow][goalCol] = 0;
  const queue = [goal];
  let head = 0;
  while (head < queue.length) {
    const current = queue[head];
    head += 1;
    const nextDistance = field[current[0]][current[1]] + 1;
    for (const [r, c] of neighbours(grid, current)) {
      if (field[r][c] === -1) {
        field[r][c] = nextDistance;
        queue.push([r, c]);
      }
    }
  }
  return field;
}

export function manhattan([r1, c1], [r2, c2]) {
  return Math.abs(r1 - r2) + Math.abs(c1 - c2);
}

/** Cells reachable from `start` on the walkable grid (ignores CellType, just connectivity). */
export function reachableCells(grid, start) {
  const field = bfsDistanceField(grid, start);
  const cells = new Set();
  field.forEach((row, r) => row.forEach((distance, c) => distance >= 0 && cells.add(key(r, c))));
  return cells;
}

/**
 * A* shortest path on a walkable grid, Manhattan heuristic. `blockedExtra` is
 * an optional set of "r,c" keys (e.g. cells the robot must not enter this
 * step) layered on top of the static walkability.
 */
export function astarPath(grid, start, goal, blockedExtra = null) {
  const [goalRow, goalCol] = goal;
  if (!isWalkable(grid, goalRow, goalCol)) return { found: false, path: [] };
  if (start[0] === goal[0] && start[1] === goal[1]) return { found: true, path: [start] };

  const open = [{ pos: start, g: 0, f: manhattan(start, goal) }];
  const cameFrom = new Map();
  const gScore = new Map([[key(...start), 0]]);
  const closed = new Set();

  while (open.length > 0) {
    let bestIndex = 0;
    for (let i = 1; i < open.length; i += 1) {
      if (open[i].f < open[bestIndex].f) bestIndex = i;
    }
    const current = open.splice(bestIndex, 1)[0];
    const currentKey = key(...current.pos);
    if (closed.has(currentKey)) continue;
    closed.add(currentKey);

    if (current.pos[0] === goalRow && current.pos[1] === goalCol) {
      const path = [current.pos];
      let cursor = currentKey;
      while (cameFrom.has(cursor)) {
        const [prev, prevPos] = cameFrom.get(cursor);
        path.push(prevPos);
        cursor = prev;
      }
      path.reverse();
      return { found: true, path };
    }

    for (const neighbour of neighbours(grid, current.pos)) {
      const nKey = key(...neighbour);
      if (blockedExtra?.has(nKey)) continue;
      const tentative = current.g + 1;
      if (tentative < (gScore.get(nKey) ?? Infinity)) {
        gScore.set(nKey, tentative);
        cameFrom.set(nKey, [currentKey, current.pos]);
        open.push({ pos: neighbour, g: tentative, f: tentative + manhattan(neighbour, goal) });
      }
    }
  }
  return { found: false, path: [] };
}

/** Breadth-first search path (same length as A* on a uniform-cost grid). */
export function bfsPath(grid, start, goal, blockedExtra = null) {
  const [goalRow, goalCol] = goal;
  if (!isWalkable(grid, goalRow, goalCol)) return { found: false, path: [] };
  if (start[0] === goal[0] && start[1] === goal[1]) return { found: true, path: [start] };

  const queue = [start];
  let head = 0;
  const cameFrom = new Map();
  const seen = new Set([key(...start)]);

  while (head < queue.length) {
    const current = queue[head];
    head += 1;
    if (current[0] === goalRow && current[1] === goalCol) {
      const path = [current];
      let cursor = key(...current);
      while (cameFrom.has(cursor)) {
        const [prev, prevPos] = cameFrom.get(cursor);
        path.push(prevPos);
        cursor = prev;
      }
      path.reverse();
      return { found: true, path };
    }
    for (const neighbour of neighbours(grid, current)) {
      const nKey = key(...neighbour);
      if (seen.has(nKey) || blockedExtra?.has(nKey)) continue;
      seen.add(nKey);
      cameFrom.set(nKey, [key(...current), current]);
      queue.push(neighbour);
    }
  }
  return { found: false, path: [] };
}

/** Every cell of a given type, in the same row-major sorted order Python uses. */
export function pointsOfType(grid, cellType) {
  const points = [];
  grid.forEach((row, r) => row.forEach((cell, c) => cell === cellType && points.push([r, c])));
  return points;
}

/**
 * Client-side pre-check mirroring simulation.warehouse.layout_from_grid /
 * validate_layout: every point of interest must be walkable and all points
 * must belong to one connected component. Returns { valid, error }.
 */
export function validateGrid(grid) {
  if (!grid.length || !grid[0]?.length) return { valid: false, error: 'the grid is empty.' };
  const height = grid.length;
  const width = grid[0].length;
  if (grid.some((row) => row.length !== width)) {
    return { valid: false, error: 'the rows have inconsistent width.' };
  }
  if (height < MIN_GRID_SIZE || width < MIN_GRID_SIZE) {
    return { valid: false, error: `the grid must be at least ${MIN_GRID_SIZE}x${MIN_GRID_SIZE}.` };
  }
  // The two checks below were missing, so this function said "valid" for grids
  // layout_from_grid refuses - and it claims above to mirror that function.
  // Measured: a 41x41 grid and grids containing cell values 99, -1 and 2.5 all
  // passed here, after which the offline sandbox ran them and reported step
  // counts, while a connected backend rejected the same grid outright. A
  // pre-check that green-lights a layout the real engine will not accept is
  // worse than no pre-check.
  if (height > MAX_GRID_SIZE || width > MAX_GRID_SIZE) {
    return { valid: false, error: `the grid must be at most ${MAX_GRID_SIZE}x${MAX_GRID_SIZE}.` };
  }
  const known = new Set(Object.values(CellType));
  const unknown = new Set();
  for (const row of grid) {
    for (const value of row) {
      if (!known.has(value)) unknown.add(value);
    }
  }
  if (unknown.size) {
    return {
      valid: false,
      error: `the grid contains unknown cell values: ${[...unknown].sort().join(', ')}.`,
    };
  }

  const storage = pointsOfType(grid, CellType.STORAGE);
  const packing = pointsOfType(grid, CellType.PACKING);
  const charging = pointsOfType(grid, CellType.CHARGING);
  if (!storage.length) return { valid: false, error: 'add at least one pick point.' };
  if (!packing.length) return { valid: false, error: 'add at least one packing station.' };
  if (!charging.length) return { valid: false, error: 'add at least one charger.' };

  const points = [...storage, ...packing, ...charging];
  for (const [r, c] of points) {
    if (!isWalkable(grid, r, c)) {
      return { valid: false, error: `the point of interest at (${r}, ${c}) is not walkable.` };
    }
  }

  const connected = reachableCells(grid, packing[0]);
  const unreachable = points.filter(([r, c]) => !connected.has(key(r, c)));
  if (unreachable.length > 0) {
    return {
      valid: false,
      error: `${unreachable.length} point(s) cannot be reached from the packing station - look for a wall sealing off part of the floor.`,
    };
  }
  return { valid: true, error: null };
}
