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

export const CELL_NAMES = {
  0: 'Empty',
  1: 'Wall',
  2: 'Shelf',
  3: 'Storage point',
  4: 'Packing station',
  5: 'Charging station',
};

// Shared with the editor's palette so a cell looks the same whether you are
// painting it or watching a replay drive over it. Aisles stay near-black so
// the things that matter (shelves, stations, robot) carry the contrast.
export const CELL_COLORS = {
  [CellType.EMPTY]: '#0d1526',
  [CellType.WALL]: '#04060b',
  [CellType.SHELF]: '#3a475e',
  [CellType.STORAGE]: '#1e40af',
  [CellType.PACKING]: '#15803d',
  [CellType.CHARGING]: '#b45309',
};

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
  if (!grid.length || !grid[0]?.length) return { valid: false, error: 'Grid is empty.' };
  const height = grid.length;
  const width = grid[0].length;
  if (grid.some((row) => row.length !== width)) {
    return { valid: false, error: 'Rows have inconsistent width.' };
  }
  if (height < 5 || width < 5) {
    return { valid: false, error: 'Grid must be at least 5x5.' };
  }

  const storage = pointsOfType(grid, CellType.STORAGE);
  const packing = pointsOfType(grid, CellType.PACKING);
  const charging = pointsOfType(grid, CellType.CHARGING);
  if (!storage.length) return { valid: false, error: 'Add at least one storage point.' };
  if (!packing.length) return { valid: false, error: 'Add at least one packing station.' };
  if (!charging.length) return { valid: false, error: 'Add at least one charging station.' };

  const points = [...storage, ...packing, ...charging];
  for (const [r, c] of points) {
    if (!isWalkable(grid, r, c)) {
      return { valid: false, error: `Point of interest at (${r}, ${c}) is not walkable.` };
    }
  }

  const connected = reachableCells(grid, packing[0]);
  const unreachable = points.filter(([r, c]) => !connected.has(key(r, c)));
  if (unreachable.length > 0) {
    return {
      valid: false,
      error: `${unreachable.length} point(s) are unreachable from the packing station - check for a wall sealing off part of the map.`,
    };
  }
  return { valid: true, error: null };
}
