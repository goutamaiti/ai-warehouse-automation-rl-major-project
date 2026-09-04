"""Grid primitives shared by the simulator, the RL environment and the baselines.

Keeping these helpers in one place guarantees that the learned policy and the
classical planners move on exactly the same grid with the same neighbourhood
definition, which is a precondition for a fair comparison.
"""

from __future__ import annotations

from collections import deque

import numpy as np

#: A grid position expressed as ``(row, column)``.
Position = tuple[int, int]

#: 4-connected neighbourhood, in the same order as the discrete action space
#: (see :mod:`environment.warehouse_env`): up, down, left, right.
DIRECTIONS: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Value used in a distance field for cells that cannot be reached.
UNREACHABLE = -1


def in_bounds(pos: Position, shape: tuple[int, int]) -> bool:
    """Return True if pos lies inside a grid of the given shape."""
    row, col = pos
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def neighbours(pos: Position, walkable: np.ndarray) -> list[Position]:
    """Return the walkable 4-connected neighbours of pos."""
    row, col = pos
    result: list[Position] = []
    for d_row, d_col in DIRECTIONS:
        candidate = (row + d_row, col + d_col)
        if in_bounds(candidate, walkable.shape) and walkable[candidate]:
            result.append(candidate)
    return result


def manhattan(a: Position, b: Position) -> int:
    """Manhattan distance: the admissible heuristic used by A* on this grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_distance_field(walkable: np.ndarray, goal: Position) -> np.ndarray:
    """Shortest-path distance from every walkable cell to goal.

    Because all moves cost the same, a breadth-first search gives exact
    shortest-path distances in O(cells).  The field is used for

    * potential-based reward shaping in the RL environment, and
    * the "optimal path length" reference in the analytics module.

    Unreachable cells (and blocked cells) hold UNREACHABLE.
    """
    if not in_bounds(goal, walkable.shape):
        raise ValueError(f"goal {goal} outside grid of shape {walkable.shape}")
    field = np.full(walkable.shape, UNREACHABLE, dtype=np.int32)
    if not walkable[goal]:
        return field
    field[goal] = 0
    queue: deque[Position] = deque([goal])
    while queue:
        current = queue.popleft()
        next_distance = field[current] + 1
        for neighbour in neighbours(current, walkable):
            if field[neighbour] == UNREACHABLE:
                field[neighbour] = next_distance
                queue.append(neighbour)
    return field


def reachable_cells(walkable: np.ndarray, start: Position) -> set[Position]:
    """Set of cells reachable from start (used for layout validation)."""
    field = bfs_distance_field(walkable, start)
    rows, cols = np.nonzero(field != UNREACHABLE)
    return {(int(r), int(c)) for r, c in zip(rows, cols)}
