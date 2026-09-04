"""Breadth-first search planning: the uninformed counterpart of A*.

On this grid every move costs one, so BFS returns paths of exactly the same
length as A*. It is kept as a second baseline because the *number of expanded
nodes* differs, which is the measurable evidence for the heuristic's value.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from simulation.navigation import DIRECTIONS, Position, in_bounds

from .astar import PlanResult, _reconstruct


def bfs(blocked: np.ndarray, start: Position, goal: Position) -> PlanResult:
    """Plan a shortest path with breadth-first search."""
    if not in_bounds(start, blocked.shape) or not in_bounds(goal, blocked.shape):
        raise ValueError("start and goal must lie inside the grid")
    if blocked[goal]:
        return PlanResult(path=(), nodes_expanded=0, found=False)
    if start == goal:
        return PlanResult(path=(start,), nodes_expanded=0, found=True)

    queue: deque[Position] = deque([start])
    came_from: dict[Position, Position] = {}
    seen = {start}
    nodes_expanded = 0

    while queue:
        current = queue.popleft()
        nodes_expanded += 1
        if current == goal:
            return PlanResult(
                path=_reconstruct(came_from, current),
                nodes_expanded=nodes_expanded,
                found=True,
            )
        for d_row, d_col in DIRECTIONS:
            neighbour = (current[0] + d_row, current[1] + d_col)
            if not in_bounds(neighbour, blocked.shape) or blocked[neighbour]:
                continue
            if neighbour in seen:
                continue
            seen.add(neighbour)
            came_from[neighbour] = current
            queue.append(neighbour)

    return PlanResult(path=(), nodes_expanded=nodes_expanded, found=False)
