"""A* shortest-path planning on the warehouse grid.

A* is the classical baseline the learned policy is compared against. It is
given exactly the information a planner would have in a real deployment: the
static map plus the currently visible dynamic obstacles.

The planner also reports how many nodes it expanded, so the "informed vs
uninformed search" comparison against :mod:`baselines.bfs` is backed by a
measured number rather than an assertion.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from simulation.navigation import DIRECTIONS, Position, in_bounds, manhattan


@dataclass(frozen=True)
class PlanResult:
    """A plan plus the search statistics that produced it."""

    path: tuple[Position, ...]
    nodes_expanded: int
    found: bool

    @property
    def length(self) -> int:
        """Number of moves in the plan (a path of n cells needs n-1 moves)."""
        return max(len(self.path) - 1, 0)


def astar(
    blocked: np.ndarray,
    start: Position,
    goal: Position,
) -> PlanResult:
    """Plan a shortest path from ``start`` to ``goal``.

    Args:
        blocked: boolean grid, ``True`` where a cell cannot be entered.
        start: current robot cell (may itself be marked blocked).
        goal: destination cell.

    Returns:
        A :class:`PlanResult`; ``found`` is ``False`` when no path exists, in
        which case ``path`` is empty.
    """
    if not in_bounds(start, blocked.shape) or not in_bounds(goal, blocked.shape):
        raise ValueError("start and goal must lie inside the grid")
    if blocked[goal]:
        return PlanResult(path=(), nodes_expanded=0, found=False)
    if start == goal:
        return PlanResult(path=(start,), nodes_expanded=0, found=True)

    counter = 0  # tie-breaker keeping the heap comparison total and stable
    open_heap: list[tuple[int, int, Position]] = [(manhattan(start, goal), counter, start)]
    came_from: dict[Position, Position] = {}
    g_score: dict[Position, int] = {start: 0}
    closed: set[Position] = set()
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
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
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbour, 1 << 30):
                g_score[neighbour] = tentative
                came_from[neighbour] = current
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative + manhattan(neighbour, goal), counter, neighbour),
                )

    return PlanResult(path=(), nodes_expanded=nodes_expanded, found=False)


def _reconstruct(came_from: dict[Position, Position], current: Position) -> tuple[Position, ...]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return tuple(reversed(path))
