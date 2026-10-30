"""A robotic-fulfilment floor: dense pod field, one-way aisles, pick stations.

Why this exists alongside :func:`simulation.warehouse.build_layout`
------------------------------------------------------------------
``build_layout`` generates the floor every committed result in ``data/results/``
was measured on and every policy in ``rl_agent/models/`` was trained against.
It is frozen - ``tests/test_frozen_contract.py`` fails if its output moves by a
single cell - so a more realistic floor has to be a *second* generator, not a
change to that one. The six measured scenarios keep their warehouse; this one is
for the live fleet.

Where the structure comes from
------------------------------
The published layout of robotic mobile fulfilment systems (RMFS), the class of
system Amazon's drive-unit floors belong to, rather than invention:

* a dense **pod field** in blocks of six pods, which is what is deployed in
  practice;
* **one-way aisles** a single cell wide, with the direction alternating between
  parallel lanes. This is the part worth having: it is the structural answer to
  the standoff this project measured in Phase 2, where two robots met in a
  one-wide aisle with no detour and jammed. A yielding rule was built for that
  and measured out as useless; making the aisle one-way makes the standoff
  impossible instead of recoverable;
* wider, two-way **cross-aisles** cutting the field into blocks;
* **pick/stow stations** along one face behind a **buffer lane** that separates
  them from the field;
* a **charging bank** on its own face, and **docks** on the opposite one.

What is deliberately NOT adopted
--------------------------------
Real drive units carry a whole pod to a stationary picker. This project's task
is a robot fetching a *package* from a storage point and delivering it to a
packing station, and that stays: changing it would redefine the research
question - what a delivery is, what the reward pays for - rather than the floor
it happens on.

So pods here are shelving the robot drives *around*, and the aisle cells facing
a pod are the storage points it picks from, exactly as in the simpler layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .navigation import Position
from .warehouse import CellType, WarehouseLayout, validate_layout

#: Direction codes for a one-way cell. 0 means unrestricted (cross-aisles,
#: buffer lanes, station approaches), which every floor needs or robots could
#: not turn around at all.
FREE, RIGHT, LEFT, DOWN, UP = 0, 1, 2, 3, 4

#: Movement deltas by direction code, matching navigation.DIRECTIONS ordering
#: for the four cardinal moves.
_DELTA = {RIGHT: (0, 1), LEFT: (0, -1), DOWN: (1, 0), UP: (-1, 0)}


@dataclass(frozen=True)
class RmfsConfig:
    """Geometry of a robotic-fulfilment floor.

    The defaults produce a floor roughly the size of one robotic zone, large
    enough that congestion is a real effect rather than an artefact of a cramped
    grid - the 21x15 floor has only 175 free cells, which is where the fleet
    saturated at sixteen robots.
    """

    width: int = 46
    height: int = 30
    #: Pods per block, as a (depth, width) pair. Six per block is what the
    #: literature reports as deployed.
    block_depth: int = 3
    block_width: int = 2
    #: Every n-th block row becomes a two-way cross-aisle instead of pods.
    cross_aisle_every: int = 3
    #: Rows of the cross-aisle, which is wider than a picking aisle.
    cross_aisle_rows: int = 2
    #: Column the pick/stow stations sit on, and the width of the buffer lane
    #: between them and the pod field.
    station_col: int = 2
    buffer_width: int = 2
    #: Vertical gap between station pairs.
    station_pitch: int = 3
    #: Charging bays along the bottom face, and their spacing.
    charger_pitch: int = 5


def build_rmfs_layout(config: RmfsConfig | None = None) -> tuple[WarehouseLayout, np.ndarray]:
    """Generate a robotic-fulfilment floor.

    Returns the layout plus a ``direction`` grid of the same shape holding a
    direction code per cell. The layout itself is an ordinary
    :class:`~simulation.warehouse.WarehouseLayout` using the same
    :class:`~simulation.warehouse.CellType` values as every other floor - pods
    are ``SHELF``, stations are ``PACKING``, bays are ``CHARGING`` - so the
    engine, the fleet, dispatch, zones and the planners all work on it
    unchanged. Only a controller that wants to honour the one-way rule needs to
    read the direction grid.
    """
    cfg = config or RmfsConfig()
    if cfg.width < 20 or cfg.height < 14:
        raise ValueError("an RMFS floor needs at least 20x14 cells to fit a pod field")

    h, w = cfg.height, cfg.width
    grid = np.full((h, w), int(CellType.EMPTY), dtype=np.int8)
    direction = np.full((h, w), FREE, dtype=np.int8)

    grid[0, :] = grid[-1, :] = int(CellType.WALL)
    grid[:, 0] = grid[:, -1] = int(CellType.WALL)

    buffer_end = cfg.station_col + cfg.buffer_width          # last buffer column
    field_c0 = buffer_end + 2
    field_c1 = w - 3
    field_r0 = 2
    field_r1 = h - 4

    # --- pod field, in blocks, with cross-aisles -------------------------
    row_step = cfg.block_depth + 1          # block plus its picking aisle
    col_step = cfg.block_width + 1
    block_row = 0
    cross_rows: list[int] = []
    for row in range(field_r0, field_r1, row_step):
        block_row += 1
        if block_row % cfg.cross_aisle_every == 0:
            for extra in range(cfg.cross_aisle_rows):
                if row + extra < field_r1:
                    cross_rows.append(row + extra)
            continue
        for col in range(field_c0, field_c1, col_step):
            for d_row in range(cfg.block_depth):
                for d_col in range(cfg.block_width):
                    r, c = row + d_row, col + d_col
                    if r < field_r1 and c < field_c1:
                        grid[r, c] = int(CellType.SHELF)

    # --- one-way picking aisles, alternating by lane ---------------------
    # A lane is a row or column of the field carrying no pods. Horizontal lanes
    # alternate left/right by block; vertical lanes alternate up/down by column.
    for r in range(field_r0, field_r1):
        if r in cross_rows:
            continue
        if any(grid[r, c] == int(CellType.SHELF) for c in range(field_c0, field_c1)):
            continue
        code = RIGHT if ((r - field_r0) // row_step) % 2 == 0 else LEFT
        for c in range(field_c0, field_c1):
            if grid[r, c] == int(CellType.EMPTY):
                direction[r, c] = code

    for c in range(field_c0, field_c1):
        if any(grid[r, c] == int(CellType.SHELF) for r in range(field_r0, field_r1)):
            continue
        code = DOWN if ((c - field_c0) // col_step) % 2 == 0 else UP
        for r in range(field_r0, field_r1):
            if grid[r, c] == int(CellType.EMPTY) and direction[r, c] == FREE:
                direction[r, c] = code

    # Cross-aisles stay two-way, which is what lets a robot reverse direction.
    for r in cross_rows:
        direction[r, :] = FREE

    # Junctions are free, which is how a one-way street grid actually works:
    # you cannot turn mid-block, you turn at the intersection. Without this the
    # lanes are parallel corridors with no connections between them - routes
    # still exist, via the cross-aisles, but they are enormous detours.
    # Measured with locked junctions: all 96 intersections one-axis-only, paths
    # still reachable, and a twelve-robot fleet delivered 2 packages against 19
    # for the two-way floor. Freeing them is what makes one-way usable.
    horizontal_lanes = {
        r
        for r in range(h)
        if any(direction[r, c] in (RIGHT, LEFT) for c in range(w))
    }
    vertical_lanes = {
        c
        for c in range(w)
        if any(direction[r, c] in (DOWN, UP) for r in range(h))
    }
    for r in horizontal_lanes:
        for c in vertical_lanes:
            if grid[r, c] != int(CellType.WALL) and grid[r, c] != int(CellType.SHELF):
                direction[r, c] = FREE

    # --- storage points: the aisle cells facing a pod --------------------
    storage: list[Position] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != int(CellType.EMPTY):
                continue
            if c < field_c0:
                continue
            touches_pod = any(
                0 <= r + dr < h
                and 0 <= c + dc < w
                and grid[r + dr, c + dc] == int(CellType.SHELF)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if touches_pod:
                grid[r, c] = int(CellType.STORAGE)
                storage.append((r, c))

    # --- pick/stow stations behind a buffer lane -------------------------
    packing: list[Position] = []
    for r in range(4, h - 4, cfg.station_pitch):
        grid[r, cfg.station_col] = int(CellType.PACKING)
        packing.append((r, cfg.station_col))

    # --- charging bank on the bottom face --------------------------------
    charging: list[Position] = []
    for c in range(field_c0 + 1, field_c1, cfg.charger_pitch):
        grid[h - 2, c] = int(CellType.CHARGING)
        charging.append((h - 2, c))

    if not storage or not packing or not charging:  # pragma: no cover - geometry guard
        raise ValueError("RMFS geometry produced an unusable floor; check the config")

    layout = WarehouseLayout(
        grid=grid,
        storage_points=tuple(sorted(storage)),
        packing_stations=tuple(sorted(packing)),
        charging_stations=tuple(sorted(charging)),
    )
    # Same validation every other floor passes: every point of interest must be
    # walkable and reachable from the others.
    validate_layout(layout)
    return layout, direction


def allows(direction: np.ndarray, origin: Position, destination: Position) -> bool:
    """May a robot move from ``origin`` to ``destination`` under the one-way rule?

    The rule is a property of the cell being *left*: a cell marked RIGHT may
    only be exited rightwards. ``FREE`` cells - cross-aisles, the buffer lane,
    station and charger approaches - permit any move, which is what lets a robot
    turn around at all.
    """
    code = int(direction[origin])
    if code == FREE:
        return True
    delta = (destination[0] - origin[0], destination[1] - origin[1])
    return delta == _DELTA[code]


def directed_blocked_grid(
    blocked: np.ndarray, direction: np.ndarray, origin: Position
) -> np.ndarray:
    """Blocked mask for a planner that is about to step out of ``origin``.

    Cells the one-way rule forbids leaving ``origin`` for are marked blocked, so
    an ordinary shortest-path search produces a legal first move without the
    planner needing to know the rule exists. Only the first step is constrained,
    which is enough: the controller replans every tick, so every step it takes
    is checked against the cell it is actually leaving.
    """
    result = blocked.copy()
    code = int(direction[origin])
    if code == FREE:
        return result
    allowed = (origin[0] + _DELTA[code][0], origin[1] + _DELTA[code][1])
    height, width = blocked.shape
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cell = (origin[0] + d_row, origin[1] + d_col)
        if not (0 <= cell[0] < height and 0 <= cell[1] < width):
            continue
        if cell != allowed:
            result[cell] = True
    return result


def direction_summary(direction: np.ndarray) -> dict:
    """Counts per direction code, for tests and the dashboard."""
    names = {FREE: "free", RIGHT: "right", LEFT: "left", DOWN: "down", UP: "up"}
    return {
        name: int((direction == code).sum()) for code, name in names.items()
    }


def astar_directed(
    blocked: np.ndarray,
    direction: np.ndarray,
    start: Position,
    goal: Position,
) -> tuple[Position, ...]:
    """Shortest path that obeys the one-way rule at *every* step.

    Filtering only the first move does not work, and the failure is instructive:
    an undirected search returns a route that is illegal from its second step,
    the controller replans, proposes the same illegal route again, and the robot
    thrashes in place. Measured with first-step-only filtering, a twelve-robot
    fleet delivered 1 package against 19 for the unconstrained version.

    The rule has to be inside the search, so this expands a node's successors
    through ``direction`` - a cell marked RIGHT has exactly one successor - and
    returns a path every step of which is legal. Manhattan distance stays an
    admissible heuristic: one-way streets only ever make the true cost longer.
    """
    import heapq

    height, width = blocked.shape
    if not (0 <= goal[0] < height and 0 <= goal[1] < width) or blocked[goal]:
        return ()
    if start == goal:
        return (start,)

    def heuristic(cell: Position) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    counter = 0
    heap: list[tuple[int, int, Position]] = [(heuristic(start), 0, start)]
    came: dict[Position, Position] = {}
    best: dict[Position, int] = {start: 0}
    seen: set[Position] = set()

    while heap:
        _, _, current = heapq.heappop(heap)
        if current in seen:
            continue
        seen.add(current)
        if current == goal:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            return tuple(reversed(path))

        code = int(direction[current])
        moves = (
            ((-1, 0), (1, 0), (0, -1), (0, 1)) if code == FREE else (_DELTA[code],)
        )
        for d_row, d_col in moves:
            nxt = (current[0] + d_row, current[1] + d_col)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if blocked[nxt]:
                continue
            tentative = best[current] + 1
            if tentative < best.get(nxt, 1 << 30):
                best[nxt] = tentative
                came[nxt] = current
                counter += 1
                heapq.heappush(heap, (tentative + heuristic(nxt), counter, nxt))
    return ()
