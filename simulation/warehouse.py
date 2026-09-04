"""Static warehouse layout: the map the robots drive on.

The layout is *static* by definition - walls, shelves and stations never move
during an episode. Everything that changes over time (robot, obstacles, tasks)
lives in :mod:`simulation.engine`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np

from .config import LayoutConfig
from .navigation import Position, reachable_cells


class CellType(IntEnum):
    """Content of a single grid cell."""

    EMPTY = 0
    WALL = 1
    SHELF = 2
    STORAGE = 3   # aisle cell in front of a shelf: where packages are picked up
    PACKING = 4   # delivery / packing station
    CHARGING = 5  # charging station


#: Cell types a robot can never enter.
BLOCKING_CELLS: frozenset[int] = frozenset({CellType.WALL, CellType.SHELF})


@dataclass(frozen=True)
class WarehouseLayout:
    """An immutable warehouse map plus the semantic points of interest."""

    grid: np.ndarray  # shape (height, width), dtype int8, values of CellType
    storage_points: tuple[Position, ...]
    packing_stations: tuple[Position, ...]
    charging_stations: tuple[Position, ...]

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def cell(self, pos: Position) -> CellType:
        return CellType(int(self.grid[pos]))

    def walkable_mask(self) -> np.ndarray:
        """Boolean mask of cells a robot may occupy (ignores dynamic traffic)."""
        return ~np.isin(self.grid, list(BLOCKING_CELLS))

    def is_walkable(self, pos: Position) -> bool:
        row, col = pos
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return int(self.grid[pos]) not in BLOCKING_CELLS

    def to_dict(self) -> dict:
        """JSON-serialisable form consumed by the API and the dashboard."""
        return {
            "height": self.height,
            "width": self.width,
            "grid": self.grid.astype(int).tolist(),
            "legend": {member.name.lower(): int(member) for member in CellType},
            "storage_points": [list(p) for p in self.storage_points],
            "packing_stations": [list(p) for p in self.packing_stations],
            "charging_stations": [list(p) for p in self.charging_stations],
        }


def build_layout(config: LayoutConfig) -> WarehouseLayout:
    """Generate a warehouse map from a :class:`LayoutConfig`.

    Structure produced (walls #, shelves S, charging C, packing P)::

        #####################
        #...................#   <- top aisle
        #.SSS.SSS.SSS.SSS...#   <- shelf blocks separated by aisles
        #.SSS.SSS.SSS.SSS...#
        #...................#
        #C.....P.....P.....C#   <- station row
        #####################

    Aisle cells that touch a shelf become STORAGE points, i.e. the positions
    where a package can be picked up.
    """
    height, width = config.height, config.width
    if height < 7 or width < 7:
        raise ValueError("warehouse must be at least 7x7 to fit shelves and stations")

    grid = np.full((height, width), int(CellType.EMPTY), dtype=np.int8)
    grid[0, :] = grid[-1, :] = int(CellType.WALL)
    grid[:, 0] = grid[:, -1] = int(CellType.WALL)

    # --- shelf blocks -----------------------------------------------------
    first_row = 1 + config.top_aisle
    last_shelf_row = height - 2 - config.station_rows
    row_step = config.shelf_block_height + config.aisle_width
    col_step = config.shelf_block_width + config.aisle_width

    block_rows = [
        r
        for r in range(first_row, last_shelf_row + 1, row_step)
        if r + config.shelf_block_height - 1 <= last_shelf_row
    ]
    block_cols = [
        c
        for c in range(2, width - 2, col_step)
        if c + config.shelf_block_width - 1 <= width - 3
    ]
    if not block_rows or not block_cols:
        raise ValueError(
            "layout leaves no room for shelf blocks; reduce the block size or "
            "increase the warehouse dimensions"
        )
    # Centre the shelf grid so that the leftover space becomes an aisle on both
    # sides instead of one wide empty strip on the right.
    block_rows = _centre(block_rows, config.shelf_block_height, last_shelf_row)
    block_cols = _centre(block_cols, config.shelf_block_width, width - 3)
    for row in block_rows:
        for col in block_cols:
            grid[
                row : row + config.shelf_block_height,
                col : col + config.shelf_block_width,
            ] = int(CellType.SHELF)

    # --- storage points (aisle cells touching a shelf) --------------------
    storage: list[Position] = []
    for row in block_rows:
        for col in block_cols:
            above = (row - 1, col)
            below = (row + config.shelf_block_height, col + config.shelf_block_width - 1)
            for candidate in (above, below):
                if grid[candidate] == int(CellType.EMPTY):
                    grid[candidate] = int(CellType.STORAGE)
                    storage.append(candidate)

    # --- stations ---------------------------------------------------------
    station_row = height - 2
    charging: list[Position] = []
    for col in _spread(1, width - 2, config.n_charging_stations, edges_first=True):
        grid[station_row, col] = int(CellType.CHARGING)
        charging.append((station_row, col))

    packing: list[Position] = []
    # Packing stations sit between the chargers, with a margin so that a robot
    # leaving a charger never starts on top of a station.
    candidates = _spread(4, max(4, width - 5), config.n_packing_stations)
    candidates += [c for c in range(1, width - 1) if c not in candidates]
    for col in candidates:
        if len(packing) == config.n_packing_stations:
            break
        if grid[station_row, col] == int(CellType.EMPTY):
            grid[station_row, col] = int(CellType.PACKING)
            packing.append((station_row, col))

    if len(packing) < config.n_packing_stations:
        raise ValueError("not enough free cells in the station row for packing stations")

    layout = WarehouseLayout(
        grid=grid,
        storage_points=tuple(sorted(storage)),
        packing_stations=tuple(sorted(packing)),
        charging_stations=tuple(sorted(charging)),
    )
    validate_layout(layout)
    return layout


def _centre(starts: list[int], block_size: int, limit: int) -> list[int]:
    """Shift a list of block start indices so the leftover gap is split evenly."""
    if not starts:
        return starts
    leftover = limit - (starts[-1] + block_size - 1)
    shift = max(0, leftover // 2)
    return [s + shift for s in starts]


def _spread(low: int, high: int, count: int, edges_first: bool = False) -> list[int]:
    """Evenly spread count integer positions across the range [low, high]."""
    if count <= 0:
        return []
    if count == 1:
        return [(low + high) // 2]
    if edges_first and count == 2:
        return [low, high]
    step = (high - low) / (count - 1)
    return [int(round(low + i * step)) for i in range(count)]


def validate_layout(layout: WarehouseLayout) -> None:
    """Raise ValueError if the map is unusable for navigation.

    A layout is only valid when every point of interest is walkable and all of
    them belong to a single connected component - otherwise an episode could
    generate a task that no planner (classical or learned) can ever complete,
    which would silently corrupt the success-rate metric.
    """
    points = layout.storage_points + layout.packing_stations + layout.charging_stations
    if not points:
        raise ValueError("layout has no points of interest")
    for pos in points:
        if not layout.is_walkable(pos):
            raise ValueError(f"point of interest {pos} is not walkable")

    walkable = layout.walkable_mask()
    connected = reachable_cells(walkable, layout.packing_stations[0])
    unreachable = [p for p in points if p not in connected]
    if unreachable:
        raise ValueError(f"points unreachable from the packing station: {unreachable}")


#: Grid dimensions accepted from a user-drawn (not procedurally generated)
#: layout. The upper bound keeps a single API request cheap to simulate.
MIN_CUSTOM_SIZE = 5
MAX_CUSTOM_SIZE = 40


def layout_from_grid(grid: Sequence[Sequence[int]]) -> WarehouseLayout:
    """Build a :class:`WarehouseLayout` from a raw grid of cell-type integers.

    This is the entry point for a user-drawn warehouse (the dashboard's
    editor): unlike :func:`build_layout`, nothing is generated - every cell
    value must already be one of :class:`CellType`, and storage/packing/
    charging points are read directly off the grid instead of being placed by
    the procedural layout algorithm. The same :func:`validate_layout` used for
    generated layouts still runs, so a hand-drawn warehouse gets the same
    connectivity and walkability guarantees as a procedural one.
    """
    array = np.asarray(grid, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("grid must be a 2D array of cell-type integers")
    height, width = array.shape
    if height < MIN_CUSTOM_SIZE or width < MIN_CUSTOM_SIZE:
        raise ValueError(f"grid must be at least {MIN_CUSTOM_SIZE}x{MIN_CUSTOM_SIZE}")
    if height > MAX_CUSTOM_SIZE or width > MAX_CUSTOM_SIZE:
        raise ValueError(f"grid must be at most {MAX_CUSTOM_SIZE}x{MAX_CUSTOM_SIZE}")

    valid_values = {int(member) for member in CellType}
    used_values = set(np.unique(array).tolist())
    unknown = used_values - valid_values
    if unknown:
        raise ValueError(f"grid contains unknown cell values: {sorted(unknown)}")

    def points_of(cell_type: CellType) -> tuple[Position, ...]:
        rows, cols = np.nonzero(array == int(cell_type))
        return tuple(sorted((int(r), int(c)) for r, c in zip(rows, cols)))

    storage = points_of(CellType.STORAGE)
    packing = points_of(CellType.PACKING)
    charging = points_of(CellType.CHARGING)
    if not storage:
        raise ValueError("layout needs at least one storage point")
    if not packing:
        raise ValueError("layout needs at least one packing station")
    if not charging:
        raise ValueError("layout needs at least one charging station")

    layout = WarehouseLayout(
        grid=array.astype(np.int8),
        storage_points=storage,
        packing_stations=packing,
        charging_stations=charging,
    )
    validate_layout(layout)
    return layout
