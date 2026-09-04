"""Tests for the static layout: geometry, semantics and validation."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import LayoutConfig
from simulation.navigation import bfs_distance_field, reachable_cells
from simulation.warehouse import CellType, WarehouseLayout, build_layout, validate_layout


def test_layout_has_requested_dimensions_and_closed_walls():
    config = LayoutConfig()
    layout = build_layout(config)
    assert layout.shape == (config.height, config.width)
    assert (layout.grid[0, :] == CellType.WALL).all()
    assert (layout.grid[-1, :] == CellType.WALL).all()
    assert (layout.grid[:, 0] == CellType.WALL).all()
    assert (layout.grid[:, -1] == CellType.WALL).all()


def test_points_of_interest_are_walkable_and_distinct():
    layout = build_layout(LayoutConfig())
    points = layout.storage_points + layout.packing_stations + layout.charging_stations
    assert len(points) == len(set(points)), "a cell was assigned two roles"
    assert all(layout.is_walkable(point) for point in points)
    assert len(layout.storage_points) > 0
    assert len(layout.packing_stations) == LayoutConfig().n_packing_stations
    assert len(layout.charging_stations) == LayoutConfig().n_charging_stations


def test_every_point_of_interest_is_mutually_reachable():
    layout = build_layout(LayoutConfig())
    connected = reachable_cells(layout.walkable_mask(), layout.charging_stations[0])
    points = layout.storage_points + layout.packing_stations + layout.charging_stations
    assert set(points).issubset(connected)


def test_storage_points_touch_a_shelf():
    layout = build_layout(LayoutConfig())
    for row, col in layout.storage_points:
        neighbours = [
            layout.grid[row + dr, col + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= row + dr < layout.height and 0 <= col + dc < layout.width
        ]
        assert CellType.SHELF in neighbours


@pytest.mark.parametrize(
    "config",
    [
        LayoutConfig(width=31, height=21, shelf_block_height=3),
        LayoutConfig(width=15, height=11, n_packing_stations=1),
        LayoutConfig(width=25, height=17, aisle_width=2),
    ],
)
def test_alternative_layouts_are_valid(config: LayoutConfig):
    validate_layout(build_layout(config))


def test_too_small_layout_is_rejected():
    with pytest.raises(ValueError):
        build_layout(LayoutConfig(width=5, height=5))


def test_validation_catches_a_sealed_off_station():
    layout = build_layout(LayoutConfig())
    grid = layout.grid.copy()
    station = layout.packing_stations[-1]
    # Wall in every neighbour of the station: it becomes an island.
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        grid[station[0] + d_row, station[1] + d_col] = int(CellType.WALL)
    sealed = WarehouseLayout(
        grid=grid,
        storage_points=layout.storage_points,
        packing_stations=layout.packing_stations,
        charging_stations=layout.charging_stations,
    )
    with pytest.raises(ValueError, match="unreachable"):
        validate_layout(sealed)


def test_distance_field_matches_manual_bfs_on_a_tiny_grid():
    walkable = np.array(
        [
            [True, True, True],
            [False, False, True],
            [True, True, True],
        ]
    )
    field = bfs_distance_field(walkable, (0, 0))
    assert field[0, 0] == 0
    assert field[0, 2] == 2
    assert field[2, 0] == 6  # forced detour around the wall row
    assert field[1, 0] == -1  # blocked cell stays unreachable


def test_layout_serialises_to_json_friendly_types():
    payload = build_layout(LayoutConfig()).to_dict()
    assert payload["height"] and payload["width"]
    assert isinstance(payload["grid"][0][0], int)
    assert payload["legend"]["charging"] == int(CellType.CHARGING)


# --- layout_from_grid: user-drawn warehouses (dashboard editor) -----------

from simulation.warehouse import layout_from_grid


def _small_valid_grid() -> list[list[int]]:
    # 7x9, border walls, one shelf, one storage cell facing it, one packing
    # station and one charging station, all mutually reachable.
    W, K = int(CellType.WALL), int(CellType.EMPTY)
    S, T, P, C = int(CellType.SHELF), int(CellType.STORAGE), int(CellType.PACKING), int(CellType.CHARGING)
    grid = [
        [W, W, W, W, W, W, W, W, W],
        [W, K, K, K, K, K, K, K, W],
        [W, K, S, S, K, K, K, K, W],
        [W, K, T, K, K, K, K, K, W],
        [W, K, K, K, K, K, K, K, W],
        [W, P, K, K, K, K, K, C, W],
        [W, W, W, W, W, W, W, W, W],
    ]
    return grid


def test_layout_from_grid_accepts_a_valid_hand_drawn_warehouse():
    layout = layout_from_grid(_small_valid_grid())
    assert layout.storage_points == ((3, 2),)
    assert layout.packing_stations == ((5, 1),)
    assert layout.charging_stations == ((5, 7),)


def test_layout_from_grid_rejects_unknown_cell_values():
    grid = _small_valid_grid()
    grid[1][1] = 99
    with pytest.raises(ValueError, match="unknown cell values"):
        layout_from_grid(grid)


def test_layout_from_grid_rejects_ragged_rows():
    grid = _small_valid_grid()
    grid[2] = grid[2][:-1]
    with pytest.raises(ValueError):
        layout_from_grid(grid)


@pytest.mark.parametrize(
    "remove_type,message",
    [
        (int(CellType.STORAGE), "storage point"),
        (int(CellType.PACKING), "packing station"),
        (int(CellType.CHARGING), "charging station"),
    ],
)
def test_layout_from_grid_requires_every_point_type(remove_type, message):
    grid = _small_valid_grid()
    grid = [[int(CellType.EMPTY) if cell == remove_type else cell for cell in row] for row in grid]
    with pytest.raises(ValueError, match=message):
        layout_from_grid(grid)


def test_layout_from_grid_rejects_an_unreachable_station():
    grid = _small_valid_grid()
    # Wall off the charging station on all four sides.
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        grid[5 + dr][7 + dc] = int(CellType.WALL)
    with pytest.raises(ValueError, match="unreachable"):
        layout_from_grid(grid)


def test_layout_from_grid_rejects_too_small_a_grid():
    with pytest.raises(ValueError):
        layout_from_grid([[0] * 4] * 4)


def test_layout_from_grid_rejects_an_oversized_grid():
    from simulation.warehouse import MAX_CUSTOM_SIZE

    with pytest.raises(ValueError):
        layout_from_grid([[0] * (MAX_CUSTOM_SIZE + 1)] * (MAX_CUSTOM_SIZE + 1))


def test_shelf_forces_a_detour_in_a_custom_hand_drawn_layout():
    """End-to-end proof that a user-painted shelf blocks the robot: A* must
    route around it and the driven path never lands on a shelf cell. Mirrors
    the equivalent check in frontend/src/lib/grid.js so both implementations
    agree on the same guarantee.
    """
    from baselines.astar import astar
    from simulation.engine import Action, WarehouseSimulation
    from simulation.config import ScenarioConfig

    W, K = int(CellType.WALL), int(CellType.EMPTY)
    S, T, P, C = int(CellType.SHELF), int(CellType.STORAGE), int(CellType.PACKING), int(CellType.CHARGING)
    grid = [
        [W, W, W, W, W, W, W],
        [W, C, K, S, K, T, W],
        [W, K, K, S, K, K, W],
        [W, K, K, K, K, K, W],
        [W, P, K, K, K, K, W],
        [W, W, W, W, W, W, W],
    ]
    layout = layout_from_grid(grid)

    plan = astar(~layout.walkable_mask(), (1, 1), (1, 5))
    assert plan.found
    assert all(layout.cell(cell) is not CellType.SHELF for cell in plan.path)

    sim = WarehouseSimulation(ScenarioConfig(max_steps=200), layout=layout)
    outcome = sim.step(Action.RIGHT)  # straight into the shelf at (1, 3)... via (1,2) first
    assert sim.robot.position == (1, 2)  # one free cell before the shelf
    outcome = sim.step(Action.RIGHT)
    assert outcome.blocked_by_static
    assert sim.robot.position == (1, 2)
