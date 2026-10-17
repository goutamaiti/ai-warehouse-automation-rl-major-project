"""Zones: named sections of the warehouse floor.

Why zones exist
---------------
Two jobs, both about where robots are when they are *not* carrying anything:

* **Idle robots need somewhere to be.** Parked in an aisle they block traffic;
  parked on a charger they block power for a robot that actually needs it. A
  zone gives each idle robot a designated waiting cell, which spreads the fleet
  across the floor so the next request has someone nearby.
* **The floor stops being one undifferentiated grid.** Reporting throughput or
  congestion per zone says *where* a warehouse is busy, which a single
  fleet-wide number cannot.

Zones are *not* assignment boundaries. Dispatch picks the nearest idle robot
anywhere on the floor, because that is what the project asks for: whichever
robot is closest to the product gets it. A robot whose waiting point is in zone
A is free to serve a request in zone B and then wait in A again.

How a zone is derived
---------------------
One zone per charging station, as a Voronoi partition of the walkable floor by
*geodesic* distance - distance through the aisles, not straight-line, because a
cell on the far side of a shelf block is not nearby in any useful sense. Ties go
to the lower station index so the partition is a pure function of the layout.

Deriving zones from the map rather than declaring them in config is deliberate:
a hand-written region list would silently stop matching the moment a layout
changed, and this project generates its layouts from parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .navigation import Position, bfs_distance_field
from .warehouse import WarehouseLayout


@dataclass(frozen=True)
class Zone:
    """One named section of the floor."""

    name: str
    #: The charging station this zone is built around.
    charger: Position
    #: Every walkable cell nearer to this zone's charger than to any other.
    cells: frozenset[Position]
    #: Where idle robots wait, nearest-first. Never a station cell, so a waiting
    #: robot blocks neither power nor a drop-off. More than one, because a zone
    #: may hold several idle robots at once and they cannot share a cell.
    waiting_points: tuple[Position, ...]

    def contains(self, position: Position) -> bool:
        return position in self.cells

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "charger": list(self.charger),
            "n_cells": len(self.cells),
            "waiting_points": [list(p) for p in self.waiting_points],
        }


def _zone_name(index: int) -> str:
    """A, B, C, ... then Z1, Z2 beyond the alphabet."""
    return chr(ord("A") + index) if index < 26 else f"Z{index + 1}"


def build_zones(layout: WarehouseLayout, max_waiting_points: int = 8) -> tuple[Zone, ...]:
    """Partition the walkable floor into one zone per charging station.

    ``max_waiting_points`` bounds how many idle robots a single zone can park;
    a fleet larger than ``len(zones) * max_waiting_points`` will find zones
    without a free waiting cell, which the fleet layer handles by leaving those
    robots where they stand rather than failing.
    """
    stations = list(layout.charging_stations)
    if not stations:  # pragma: no cover - validate_layout rejects this already
        raise ValueError("cannot build zones: the layout has no charging stations")

    walkable = layout.walkable_mask()
    fields = [bfs_distance_field(walkable, station) for station in stations]
    reserved = (
        set(layout.charging_stations)
        | set(layout.packing_stations)
        | set(layout.storage_points)
    )

    members: list[list[Position]] = [[] for _ in stations]
    for row, col in zip(*np.nonzero(walkable)):
        cell = (int(row), int(col))
        reachable = [
            (int(field[cell]), index)
            for index, field in enumerate(fields)
            if field[cell] >= 0
        ]
        if not reachable:  # pragma: no cover - layout validation prevents this
            continue
        # min() breaks ties on the second element, so the lower station index
        # wins and the partition stays a pure function of the layout.
        members[min(reachable)[1]].append(cell)

    zones: list[Zone] = []
    for index, station in enumerate(stations):
        cells = members[index]
        field = fields[index]
        # Waiting cells: closest to this zone's charger first, so idle robots
        # sit near power without occupying it. Stations are excluded, and
        # storage points too - waiting on a pickup cell would block a delivery.
        candidates = sorted(
            (cell for cell in cells if cell not in reserved),
            key=lambda cell: (int(field[cell]), cell),
        )
        zones.append(
            Zone(
                name=_zone_name(index),
                charger=station,
                cells=frozenset(cells),
                waiting_points=tuple(candidates[:max_waiting_points]),
            )
        )
    return tuple(zones)


def zone_of(zones: tuple[Zone, ...], position: Position) -> Zone | None:
    """Which zone a cell belongs to, or ``None`` if it is not on the floor."""
    for zone in zones:
        if zone.contains(position):
            return zone
    return None
