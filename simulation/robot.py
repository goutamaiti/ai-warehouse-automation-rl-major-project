"""Virtual warehouse robot: position, battery and payload state.

The robot is deliberately "dumb": it holds state and applies energy
bookkeeping, but it never decides where to go. Decisions come either from the
RL policy or from a classical planner, both of which drive the robot through
:class:`simulation.engine.WarehouseSimulation`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import BatteryConfig
from .navigation import Position


@dataclass
class Robot:
    """Mutable state of a single virtual robot."""

    robot_id: int
    position: Position
    battery: float
    carrying: bool = False
    #: Total energy (battery percentage points) drawn since the last reset.
    energy_consumed: float = 0.0

    def consume(self, amount: float) -> None:
        """Draw energy from the battery, clamped at zero."""
        if amount < 0:
            raise ValueError("energy consumption must be non-negative")
        drawn = min(amount, self.battery)
        self.battery -= drawn
        self.energy_consumed += drawn

    def charge(self, config: BatteryConfig) -> float:
        """Charge for one time step, returning the amount actually gained."""
        gained = min(config.charge_rate, config.capacity - self.battery)
        self.battery += gained
        return gained

    def battery_fraction(self, config: BatteryConfig) -> float:
        return float(self.battery / config.capacity) if config.capacity else 0.0

    def is_depleted(self) -> bool:
        return self.battery <= 0.0

    def battery_state(self, config: BatteryConfig) -> str:
        """Coarse battery label used by the dashboard and the A* controller."""
        if self.battery <= config.critical_threshold:
            return "critical"
        if self.battery <= config.low_threshold:
            return "low"
        return "ok"

    def to_dict(self, config: BatteryConfig) -> dict:
        return {
            "id": self.robot_id,
            "position": list(self.position),
            "battery": round(self.battery, 2),
            "battery_state": self.battery_state(config),
            "carrying": self.carrying,
            "energy_consumed": round(self.energy_consumed, 2),
        }
