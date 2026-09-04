"""Visualisation of the simulation: ASCII for the terminal, JSON for the web UI.

Rendering is intentionally a *read-only consumer* of the simulation state. No
function in this module may modify the world, so training can run with
rendering switched off and still behave identically.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import WarehouseSimulation
from .warehouse import CellType

#: Character used for each cell type when rendering to text.
CELL_GLYPHS: dict[int, str] = {
    CellType.EMPTY: ".",
    CellType.WALL: "#",
    CellType.SHELF: "=",
    CellType.STORAGE: "o",
    CellType.PACKING: "P",
    CellType.CHARGING: "C",
}

ROBOT_GLYPH = "R"
ROBOT_LOADED_GLYPH = "@"
OBSTACLE_GLYPH = "X"
PICKUP_GLYPH = "p"
DROPOFF_GLYPH = "d"


def render_ascii(sim: WarehouseSimulation) -> str:
    """Render the current state as text.

    Legend: ``#`` wall, ``=`` shelf, ``o`` storage, ``P`` packing station,
    ``C`` charging station, ``p`` active pickup, ``d`` active drop-off,
    ``X`` dynamic obstacle, ``R`` robot, ``@`` robot carrying a package.
    """
    grid = sim.layout.grid
    canvas = [[CELL_GLYPHS[int(cell)] for cell in row] for row in grid]

    task = sim.task
    canvas[task.pickup[0]][task.pickup[1]] = PICKUP_GLYPH
    canvas[task.dropoff[0]][task.dropoff[1]] = DROPOFF_GLYPH
    for obstacle in sim.obstacles.obstacles:
        canvas[obstacle.position[0]][obstacle.position[1]] = OBSTACLE_GLYPH
    row, col = sim.robot.position
    canvas[row][col] = ROBOT_LOADED_GLYPH if sim.robot.carrying else ROBOT_GLYPH

    header = (
        f"step={sim.counters.steps:>4}  battery={sim.robot.battery:6.1f}%  "
        f"task={task.status.value:<10} dist={sim.distance_to_target():>3}"
    )
    return header + "\n" + "\n".join("".join(line) for line in canvas)


@dataclass
class EpisodeRecorder:
    """Collects one frame per step so an episode can be replayed later.

    The dashboard replays exactly these files, which means the animation the
    examiner sees is a recording of a real run, never a mock-up.
    """

    controller: str
    scenario: str
    seed: int
    layout: dict[str, Any]
    reward_config: dict[str, Any] = field(default_factory=dict)
    frames: list[dict] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_simulation(
        cls, sim: WarehouseSimulation, controller: str, seed: int
    ) -> EpisodeRecorder:
        return cls(
            controller=controller,
            scenario=sim.config.name,
            seed=seed,
            layout=sim.layout.to_dict(),
            # The exact weights that produced every reward number in this
            # recording, so a viewer (dashboard, report) never has to guess or
            # duplicate them from a config file.
            reward_config=dataclasses.asdict(sim.config.reward),
        )

    def capture(
        self,
        sim: WarehouseSimulation,
        events: list[str] | None = None,
        reward: float | None = None,
        reward_components: dict[str, float] | None = None,
    ) -> None:
        """Append one frame.

        ``reward``/``reward_components`` are the values the environment
        computed for the step that produced this frame - the same numbers
        shown in the dashboard's reward panel. The very first frame (the
        state right after reset) carries neither, since no action has
        happened yet.
        """
        frame = sim.snapshot(events)
        if reward is not None:
            frame["reward"] = round(float(reward), 4)
        if reward_components is not None:
            frame["reward_components"] = {
                key: round(float(value), 4) for key, value in reward_components.items()
            }
        self.frames.append(frame)

    def finalise(self, sim: WarehouseSimulation, extra: dict | None = None) -> None:
        self.summary = {
            "controller": self.controller,
            "scenario": self.scenario,
            "seed": self.seed,
            **sim.counters.to_dict(),
            **(extra or {}),
        }

    def to_dict(self) -> dict:
        return {
            "controller": self.controller,
            "scenario": self.scenario,
            "seed": self.seed,
            "layout": self.layout,
            "reward_config": self.reward_config,
            "frames": self.frames,
            "summary": self.summary,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1)
        return path
