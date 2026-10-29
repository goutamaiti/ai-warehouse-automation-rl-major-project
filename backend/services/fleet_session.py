"""A warehouse that keeps running, and the requests people drop into it.

Why this is not ``run_live_episode``
------------------------------------
``simulation_service.run_live_episode`` runs a whole episode and hands back the
finished recording for the browser to replay. That is the right shape for "show
me what happened", and it is how every number in ``data/results/`` is produced.

It cannot do what the dashboard's product-location card needs: a floor that is
*already running* when someone adds a product, so the nearest free robot starts
moving on the next tick. That needs a simulation that outlives a single request,
which is what a :class:`FleetSession` is.

What this module is careful about
---------------------------------
* **It never resets.** A delivery completes and the robot takes the next
  request; the warehouse does not stop between products.
* **It is not a source of reported results.** The committed tables come from the
  offline pipeline on fixed seeds. A session is interactive, its tick count
  depends on how long someone leaves the page open, and requests arrive when a
  person decides to add them - none of which is a controlled experiment. The API
  says so, so a live number is never mistaken for a measured one.
* **Only the classical controller drives it, for now.** The trained policies
  have never seen another robot, and continuous operation is off-distribution
  for them a second time. Wiring them in before that is stated plainly would
  produce a comparison that looks fair and is not.
"""

from __future__ import annotations

from typing import Any

from baselines.fleet_controller import FleetPlannerController
from simulation.config import ScenarioConfig
from simulation.dispatch import TaskQueue
from simulation.fleet import FleetSimulation
from simulation.navigation import Position
from simulation.rmfs import RmfsConfig, build_rmfs_layout, direction_summary
from simulation.warehouse import CellType, WarehouseLayout

#: Upper bound on queued-but-unserved requests. Not a limit on the warehouse -
#: a bound on what one browser session can pile up, so a stuck page cannot grow
#: the queue without end.
MAX_PENDING_REQUESTS = 500

#: Ticks a single advance() call may cover, so one HTTP request cannot occupy
#: the server for an unbounded time.
MAX_TICKS_PER_CALL = 200


class FleetSessionError(Exception):
    """Raised for conditions the API turns into a 4xx response."""


class FleetSession:
    """One continuously running warehouse floor."""

    def __init__(
        self,
        config: ScenarioConfig,
        n_robots: int = 3,
        controller: str = "astar",
        layout: WarehouseLayout | None = None,
        seed: int = 0,
        floor: str = "default",
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.floor = floor
        #: One-way lane directions, when the floor has them. Only the controller
        #: reads this; the engine's movement rules are unchanged, so a policy is
        #: never silently overridden by the floor.
        self.directions = None
        if floor == "rmfs":
            if layout is not None:
                raise FleetSessionError(
                    "the realistic floor generates its own layout; it cannot be "
                    "combined with a hand-drawn one"
                )
            layout, self.directions = build_rmfs_layout(RmfsConfig())
        elif floor != "default":
            raise FleetSessionError(
                f"unknown floor {floor!r}; expected 'default' or 'rmfs'"
            )
        self.queue = TaskQueue()
        try:
            self.sim = FleetSimulation(
                config,
                n_robots=n_robots,
                layout=layout,
                task_queue=self.queue,
                directions=self.directions,
            )
        except ValueError as exc:
            # Includes the unplaceable-fleet message, which names how many
            # robots the floor can actually hold.
            raise FleetSessionError(str(exc)) from exc
        self.sim.reset(seed=self.seed)
        try:
            self.controller = FleetPlannerController(
                planner=controller, directions=self.directions
            )
        except ValueError as exc:
            raise FleetSessionError(str(exc)) from exc
        self.controller.reset(self.sim)
        self.ticks = 0

    # ------------------------------------------------------------------
    # requests
    # ------------------------------------------------------------------
    def add_request(
        self, pickup: Position, dropoff: Position, priority: str = "normal"
    ) -> dict[str, Any]:
        """Queue a product. The nearest idle robot takes it on the next tick."""
        pending = len(self.queue.waiting())
        if pending >= MAX_PENDING_REQUESTS:
            raise FleetSessionError(
                f"{pending} requests are already waiting (limit "
                f"{MAX_PENDING_REQUESTS}); let the fleet catch up before adding more"
            )

        pickup = self._validate_cell(pickup, "pickup")
        dropoff = self._validate_cell(dropoff, "dropoff")
        if pickup == dropoff:
            raise FleetSessionError("pickup and dropoff cannot be the same cell")
        if self.sim.path_distance(pickup, dropoff) < 0:
            raise FleetSessionError(
                f"no route from pickup {list(pickup)} to dropoff {list(dropoff)}"
            )

        try:
            request = self.queue.add(pickup, dropoff, step=self.ticks, priority=priority)
        except ValueError as exc:
            raise FleetSessionError(str(exc)) from exc
        return request.to_dict()

    def _validate_cell(self, cell: Any, label: str) -> Position:
        """Reject anything a robot could not actually use, with a usable message."""
        # The length is checked before unpacking. Indexing cell[0] and cell[1]
        # alone catches "too few" through IndexError but accepts "too many" and
        # silently drops the rest: [1, 2, 3] was taken as [1, 2] and the caller
        # got a confident 200 for a cell they did not ask for. A coordinate with
        # a spare element is exactly the kind of thing this method exists to
        # refuse, so it says so instead of guessing.
        try:
            length = len(cell)
        except TypeError as exc:
            raise FleetSessionError(
                f"{label} must be a [row, column] pair, got {cell!r}"
            ) from exc
        if length != 2:
            plural = "value" if length == 1 else "values"
            raise FleetSessionError(
                f"{label} must be a [row, column] pair, got {length} {plural}: {cell!r}"
            )
        try:
            row, col = int(cell[0]), int(cell[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise FleetSessionError(
                f"{label} must be a [row, column] pair, got {cell!r}"
            ) from exc

        position = (row, col)
        height, width = self.sim.layout.shape
        if not (0 <= row < height and 0 <= col < width):
            raise FleetSessionError(
                f"{label} {list(position)} is outside the {height}x{width} warehouse"
            )
        if not self.sim.layout.is_walkable(position):
            kind = self.sim.layout.cell(position).name.lower()
            raise FleetSessionError(
                f"{label} {list(position)} is a {kind} cell; a robot cannot stand there"
            )
        return position

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def tick(self) -> dict[str, Any]:
        """Advance one tick and return the resulting frame."""
        if self.sim.done:
            # A finished floor is reported, not silently restarted: whoever is
            # watching needs to know it stopped and why.
            return self.state(stopped_reason=self._stopped_reason())
        actions = self.controller.act(self.sim)
        outcome = self.sim.step(actions)
        self.ticks += 1
        return self.state(events=outcome.events())

    def advance(self, ticks: int = 1) -> dict[str, Any]:
        """Advance several ticks and return only the final frame."""
        if ticks < 1:
            raise FleetSessionError("ticks must be at least 1")
        if ticks > MAX_TICKS_PER_CALL:
            raise FleetSessionError(
                f"cannot advance {ticks} ticks in one call (limit "
                f"{MAX_TICKS_PER_CALL}); call repeatedly or use the stream"
            )
        frame = self.state()
        for _ in range(ticks):
            frame = self.tick()
            if self.sim.done:
                break
        return frame

    def _stopped_reason(self) -> str | None:
        if not self.sim.done:
            return None
        counters = self.sim.fleet_counters()
        if counters["robots_alive"] == 0:
            return "every robot ran out of charge"
        return "the fleet deadlocked with work outstanding"

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def state(
        self,
        events: list[tuple[int, str]] | None = None,
        stopped_reason: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.sim.snapshot(events)
        snapshot.update(
            {
                "ticks": self.ticks,
                "scenario": self.config.name,
                "floor": self.floor,
                "controller": self.controller.name,
                "n_robots": self.sim.n_robots,
                "fleet": self.sim.fleet_counters(),
                "stopped": self.sim.done,
                "stopped_reason": stopped_reason or self._stopped_reason(),
                # Stated in the payload itself so a live figure is never
                # mistaken for a measured one from data/results/.
                "provenance": (
                    "live interactive session - not a measured result; reported "
                    "numbers come from the offline pipeline on fixed seeds"
                ),
            }
        )
        return snapshot

    def layout(self) -> dict[str, Any]:
        """The static map plus the zones, for drawing the floor once."""
        payload = self.sim.layout.to_dict()
        payload["zones"] = [z.to_dict(include_cells=True) for z in self.sim.zones]
        # Convenient for a UI that offers "pick a storage point" rather than
        # making someone type coordinates.
        # The realistic floor has 306 pick points; a dropdown of all of them is
        # useless, so the UI gets a readable sample plus the full count.
        storage = self.sim.layout.storage_points
        step = max(1, len(storage) // 24)
        payload["pickup_choices"] = [list(p) for p in storage[::step]][:24]
        payload["pickup_total"] = len(storage)
        payload["dropoff_choices"] = [
            list(p) for p in self.sim.layout.packing_stations
        ]
        payload["floor"] = self.floor
        if self.directions is not None:
            # Sent once with the layout, never per frame: 1,380 integers is
            # fine alongside the grid and wasteful ten times a second.
            payload["directions"] = self.directions.astype(int).tolist()
            payload["direction_summary"] = direction_summary(self.directions)
        return payload


#: The process-wide session. One warehouse, because the dashboard shows one
#: floor and a second would quietly compete for the same CPU.
_session: FleetSession | None = None


def current() -> FleetSession:
    if _session is None:
        raise FleetSessionError(
            "no warehouse is running; start one with POST /api/fleet/start"
        )
    return _session


def start(
    config: ScenarioConfig,
    n_robots: int = 3,
    controller: str = "astar",
    layout: WarehouseLayout | None = None,
    seed: int = 0,
    floor: str = "default",
) -> FleetSession:
    """Start a warehouse, replacing any already running."""
    global _session
    _session = FleetSession(
        config,
        n_robots=n_robots,
        controller=controller,
        layout=layout,
        seed=seed,
        floor=floor,
    )
    return _session


def stop() -> None:
    global _session
    _session = None


def is_running() -> bool:
    return _session is not None
