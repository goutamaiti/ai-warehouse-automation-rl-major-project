"""HTTP routes for the continuously running warehouse.

These sit apart from ``backend/api/routes.py`` on purpose. Those routes are
read-mostly and serve the *measured* project: scenarios, recorded replays and
the committed result tables. These ones drive a live floor whose numbers depend
on how long a page was left open and when a person added a product, so mixing
them into the same router would blur the line between a measurement and a demo.
Every payload here carries a ``provenance`` field saying which it is.
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services import fleet_session as service
from backend.services.simulation_service import ServiceError, load_scenario_config
from simulation.warehouse import MAX_CUSTOM_SIZE, layout_from_grid

router = APIRouter(prefix="/api/fleet")

#: Seconds between frames on the stream. 0.1 is fast enough to read as motion
#: and slow enough that a browser keeps up with the JSON.
DEFAULT_TICK_SECONDS = 0.1
MIN_TICK_SECONDS = 0.02
MAX_TICK_SECONDS = 2.0

#: Frames one stream will send before closing. A browser tab left open
#: overnight should not hold a simulation loop for ever; the client reconnects.
MAX_STREAM_FRAMES = 5_000


class StartRequest(BaseModel):
    """Body of ``POST /api/fleet/start``."""

    scenario: str = Field(
        default="default",
        description="base scenario: supplies the battery model, obstacles and step rules",
    )
    n_robots: int = Field(
        default=3,
        ge=1,
        description=(
            "fleet size. Not capped here - an unplaceable fleet is rejected by "
            "the simulation, which reports how many the floor can hold"
        ),
    )
    controller: str = Field(default="astar", description="astar | bfs")
    floor: str = Field(
        default="default",
        description=(
            "'default' = the 21x15 scenario floor; 'rmfs' = the realistic "
            "robotic-fulfilment floor with a pod field and one-way aisles"
        ),
    )
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    layout: list[list[int]] | None = Field(
        default=None,
        description="optional user-drawn grid of CellType integers, as for /api/run",
    )


class TaskRequest(BaseModel):
    """Body of ``POST /api/fleet/task`` - one product to move."""

    pickup: list[int] = Field(description="[row, column] where the product is")
    dropoff: list[int] = Field(description="[row, column] to deliver it to")
    priority: Literal["normal", "urgent"] = Field(
        default="normal", description="urgent requests are dispatched before normal ones"
    )


class AdvanceRequest(BaseModel):
    ticks: int = Field(default=1, ge=1, description="how many ticks to run")


def _handle(call, *args, **kwargs):
    """Translate service errors into 4xx instead of a 500 traceback."""
    try:
        return call(*args, **kwargs)
    except service.FleetSessionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "running": service.is_running(),
        "tick_seconds": DEFAULT_TICK_SECONDS,
        "note": (
            "A live fleet is a demonstration, not a measurement. Reported "
            "results come from scripts/run_experiments.py on fixed seeds."
        ),
    }


@router.post("/start")
def start(request: StartRequest) -> dict[str, Any]:
    """Start a warehouse running. Replaces one already running."""
    config = _handle(load_scenario_config, request.scenario)

    custom = None
    if request.layout is not None:
        if len(request.layout) > MAX_CUSTOM_SIZE or any(
            len(row) > MAX_CUSTOM_SIZE for row in request.layout
        ):
            raise HTTPException(
                status_code=400,
                detail=f"layout must be at most {MAX_CUSTOM_SIZE}x{MAX_CUSTOM_SIZE}",
            )
        try:
            custom = layout_from_grid(request.layout)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid layout: {exc}") from exc

    session = _handle(
        service.start,
        config,
        n_robots=request.n_robots,
        controller=request.controller,
        layout=custom,
        seed=request.seed,
        floor=request.floor,
    )
    return {"layout": session.layout(), "state": session.state()}


@router.post("/stop")
def stop() -> dict[str, Any]:
    service.stop()
    return {"running": False}


@router.get("/layout")
def layout() -> dict[str, Any]:
    """The static map and zones - fetched once, then the stream carries motion."""
    return _handle(lambda: service.current().layout())


@router.get("/state")
def state() -> dict[str, Any]:
    """The current frame, without advancing time."""
    return _handle(lambda: service.current().state())


@router.post("/task")
def add_task(request: TaskRequest) -> dict[str, Any]:
    """Add a product. The nearest idle robot is assigned on the next tick.

    This is what the dashboard's product-location card calls. Dispatch runs at
    the top of the next tick, so a robot starts moving immediately rather than
    at some later scheduling pass.
    """
    session = _handle(service.current)
    created = _handle(session.add_request, request.pickup, request.dropoff, request.priority)
    return {"request": created, "queue": session.queue.stats()}


@router.post("/advance")
def advance(request: AdvanceRequest) -> dict[str, Any]:
    """Step the warehouse on, for tests and deterministic demonstrations."""
    session = _handle(service.current)
    return _handle(session.advance, request.ticks)


@router.get("/stream")
async def stream(
    tick_seconds: float = DEFAULT_TICK_SECONDS,
    max_frames: int = MAX_STREAM_FRAMES,
) -> StreamingResponse:
    """Server-sent events: one frame per tick, for as long as the client listens.

    The stream *is* the warehouse clock - it advances the simulation as it
    yields - so closing the page pauses the floor rather than leaving a loop
    running behind it. That is deliberate for a single-floor demonstration; a
    deployment with several viewers would move the clock into its own task.
    """
    if not (MIN_TICK_SECONDS <= tick_seconds <= MAX_TICK_SECONDS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"tick_seconds must be between {MIN_TICK_SECONDS} and "
                f"{MAX_TICK_SECONDS}, got {tick_seconds}"
            ),
        )
    frames = max(1, min(int(max_frames), MAX_STREAM_FRAMES))
    session = _handle(service.current)

    async def frames_iter():
        # The first frame is the current state, so a client that connects
        # mid-run draws the floor immediately instead of after one tick.
        yield f"data: {json.dumps(session.state())}\n\n"
        for _ in range(frames):
            await asyncio.sleep(tick_seconds)
            if not service.is_running():
                yield 'data: {"stopped": true, "stopped_reason": "the warehouse was stopped"}\n\n'
                return
            frame = session.tick()
            yield f"data: {json.dumps(frame)}\n\n"
            if frame.get("stopped"):
                return

    return StreamingResponse(
        frames_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Without this an intermediate proxy may buffer the whole stream and
            # the page would sit blank until the run finished.
            "X-Accel-Buffering": "no",
        },
    )
