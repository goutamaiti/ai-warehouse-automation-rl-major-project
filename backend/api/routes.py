"""HTTP routes of the warehouse dashboard API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from backend.services import simulation_service as service
from simulation.warehouse import MAX_CUSTOM_SIZE

router = APIRouter(prefix="/api")


class RunRequest(BaseModel):
    """Body of ``POST /api/run``."""

    scenario: str = Field(default="default", description="base scenario: supplies every rule that is not the layout (battery, reward, obstacles, step budget)")
    controller: str = Field(default="astar", description="astar | bfs | random | ppo")
    seed: int = Field(default=0, ge=0, le=2**31 - 1, description="episode seed")
    layout: list[list[int]] | None = Field(
        default=None,
        description=(
            "optional user-drawn grid of CellType integers (0 empty, 1 wall, "
            "2 shelf, 3 storage, 4 packing, 5 charging) that overrides the "
            "scenario's own layout; the scenario's other rules still apply"
        ),
    )

    @field_validator("layout")
    @classmethod
    def _bounded_grid(cls, value):
        if value is None:
            return value
        if len(value) > MAX_CUSTOM_SIZE or any(len(row) > MAX_CUSTOM_SIZE for row in value):
            raise ValueError(f"layout must be at most {MAX_CUSTOM_SIZE}x{MAX_CUSTOM_SIZE}")
        return value


def _handle(call, *args, **kwargs):
    """Run a service call, translating service errors into HTTP 404/400."""
    try:
        return call(*args, **kwargs)
    except service.ServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe, also used by the frontend to detect a running backend."""
    return {"status": "ok", "controllers": service.available_controllers()}


@router.get("/scenarios")
def scenarios() -> list[dict[str, Any]]:
    return service.list_scenarios()


@router.get("/scenarios/{name}/layout")
def layout(name: str) -> dict[str, Any]:
    return _handle(service.get_layout, name)


@router.get("/episodes")
def episodes() -> list[dict[str, Any]]:
    return service.list_recorded_episodes()


@router.get("/episodes/{episode_id}")
def episode(episode_id: str) -> dict[str, Any]:
    return _handle(service.load_recorded_episode, episode_id)


@router.get("/results")
def results() -> list[dict[str, Any]]:
    return service.list_results()


@router.post("/run")
def run(request: RunRequest) -> dict[str, Any]:
    """Run one episode live and return the full replay.

    When ``layout`` is given, the episode runs on that user-drawn grid instead
    of the named scenario's procedurally generated map; every other rule
    (battery, reward weights, obstacles, step budget) still comes from
    ``scenario``.
    """
    return _handle(
        service.run_live_episode,
        request.scenario,
        request.controller,
        request.seed,
        request.layout,
    )


@router.get("/run")
def run_via_get(
    scenario: str = Query(default="default"),
    controller: str = Query(default="astar"),
    seed: int = Query(default=0, ge=0, le=2**31 - 1),
) -> dict[str, Any]:
    """Convenience GET variant, handy for quick checks from the browser."""
    return _handle(service.run_live_episode, scenario, controller, seed)
