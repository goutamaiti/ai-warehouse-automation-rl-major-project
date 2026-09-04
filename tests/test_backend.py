"""Tests for the FastAPI layer, driven through the ASGI test client."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app

client = TestClient(create_app())


def test_health_lists_controllers():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    names = {controller["name"] for controller in payload["controllers"]}
    assert {"astar", "bfs", "random", "ppo"} <= names


def test_scenarios_include_the_shipped_configs():
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    names = {scenario["name"] for scenario in response.json()}
    assert {"default", "dynamic_obstacles", "battery_constrained"} <= names


def test_layout_returns_a_grid_with_stations():
    payload = client.get("/api/scenarios/default/layout").json()
    assert len(payload["grid"]) == payload["height"]
    assert len(payload["grid"][0]) == payload["width"]
    assert payload["packing_stations"] and payload["charging_stations"]


def test_unknown_scenario_returns_404():
    assert client.get("/api/scenarios/nope/layout").status_code == 404


def test_run_returns_a_replay_that_ends_in_a_delivery():
    response = client.post("/api/run", json={"scenario": "default", "controller": "astar", "seed": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["tasks_delivered"] == 1
    assert len(payload["frames"]) == payload["summary"]["steps"] + 1
    assert payload["frames"][-1]["task"]["status"] == "delivered"


def test_run_rejects_an_unknown_controller():
    response = client.post("/api/run", json={"scenario": "default", "controller": "magic", "seed": 0})
    assert response.status_code == 404


def test_run_rejects_an_out_of_range_seed():
    response = client.post("/api/run", json={"scenario": "default", "controller": "astar", "seed": -1})
    assert response.status_code == 422


def test_episode_id_cannot_escape_the_episodes_directory():
    response = client.get("/api/episodes/..%2F..%2Fpyproject")
    assert response.status_code in (404, 400)


@pytest.mark.parametrize("path", ["/api/episodes", "/api/results"])
def test_listing_endpoints_return_lists(path: str):
    response = client.get(path)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_ppo_model_lookup_prefers_the_scenario_specific_policy():
    from backend.services import simulation_service as service

    path = service.ppo_model_path("dynamic_obstacles")
    if path is None:
        pytest.skip("no trained models present in this checkout")
    assert path.name in {"ppo_dynamic.zip", "ppo_default.zip"}


def _tiny_custom_grid():
    # 0 empty, 1 wall, 2 shelf, 3 storage, 4 packing, 5 charging
    return [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 2, 3, 0, 0, 1],
        [1, 4, 0, 0, 0, 5, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]


def test_run_accepts_a_custom_layout():
    response = client.post(
        "/api/run",
        json={"scenario": "default", "controller": "astar", "seed": 0, "layout": _tiny_custom_grid()},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["layout"]["width"] == 7 and payload["layout"]["height"] == 5
    assert payload["summary"]["tasks_delivered"] == 1


def test_run_rejects_an_invalid_custom_layout():
    grid = _tiny_custom_grid()
    grid[2][3] = 0  # remove the only storage point
    response = client.post(
        "/api/run",
        json={"scenario": "default", "controller": "astar", "seed": 0, "layout": grid},
    )
    assert response.status_code == 404
    assert "invalid layout" in response.json()["detail"]


def test_run_rejects_an_oversized_custom_layout():
    huge_row = [0] * 60
    response = client.post(
        "/api/run",
        json={"scenario": "default", "controller": "astar", "seed": 0, "layout": [huge_row] * 60},
    )
    assert response.status_code == 422


def test_run_replay_carries_reward_breakdown_per_frame():
    response = client.post("/api/run", json={"scenario": "default", "controller": "astar", "seed": 5})
    payload = response.json()
    assert payload["reward_config"]["delivery_reward"] == 20.0
    assert "reward" not in payload["frames"][0]  # initial state, nothing happened yet
    assert payload["frames"][1]["reward"] is not None
    assert "reward_components" in payload["frames"][1]
