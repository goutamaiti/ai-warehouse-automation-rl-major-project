"""Tests for the continuously running warehouse and its HTTP surface.

The behaviour these pin is the one the dashboard's product-location card
depends on: a floor that is already running when a product is added, and the
nearest free robot moving on the next tick rather than at some later scheduling
pass.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import fleet_session as service


@pytest.fixture(autouse=True)
def _no_leftover_warehouse():
    """Each test starts with no floor running, and leaves none behind."""
    service.stop()
    yield
    service.stop()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def start(client: TestClient, **kwargs) -> dict:
    body = {"scenario": "default", "n_robots": 3, "seed": 1000, **kwargs}
    response = client.post("/api/fleet/start", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_nothing_is_running_until_a_warehouse_is_started(client):
    assert client.get("/api/fleet/health").json()["running"] is False
    # Asking for state before starting is a client error, not a 500.
    assert client.get("/api/fleet/state").status_code == 400
    assert client.post("/api/fleet/task", json={"pickup": [2, 3], "dropoff": [13, 4]}).status_code == 400


def test_starting_returns_the_layout_and_the_first_frame(client):
    payload = start(client)
    assert client.get("/api/fleet/health").json()["running"] is True
    assert [z["name"] for z in payload["layout"]["zones"]] == ["A", "B"]
    assert payload["layout"]["pickup_choices"], "a UI needs somewhere to pick from"
    assert payload["layout"]["dropoff_choices"]
    assert len(payload["state"]["robots"]) == 3
    assert payload["state"]["ticks"] == 0


def test_stopping_releases_the_warehouse(client):
    start(client)
    assert client.post("/api/fleet/stop").json()["running"] is False
    assert client.get("/api/fleet/state").status_code == 400


def test_starting_again_replaces_the_running_warehouse(client):
    start(client, n_robots=3)
    assert len(start(client, n_robots=5)["state"]["robots"]) == 5


def test_an_unplaceable_fleet_is_refused_with_a_usable_message(client):
    response = client.post(
        "/api/fleet/start", json={"scenario": "default", "n_robots": 10_000}
    )
    assert response.status_code == 400
    assert "cannot place" in response.json()["detail"]


def test_battery_rules_are_refused_on_the_realistic_floor(client):
    # Measured: on the 46x30 floor these rules run every robot flat (5 robots,
    # 2 of 5 products delivered by tick 133), so the combination is refused.
    for scenario in ("battery_tuned", "battery_constrained"):
        response = client.post(
            "/api/fleet/start", json={"scenario": scenario, "n_robots": 5, "floor": "rmfs"}
        )
        assert response.status_code == 400
        assert "21x15" in response.json()["detail"]
    # ...while the same rules on the floor they were sized for still start.
    start(client, scenario="battery_tuned")


def test_an_unknown_scenario_is_a_404(client):
    assert client.post("/api/fleet/start", json={"scenario": "nope"}).status_code == 404


def test_an_unknown_controller_is_refused(client):
    response = client.post(
        "/api/fleet/start", json={"scenario": "default", "controller": "psychic"}
    )
    assert response.status_code == 400


# ----------------------------------------------------------------------
# The product-location card
# ----------------------------------------------------------------------


def test_a_product_is_assigned_to_the_nearest_robot_on_the_next_tick(client):
    """The headline behaviour: add a location, a robot starts moving."""
    payload = start(client)
    pickup = payload["layout"]["pickup_choices"][0]
    dropoff = payload["layout"]["dropoff_choices"][0]

    created = client.post(
        "/api/fleet/task", json={"pickup": pickup, "dropoff": dropoff}
    ).json()
    assert created["request"]["status"] == "waiting"
    assert created["request"]["robot_id"] is None

    # Distances must be read BEFORE the tick: dispatch happens at the top of
    # the tick, so by the time it returns the chosen robot has already moved a
    # cell closer and the comparison would be off by one.
    session = service.current()
    before = {
        rid: session.sim.path_distance(m.position, tuple(pickup))
        for rid, m in session.sim.members.items()
    }

    frame = client.post("/api/fleet/advance", json={"ticks": 1}).json()

    assert len(frame["dispatch"]) == 1
    decision = frame["dispatch"][0]
    assigned = [r for r in frame["robots"] if r["id"] == decision["robot_id"]][0]
    assert assigned["task"] is not None
    assert assigned["task"]["pickup"] == pickup

    # And it really was the nearest at the moment the decision was made.
    assert decision["distance"] == min(d for d in before.values() if d >= 0), before
    assert before[decision["robot_id"]] == decision["distance"]


def test_a_product_is_delivered_and_the_request_records_its_turnaround(client):
    payload = start(client)
    pickup = payload["layout"]["pickup_choices"][0]
    dropoff = payload["layout"]["dropoff_choices"][0]
    client.post("/api/fleet/task", json={"pickup": pickup, "dropoff": dropoff})

    frame = None
    for _ in range(6):
        frame = client.post("/api/fleet/advance", json={"ticks": 10}).json()
        if frame["queue"]["delivered"]:
            break

    assert frame is not None and frame["queue"]["delivered"] == 1
    request = frame["requests"][0]
    assert request["status"] == "delivered"
    assert request["turnaround"] is not None and request["turnaround"] > 0
    assert request["robot_id"] is not None


def test_several_products_go_to_different_robots(client):
    payload = start(client, n_robots=3)
    dropoff = payload["layout"]["dropoff_choices"][0]
    for pickup in payload["layout"]["pickup_choices"][:3]:
        client.post("/api/fleet/task", json={"pickup": pickup, "dropoff": dropoff})

    frame = client.post("/api/fleet/advance", json={"ticks": 1}).json()
    assigned = {d["robot_id"] for d in frame["dispatch"]}
    assert len(assigned) == 3, "each product should have gone to its own robot"


# ----------------------------------------------------------------------
# Validation - a person types these coordinates, so the errors must be usable
# ----------------------------------------------------------------------


def test_a_product_on_a_shelf_is_refused_by_name(client):
    start(client)
    session = service.current()
    from simulation.warehouse import CellType

    shelf = next(
        (int(r), int(c))
        for r, c in zip(*(session.sim.layout.grid == int(CellType.SHELF)).nonzero())
    )
    response = client.post(
        "/api/fleet/task",
        json={"pickup": list(shelf), "dropoff": session.sim.layout.packing_stations[0]},
    )
    assert response.status_code == 400
    assert "shelf" in response.json()["detail"]


def test_a_product_outside_the_warehouse_is_refused(client):
    start(client)
    response = client.post(
        "/api/fleet/task", json={"pickup": [999, 999], "dropoff": [13, 4]}
    )
    assert response.status_code == 400
    assert "outside" in response.json()["detail"]


def test_pickup_and_dropoff_cannot_be_the_same_cell(client):
    payload = start(client)
    pickup = payload["layout"]["pickup_choices"][0]
    response = client.post("/api/fleet/task", json={"pickup": pickup, "dropoff": pickup})
    assert response.status_code == 400
    assert "same cell" in response.json()["detail"]


def test_a_malformed_coordinate_is_a_client_error(client):
    start(client)
    assert (
        client.post("/api/fleet/task", json={"pickup": [1], "dropoff": [13, 4]}).status_code
        == 400
    )


def test_advancing_too_far_in_one_call_is_refused(client):
    start(client)
    response = client.post("/api/fleet/advance", json={"ticks": 10_000})
    assert response.status_code == 400
    assert "limit" in response.json()["detail"]


def test_the_pending_request_cap_is_enforced(client):
    payload = start(client, n_robots=1)
    pickup = payload["layout"]["pickup_choices"][0]
    dropoff = payload["layout"]["dropoff_choices"][0]
    session = service.current()
    for _ in range(service.MAX_PENDING_REQUESTS):
        session.queue.add(tuple(pickup), tuple(dropoff), step=0)

    response = client.post("/api/fleet/task", json={"pickup": pickup, "dropoff": dropoff})
    assert response.status_code == 400
    assert "already waiting" in response.json()["detail"]


# ----------------------------------------------------------------------
# Honesty of the payload
# ----------------------------------------------------------------------


def test_every_frame_says_it_is_a_demonstration_not_a_measurement(client):
    """A live figure must never be mistaken for one from data/results/."""
    payload = start(client)
    assert "not a measured result" in payload["state"]["provenance"]
    assert "not a measurement" in client.get("/api/fleet/health").json()["note"]


def test_an_idle_floor_reports_idle_rather_than_deadlocked(client):
    """Regression: a caught-up warehouse used to look jammed."""
    start(client)
    frame = client.post("/api/fleet/advance", json={"ticks": 20}).json()
    assert frame["idle"] is True
    assert frame["deadlocked"] is False
    assert frame["stopped"] is False


# ----------------------------------------------------------------------
# The stream
# ----------------------------------------------------------------------


def test_the_stream_sends_one_frame_per_tick(client):
    start(client)
    frames = []
    with client.stream(
        "GET", "/api/fleet/stream?tick_seconds=0.02&max_frames=3"
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
            if len(frames) >= 4:
                break

    assert len(frames) >= 2
    # The first frame is the current state; the rest advance one tick each.
    ticks = [f["ticks"] for f in frames if "ticks" in f]
    assert ticks == sorted(ticks)
    assert ticks[-1] > ticks[0]


def test_the_stream_refuses_an_absurd_tick_rate(client):
    start(client)
    assert client.get("/api/fleet/stream?tick_seconds=0").status_code == 400
    assert client.get("/api/fleet/stream?tick_seconds=99").status_code == 400


def test_streaming_without_a_warehouse_is_a_client_error(client):
    assert client.get("/api/fleet/stream").status_code == 400


def test_a_product_can_be_queued_as_urgent_and_bad_priorities_are_refused():
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    client.post("/api/fleet/start", json={"scenario": "default", "n_robots": 3, "seed": 0})
    ok = client.post("/api/fleet/task", json={"pickup": [2, 3], "dropoff": [13, 4], "priority": "urgent"})
    assert ok.status_code == 200
    assert ok.json()["request"]["priority"] == "urgent"
    default = client.post("/api/fleet/task", json={"pickup": [2, 7], "dropoff": [13, 4]})
    assert default.json()["request"]["priority"] == "normal"
    bad = client.post("/api/fleet/task", json={"pickup": [2, 11], "dropoff": [13, 4], "priority": "high"})
    assert bad.status_code == 422
    client.post("/api/fleet/stop")
