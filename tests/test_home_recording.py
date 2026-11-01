"""The recorded run on the website's home page, which needs no backend.

The page presents this file as a real simulated run, so these pin that it still
is one: a whole run from an empty floor to the last delivery, and battery ranges
that match the rules the page names. Regenerate it with
``python scripts/run_fleet_studies.py home_floor``.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.services.simulation_service import load_scenario_config

RECORDING = Path(__file__).resolve().parents[1] / "data" / "fleet" / "home_floor.json"


def load() -> dict:
    return json.loads(RECORDING.read_text(encoding="utf-8"))


def test_the_recording_is_a_whole_run_that_delivers_everything():
    recording = load()
    frames = recording["frames"]
    assert frames[0]["delivered"] == 0
    assert frames[-1]["delivered"] == recording["products"]
    assert all(robot[4] == 1 for robot in frames[-1]["robots"]), "no robot may end flat"

    delivered = [frame["delivered"] for frame in frames]
    assert delivered == sorted(delivered)


def test_the_quoted_battery_ranges_match_the_rules():
    ranges = load()["moves_on_full_charge"]
    assert ranges
    for rules, moves in ranges.items():
        battery = load_scenario_config(rules).battery
        assert moves == round(battery.capacity / battery.move_cost)
