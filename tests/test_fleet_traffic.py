"""Traffic rules of the fleet controller, on the one-way 46x30 floor.

Each rule here exists because robots were measured dying from it, not from a
battery that was too small. Over 200-product runs, 31 of 33 dead robots had stood
still for at least 40 ticks. The numbers are in docs/experiments.md.
"""

from __future__ import annotations

import random

import pytest

from baselines.fleet_controller import STALL_TICKS, FleetPlannerController
from simulation.config import find_scenario, load_scenario
from simulation.dispatch import TaskQueue
from simulation.engine import Action
from simulation.fleet import STALL_WARNING_TICKS, FleetSimulation
from simulation.rmfs import build_rmfs_layout

#: Open cells of the two-way corridor above the chargers.
LEFT_CELL, RIGHT_CELL = (27, 13), (27, 14)
#: One cell below (27, 12), and the next charger along the row.
NEAR_CHARGER, NEXT_CHARGER = (28, 12), (28, 17)
#: A two-way junction on a one-way lane, and the aisle cell below it.
LANE_EXIT, EXIT_BELOW = (21, 17), (22, 17)
A_TASK = object()


def floor(n_robots: int = 2, seed: int = 0, **controller_args):
    layout, direction = build_rmfs_layout()
    queue = TaskQueue()
    sim = FleetSimulation(
        load_scenario(find_scenario("default")),
        n_robots=n_robots,
        layout=layout,
        task_queue=queue,
        directions=direction,
    )
    sim.reset(seed=seed)
    controller = FleetPlannerController(directions=direction, **controller_args)
    controller.reset(sim)
    return sim, queue, controller


def place(sim, *cells):
    for robot_id, cell in enumerate(cells):
        sim.members[robot_id].robot.position = cell


# ----------------------------------------------------------------------
# Boxed-in robots
# ----------------------------------------------------------------------


def test_a_boxed_in_robot_attempts_the_move_rather_than_waiting():
    """Waiting hid the refusal, so nothing ever learned the robot needed a cell."""
    sim, _, controller = floor(battery_aware=False)
    place(sim, LEFT_CELL, RIGHT_CELL)
    sim.target_for = lambda robot_id: RIGHT_CELL  # occupied by robot 1

    assert controller._act_one(sim, 0) == int(Action.RIGHT)


def test_a_working_robot_steps_aside_when_head_on():
    sim, _, controller = floor()
    place(sim, LEFT_CELL, RIGHT_CELL)
    sim.members[0].task = sim.members[1].task = A_TASK

    actions = controller._make_way(sim, {0: int(Action.RIGHT), 1: int(Action.LEFT)})

    assert actions[0] == int(Action.RIGHT)
    assert actions[1] not in (int(Action.LEFT), int(Action.WAIT))


def test_a_working_robot_keeps_its_place_in_ordinary_traffic():
    sim, _, controller = floor()
    place(sim, LEFT_CELL, RIGHT_CELL)
    sim.members[1].task = A_TASK

    actions = controller._make_way(sim, {0: int(Action.RIGHT), 1: int(Action.WAIT)})

    assert actions[1] == int(Action.WAIT)


def test_a_jammed_working_robot_makes_way():
    sim, _, controller = floor()
    place(sim, LEFT_CELL, RIGHT_CELL)
    sim.members[1].task = A_TASK
    controller._stalled[1] = STALL_TICKS

    actions = controller._make_way(sim, {0: int(Action.RIGHT), 1: int(Action.WAIT)})

    assert actions[1] != int(Action.WAIT)


# ----------------------------------------------------------------------
# A passing robot on the exit
# ----------------------------------------------------------------------


def on_a_lane_with_a_peer_on_the_exit(**controller_args):
    sim, _, controller = floor(battery_aware=False, **controller_args)
    place(sim, LANE_EXIT, EXIT_BELOW)
    sim.target_for = lambda robot_id: (28, 17)
    return sim, controller


def test_a_robot_waits_for_a_passing_peer_rather_than_missing_its_exit():
    sim, controller = on_a_lane_with_a_peer_on_the_exit()

    assert controller._act_one(sim, 0) == int(Action.DOWN)


def test_a_robot_gives_up_waiting_for_a_peer_that_does_not_move():
    sim, controller = on_a_lane_with_a_peer_on_the_exit()
    controller._stalled[0] = STALL_TICKS

    assert controller._act_one(sim, 0) == int(Action.RIGHT)


def test_a_robot_does_not_wait_for_a_flat_robot():
    sim, controller = on_a_lane_with_a_peer_on_the_exit()
    sim.members[1].alive = False

    assert controller._act_one(sim, 0) == int(Action.RIGHT)


# ----------------------------------------------------------------------
# Robots dodging in step
# ----------------------------------------------------------------------


def test_the_higher_numbered_robot_holds_instead_of_reversing():
    sim, _, controller = floor()
    place(sim, LEFT_CELL, RIGHT_CELL)
    controller._last_position[1] = LEFT_CELL
    actions = {1: int(Action.LEFT)}

    controller._hold_reversals(sim, actions)

    assert actions[1] == int(Action.WAIT)


def test_a_reversal_with_nobody_near_is_left_alone():
    sim, _, controller = floor()
    place(sim, (5, 5), RIGHT_CELL)
    controller._last_position[1] = LEFT_CELL
    actions = {1: int(Action.LEFT)}

    controller._hold_reversals(sim, actions)

    assert actions[1] == int(Action.LEFT)


# ----------------------------------------------------------------------
# Choosing a charger
# ----------------------------------------------------------------------


def low_robots(*cells, battery=60.0):
    sim, _, controller = floor()
    place(sim, *cells)
    for member in sim.members.values():
        member.robot.battery = battery
    return sim, controller


def test_two_robots_do_not_queue_for_one_charger_when_another_is_free():
    sim, controller = low_robots((27, 12), LEFT_CELL)

    assert controller._goal_for(sim, 0) == NEAR_CHARGER
    assert controller._goal_for(sim, 1) == NEXT_CHARGER


def test_a_robot_keeps_the_charger_it_claimed():
    sim, controller = low_robots((27, 12), LEFT_CELL)
    controller._goal_for(sim, 1)
    assert controller._goal_for(sim, 1) == controller._claimed[1]


def test_a_flat_robot_on_a_charger_is_routed_around():
    sim, controller = low_robots(LEFT_CELL, NEAR_CHARGER)
    sim.members[1].alive = False

    assert controller._goal_for(sim, 0) == NEXT_CHARGER


def test_a_nearly_flat_robot_does_not_set_off_for_a_far_charger():
    sim, controller = low_robots((27, 12), LEFT_CELL, battery=3.0)
    controller._claimed[1] = NEAR_CHARGER

    assert controller._charger_for(sim, 0)[0] == NEAR_CHARGER


def test_a_claim_ends_when_the_robot_stops_needing_the_charger():
    sim, controller = low_robots((27, 12), LEFT_CELL)
    controller._goal_for(sim, 0)
    sim.members[0].robot.battery = 100.0

    controller._goal_for(sim, 0)

    assert 0 not in controller._claimed


# ----------------------------------------------------------------------
# The stuck-robot warning
# ----------------------------------------------------------------------


def test_a_robot_held_still_with_work_is_reported_as_stalled():
    """The deadlock detector cannot see this: the other robot keeps moving."""
    sim, queue, _ = floor()
    layout = sim.layout
    queue.add(layout.storage_points[0], layout.packing_stations[0], step=0)
    sim.step({})
    worker = next(r for r, m in sim.members.items() if m.task is not None)
    other = 1 - worker
    # Off the chargers: a robot gaining charge is working, not stuck.
    sim.members[worker].robot.position = (27, 20)
    sim.members[other].robot.position = LEFT_CELL
    sim.step({other: int(Action.RIGHT)})  # the distance from its new cell

    for tick in range(STALL_WARNING_TICKS):
        assert sim.fleet_counters()["stalled_robots"] == []
        sim.step({other: int(Action.LEFT if tick % 2 == 0 else Action.RIGHT)})

    assert sim.fleet_counters()["stalled_robots"] == [worker]


def test_an_idle_robot_is_never_reported_as_stalled():
    sim, _, _ = floor()
    other = 1
    sim.members[other].robot.position = LEFT_CELL
    for tick in range(STALL_WARNING_TICKS + 5):
        sim.step({other: int(Action.RIGHT if tick % 2 == 0 else Action.LEFT)})

    assert sim.fleet_counters()["stalled_robots"] == []


# ----------------------------------------------------------------------
# The runs that exposed the jams
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_robots, seed", [(5, 2), (20, 1)])
def test_the_runs_that_jammed_now_keep_moving(n_robots, seed):
    """Before these rules both runs had robots stuck within 400 ticks for good."""
    sim, queue, controller = floor(n_robots=n_robots, seed=seed)
    rng = random.Random(seed)
    for _ in range(200):
        queue.add(
            rng.choice(sim.layout.storage_points),
            rng.choice(sim.layout.packing_stations),
            step=0,
        )

    for _ in range(500):
        sim.step(controller.act(sim))
        counters = sim.fleet_counters()
        assert counters["stalled_robots"] == []

    assert not sim.done
    assert counters["robots_alive"] == n_robots
