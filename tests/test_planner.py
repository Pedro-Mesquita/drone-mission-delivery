from __future__ import annotations

import time
from itertools import pairwise

import pytest

from mission_delivery.config import PlannerConfig
from mission_delivery.occupancy import OccupancyGrid
from mission_delivery.planner import path_is_free, plan_path
from mission_delivery.projection import GroundPoint as P

CONFIG = PlannerConfig(cell_size=0.1, drone_radius=0.4, safety_margin=0.5, confirm_hits=1)
FENCE = [P(0, -5), P(0, 5), P(30, 5), P(30, -5)]
START = P(1, 0)
GOAL = P(29, 0)


def box(north: float, east: float, north_to: float, east_to: float) -> list[P]:
    return [P(north, east), P(north, east_to), P(north_to, east_to), P(north_to, east)]


RIGHT = box(8, 0, 9, 5)
LEFT = box(15, -5, 16, 0)
GAP_WIDTH = 2.6
GAP_LEFT = box(22, -5, 23, -GAP_WIDTH / 2)
GAP_RIGHT = box(22, GAP_WIDTH / 2, 23, 5)


def grid_with(*polygons: list[P]) -> OccupancyGrid:
    grid = OccupancyGrid(FENCE, CONFIG)
    for polygon in polygons:
        grid.add_observation(polygon)
    return grid


def east_at(path: list[P], north: float) -> float:
    for a, b in pairwise(path):
        if min(a.north, b.north) <= north <= max(a.north, b.north) and a.north != b.north:
            t = (north - a.north) / (b.north - a.north)
            return a.east + t * (b.east - a.east)
    raise AssertionError(f"path never crosses north={north}")


def planned(grid: OccupancyGrid, start: P = START, goal: P = GOAL) -> list[P]:
    path = plan_path(grid, start, goal)
    assert path is not None
    assert path[-1] == goal
    return path


def test_free_track_is_a_straight_line() -> None:
    assert planned(grid_with()) == [GOAL]


def test_obstacle_on_the_right_is_passed_on_the_left() -> None:
    grid = grid_with(RIGHT)
    path = planned(grid)
    assert path_is_free(grid, START, path)
    assert east_at([START, *path], 8.5) < 0


def test_obstacle_on_the_left_is_passed_on_the_right() -> None:
    grid = grid_with(LEFT)
    path = planned(grid)
    assert path_is_free(grid, START, path)
    assert east_at([START, *path], 15.5) > 0


def test_reference_track_goes_through_every_gap() -> None:
    grid = grid_with(RIGHT, LEFT, GAP_LEFT, GAP_RIGHT)
    path = planned(grid)
    route = [START, *path]
    assert path_is_free(grid, START, path)
    assert east_at(route, 8.5) < 0
    assert east_at(route, 15.5) > 0
    assert abs(east_at(route, 22.5)) < GAP_WIDTH / 2
    assert len(path) <= 8


def test_obstacles_learned_one_at_a_time_lead_to_the_same_track() -> None:
    grid = grid_with()
    position = START
    for obstacle in (RIGHT, LEFT, GAP_LEFT, GAP_RIGHT):
        grid.add_observation(obstacle)
        path = planned(grid, position)
        assert path_is_free(grid, position, path)
        position = path[0]
    assert path_is_free(grid, position, planned(grid, position))


def test_gap_narrower_than_the_drone_has_no_path() -> None:
    narrow = 2 * CONFIG.inflation_radius - 0.2
    grid = grid_with(box(22, -5, 23, -narrow / 2), box(22, narrow / 2, 23, 5))
    assert plan_path(grid, START, GOAL) is None


def test_goal_on_an_obstacle_has_no_path() -> None:
    assert plan_path(grid_with(box(28, -1, 30, 1)), START, GOAL) is None


def test_goal_outside_the_fence_has_no_path() -> None:
    assert plan_path(grid_with(), START, P(40, 0)) is None


def test_start_inside_the_margin_escapes_without_crossing_the_obstacle() -> None:
    obstacle = box(5, -1, 6, 1)
    grid = grid_with(obstacle)
    start = P(4.6, 0)
    assert not grid.is_free(start)
    path = planned(grid, start)
    assert grid.segment_is_free(start, path[0], inflated=False)
    first_free = next(index for index, point in enumerate(path) if grid.is_free(point))
    assert path_is_free(grid, path[first_free], path[first_free + 1 :])


def test_start_on_the_obstacle_itself_has_no_path() -> None:
    grid = grid_with(box(5, -1, 6, 1))
    assert plan_path(grid, P(5.5, 0), GOAL) is None


@pytest.mark.parametrize("cell_size", [0.1, 0.2])
def test_planning_is_fast_enough_to_replan_in_hover(cell_size: float) -> None:
    config = PlannerConfig(cell_size=cell_size, confirm_hits=1)
    grid = OccupancyGrid(FENCE, config)
    for obstacle in (RIGHT, LEFT, GAP_LEFT, GAP_RIGHT):
        grid.add_observation(obstacle)
    began = time.perf_counter()
    assert plan_path(grid, START, GOAL) is not None
    assert time.perf_counter() - began < 1.0
