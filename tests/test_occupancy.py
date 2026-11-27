from __future__ import annotations

import pytest

from mission_delivery.config import PlannerConfig
from mission_delivery.occupancy import OccupancyGrid
from mission_delivery.projection import GroundPoint as P

CONFIG = PlannerConfig(cell_size=0.1, drone_radius=0.4, safety_margin=0.5, confirm_hits=3)
FENCE = [P(0, -5), P(0, 5), P(30, 5), P(30, -5)]
SHEET = [P(10, -0.5), P(10, 0.5), P(11, 0.5), P(11, -0.5)]


def grid_with(*polygons: list[P], hits: int = 3) -> OccupancyGrid:
    grid = OccupancyGrid(FENCE, CONFIG)
    for polygon in polygons:
        for _ in range(hits):
            grid.add_observation(polygon)
    return grid


def test_empty_track_is_free_inside_the_fence() -> None:
    grid = grid_with()
    assert grid.is_free(P(15, 0))
    assert grid.segment_is_free(P(2, 0), P(28, 0))


def test_outside_the_fence_is_blocked() -> None:
    grid = grid_with()
    assert not grid.is_free(P(15, 6))
    assert not grid.is_free(P(-1, 0))
    assert not grid.is_free(P(100, 0))


def test_fence_is_inflated() -> None:
    grid = grid_with()
    assert not grid.is_free(P(15, 4.5))
    assert grid.is_free(P(15, 3.8))


def test_obstacle_needs_confirmation() -> None:
    grid = OccupancyGrid(FENCE, CONFIG)
    assert not grid.add_observation(SHEET)
    assert not grid.add_observation(SHEET)
    assert grid.pending.any()
    assert not grid.obstacle[grid.pending].any()  # pending, not yet an obstacle
    assert grid.add_observation(SHEET)
    assert not grid.pending.any()
    assert not grid.is_free(P(10.5, 0))
    assert not grid.add_observation(SHEET)


def test_obstacle_is_inflated_by_radius_and_margin() -> None:
    grid = grid_with(SHEET)
    reach = 0.5 + CONFIG.inflation_radius
    assert not grid.is_free(P(10.5, reach - 0.05))
    assert grid.is_free(P(10.5, reach + 0.3))
    assert not grid.is_free(P(10 - CONFIG.inflation_radius + 0.05, 0))


def test_raw_obstacle_is_not_inflated() -> None:
    grid = grid_with(SHEET)
    cell = grid.cell_of(P(10.5, 0.8))
    assert cell is not None
    assert not grid.obstacle[cell]
    assert grid.blocked[cell]


def test_segment_over_an_obstacle_is_not_free() -> None:
    grid = grid_with(SHEET)
    assert not grid.segment_is_free(P(5, 0), P(15, 0))
    assert grid.segment_is_free(P(5, 3), P(15, 3))


def test_cell_center_round_trip() -> None:
    grid = grid_with()
    cell = grid.cell_of(P(12.34, -1.23))
    assert cell is not None
    center = grid.center_of(cell)
    assert center.north == pytest.approx(12.34, abs=0.05)
    assert center.east == pytest.approx(-1.23, abs=0.05)
    assert grid.cell_of(center) == cell


def test_geofence_needs_three_points() -> None:
    with pytest.raises(ValueError):
        OccupancyGrid([P(0, 0), P(1, 1)], CONFIG)


class TestPending:
    """A sighted, unconfirmed cell blocks flying near it until it is resolved."""

    TRACK = (P(-5, -5), P(10, -5), P(10, 5), P(-5, 5))

    def grid(self) -> OccupancyGrid:
        grid = OccupancyGrid(self.TRACK, PlannerConfig())
        grid.add_coverage(self.TRACK)
        return grid

    def test_one_sighting_blocks_the_planner(self) -> None:
        grid = self.grid()
        assert grid.is_free(P(3.5, 0.0))

        grid.add_observation([P(4, -0.2), P(4.5, -0.2), P(4.5, 0.2), P(4, 0.2)])

        assert grid.pending.any()
        assert not grid.is_free(P(3.5, 0.0))  # within the margin of the pending cells
        assert grid.is_free(P(4.2, 0.0), inflated=False)  # still not an obstacle
        assert grid.is_seen(P(3.5, 0.0))

    def test_a_clear_photo_frees_it(self) -> None:
        grid = self.grid()
        grid.add_observation([P(4, -0.2), P(4.5, -0.2), P(4.5, 0.2), P(4, 0.2)])

        grid.clear_unconfirmed([P(2, -2), P(6, -2), P(6, 2), P(2, 2)], [])

        assert not grid.pending.any()
        assert grid.is_free(P(3.5, 0.0))

    def test_a_photo_that_shows_it_again_does_not_free_it(self) -> None:
        grid = self.grid()
        box = [P(4, -0.2), P(4.5, -0.2), P(4.5, 0.2), P(4, 0.2)]
        grid.add_observation(box)

        grid.clear_unconfirmed([P(2, -2), P(6, -2), P(6, 2), P(2, 2)], [box])

        assert grid.pending.any()

    def test_a_confirmed_obstacle_is_never_cleared(self) -> None:
        grid = self.grid()
        box = [P(4, -0.2), P(4.5, -0.2), P(4.5, 0.2), P(4, 0.2)]
        grid.add_observation(box)
        grid.add_observation(box)

        grid.clear_unconfirmed([P(2, -2), P(6, -2), P(6, 2), P(2, 2)], [])

        assert not grid.is_free(P(4.25, 0.0), inflated=False)
