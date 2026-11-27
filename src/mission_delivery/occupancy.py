"""Occupancy grid of the ground obstacles seen by the camera, bounded by the geofence."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from mission_delivery.config import PlannerConfig
from mission_delivery.projection import GroundPoint

Cell = tuple[int, int]

_FIXED_POINT_BITS = 8


class OccupancyGrid:
    """Ground grid in NED, row = north, column = east.

    A cell becomes an obstacle after `confirm_hits` observations, and stays one.
    Until then a sighted cell is pending: it blocks the planner like an
    obstacle, and a photo that shows it clearly with no detection
    (`clear_unconfirmed`) frees it.
    Everything outside the geofence is an obstacle. `blocked` is the obstacle
    map inflated by the drone radius plus the safety margin, so the planner can
    treat the drone as a point.

    The grid also keeps the ground each photo covered (`add_coverage`): the
    drone only flies where its whole inflated footprint has been seen
    (`is_seen`).
    """

    def __init__(self, geofence: Sequence[GroundPoint], config: PlannerConfig) -> None:
        if len(geofence) < 3:
            raise ValueError(f"OccupancyGrid: geofence needs 3 or more points, got {len(geofence)}")
        self.config = config
        self.cell_size = config.cell_size
        # Rasterizing the obstacle and the drone position each lose up to half a cell diagonal.
        self._inflation = config.inflation_radius + math.sqrt(2) * config.cell_size
        self._radius_cells = math.ceil(self._inflation / config.cell_size)
        padding = (self._radius_cells + 1) * config.cell_size
        self.north_min = min(point.north for point in geofence) - padding
        self.east_min = min(point.east for point in geofence) - padding
        north_max = max(point.north for point in geofence) + padding
        east_max = max(point.east for point in geofence) + padding
        self.rows = math.ceil((north_max - self.north_min) / config.cell_size)
        self.cols = math.ceil((east_max - self.east_min) / config.cell_size)
        self._outside = ~self._rasterize(geofence)
        self._hits = np.zeros((self.rows, self.cols), dtype=np.int32)
        self._blocked: NDArray[np.bool_] | None = None
        self._seen = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self._unseen_nearby: NDArray[np.bool_] | None = None

    def add_observation(self, polygon: Sequence[GroundPoint]) -> bool:
        """Records one sighting of an obstacle.

        Returns True if it confirmed new obstacle cells, which is when a replan may be needed.
        """
        before = self._confirmed()
        self._hits[self._rasterize(polygon)] += 1
        self._blocked = None  # new pending cells
        return bool(np.any(self._confirmed() & ~before))

    def clear_unconfirmed(
        self, area: Sequence[GroundPoint], detected: Sequence[Sequence[GroundPoint]]
    ) -> None:
        """Records a photo that showed `area` clearly with only `detected` in it:
        the pending cells of `area` outside `detected` go back to free.

        `area` must leave out the photo's edges, where a cut cartolina is often missed.
        """
        shown = self._rasterize(area)
        for polygon in detected:
            shown &= ~self._rasterize(polygon)
        self._hits[shown & ~self._confirmed()] = 0
        self._blocked = None

    def add_coverage(self, polygon: Sequence[GroundPoint]) -> None:
        """Records the ground covered by one photo."""
        self._seen |= self._rasterize(polygon)
        self._unseen_nearby = None

    def is_seen(self, point: GroundPoint) -> bool:
        """Whether every cell within the inflation radius of `point` has been seen."""
        if self._unseen_nearby is None:
            source = (~self._seen).astype(np.uint8)
            dilated = cv2.dilate(source, self._disk(), borderType=cv2.BORDER_REPLICATE)
            self._unseen_nearby = np.asarray(dilated, dtype=np.bool_)
        return self._is_free_in(self._unseen_nearby, point)

    @property
    def pending(self) -> NDArray[np.bool_]:
        """Cells sighted at least once but neither confirmed nor cleared."""
        return (self._hits > 0) & ~self._confirmed()

    @property
    def obstacle(self) -> NDArray[np.bool_]:
        """Confirmed obstacle cells and cells outside the geofence, not inflated."""
        return self._confirmed() | self._outside

    @property
    def blocked(self) -> NDArray[np.bool_]:
        """Cells the drone's center must not enter: obstacles and pending cells, inflated."""
        if self._blocked is None:
            source = (self.obstacle | self.pending).astype(np.uint8)
            inflated = cv2.dilate(source, self._disk(), borderType=cv2.BORDER_REPLICATE)
            self._blocked = np.asarray(inflated, dtype=np.bool_)
        return self._blocked

    def cell_of(self, point: GroundPoint) -> Cell | None:
        """Cell containing `point`, or None outside the grid."""
        row = math.floor((point.north - self.north_min) / self.cell_size)
        col = math.floor((point.east - self.east_min) / self.cell_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def center_of(self, cell: Cell) -> GroundPoint:
        row, col = cell
        return GroundPoint(
            north=self.north_min + (row + 0.5) * self.cell_size,
            east=self.east_min + (col + 0.5) * self.cell_size,
        )

    def is_free(self, point: GroundPoint, *, inflated: bool = True) -> bool:
        """Whether `point` is out of the blocked cells.

        With `inflated=False`, only the raw obstacles count.
        """
        return self._is_free_in(self.blocked if inflated else self.obstacle, point)

    def segment_is_free(
        self, start: GroundPoint, end: GroundPoint, *, inflated: bool = True
    ) -> bool:
        """Whether the straight flight from `start` to `end` stays out of blocked cells."""
        grid = self.blocked if inflated else self.obstacle
        length = math.hypot(end.north - start.north, end.east - start.east)
        steps = max(1, math.ceil(length / (self.cell_size / 4)))
        for step in range(steps + 1):
            t = step / steps
            point = GroundPoint(
                north=start.north + t * (end.north - start.north),
                east=start.east + t * (end.east - start.east),
            )
            if not self._is_free_in(grid, point):
                return False
        return True

    def _is_free_in(self, grid: NDArray[np.bool_], point: GroundPoint) -> bool:
        cell = self.cell_of(point)
        return cell is not None and not grid[cell]

    def _confirmed(self) -> NDArray[np.bool_]:
        return self._hits >= self.config.confirm_hits

    def _rasterize(self, polygon: Sequence[GroundPoint]) -> NDArray[np.bool_]:
        # Cell (row, col) is pixel (x=col, y=row) and its center sits on integer coordinates.
        scale = (1 << _FIXED_POINT_BITS) / self.cell_size
        offset = 0.5 * (1 << _FIXED_POINT_BITS)
        vertices = np.array(
            [
                [
                    round((point.east - self.east_min) * scale - offset),
                    round((point.north - self.north_min) * scale - offset),
                ]
                for point in polygon
            ],
            dtype=np.int32,
        )
        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], color=1, shift=_FIXED_POINT_BITS)
        return mask.astype(np.bool_)

    def _disk(self) -> NDArray[np.uint8]:
        radius = self._radius_cells
        offsets = np.arange(-radius, radius + 1)
        distance = np.hypot(offsets[:, None], offsets[None, :]) * self.cell_size
        return (distance <= self._inflation).astype(np.uint8)
