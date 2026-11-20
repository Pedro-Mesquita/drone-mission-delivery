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
    Everything outside the geofence is an obstacle. `blocked` is the obstacle
    map inflated by the drone radius plus the safety margin, so the planner can
    treat the drone as a point.
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

    def add_observation(self, polygon: Sequence[GroundPoint]) -> bool:
        """Records one sighting of an obstacle. Returns True if new obstacle cells were confirmed."""
        before = self._confirmed()
        self._hits[self._rasterize(polygon)] += 1
        added = bool(np.any(self._confirmed() & ~before))
        if added:
            self._blocked = None
        return added

    @property
    def obstacle(self) -> NDArray[np.bool_]:
        """Confirmed obstacle cells and cells outside the geofence, not inflated."""
        return self._confirmed() | self._outside

    @property
    def blocked(self) -> NDArray[np.bool_]:
        """Cells the drone's center must not enter."""
        if self._blocked is None:
            source = self.obstacle.astype(np.uint8)
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

    def is_free(self, point: GroundPoint) -> bool:
        cell = self.cell_of(point)
        return cell is not None and not self.blocked[cell]

    def segment_is_free(self, start: GroundPoint, end: GroundPoint) -> bool:
        """Whether the straight flight from `start` to `end` stays out of blocked cells."""
        length = math.hypot(end.north - start.north, end.east - start.east)
        steps = max(1, math.ceil(length / (self.cell_size / 4)))
        for step in range(steps + 1):
            t = step / steps
            point = GroundPoint(
                north=start.north + t * (end.north - start.north),
                east=start.east + t * (end.east - start.east),
            )
            if not self.is_free(point):
                return False
        return True

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
