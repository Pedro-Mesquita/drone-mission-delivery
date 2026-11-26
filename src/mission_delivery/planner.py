"""A* path planning over the occupancy grid.

Unseen cells count as free: the first plan is the straight line to the goal,
and the mission replans when the camera confirms an obstacle on the path.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from mission_delivery.occupancy import Cell, OccupancyGrid
from mission_delivery.projection import GroundPoint

_SQRT2 = math.sqrt(2)
_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
)


def plan_path(
    grid: OccupancyGrid, start: GroundPoint, goal: GroundPoint
) -> list[GroundPoint] | None:
    """Waypoints from `start` (excluded) to `goal` (included), or None if no path exists.

    If `start` is inside the inflated margin of an obstacle, the path first
    leaves it through the nearest free cell, without crossing the obstacle itself.
    """
    start_cell = grid.cell_of(start)
    goal_cell = grid.cell_of(goal)
    if start_cell is None or goal_cell is None or not grid.is_free(goal):
        return None
    waypoints: list[GroundPoint] = []
    origin = start
    if grid.blocked[start_cell]:
        exit_path = _escape(grid, start_cell)
        if exit_path is None:
            return None
        exit_points = [grid.center_of(cell) for cell in exit_path[1:]]
        waypoints = _simplify(grid, start, exit_points, inflated=False)
        start_cell = exit_path[-1]
        origin = exit_points[-1]
    cells = _astar(grid.blocked, start_cell, goal_cell)
    if cells is None:
        return None
    points = [grid.center_of(cell) for cell in cells[1:-1]] + [goal]
    return waypoints + _simplify(grid, origin, points, inflated=True)


def path_is_free(grid: OccupancyGrid, start: GroundPoint, waypoints: list[GroundPoint]) -> bool:
    """Whether flying from `start` through `waypoints` stays out of blocked cells."""
    previous = start
    for waypoint in waypoints:
        if not grid.segment_is_free(previous, waypoint):
            return False
        previous = waypoint
    return True


def _astar(blocked: NDArray[np.bool_], start: Cell, goal: Cell) -> list[Cell] | None:
    rows, cols = blocked.shape
    cost: dict[Cell, float] = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    queue: list[tuple[float, float, Cell]] = [(_octile(start, goal), 0.0, start)]
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current == goal:
            return _backtrack(parent, goal)
        if current_cost > cost[current]:
            continue
        row, col = current
        for d_row, d_col, step in _NEIGHBORS:
            nr, nc = row + d_row, col + d_col
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                continue
            if d_row and d_col and (blocked[row, nc] or blocked[nr, col]):
                continue
            neighbor = (nr, nc)
            new_cost = current_cost + step
            if new_cost < cost.get(neighbor, math.inf):
                cost[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(queue, (new_cost + _octile(neighbor, goal), new_cost, neighbor))
    return None


def _octile(a: Cell, b: Cell) -> float:
    d_row, d_col = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(d_row, d_col) + (_SQRT2 - 1) * min(d_row, d_col)


def _backtrack(parent: dict[Cell, Cell], goal: Cell) -> list[Cell]:
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path[::-1]


def _escape(grid: OccupancyGrid, start: Cell) -> list[Cell] | None:
    blocked, obstacle = grid.blocked, grid.obstacle
    parent: dict[Cell, Cell] = {}
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if not blocked[current]:
            return _backtrack(parent, current)
        row, col = current
        for d_row, d_col, _ in _NEIGHBORS:
            neighbor = (row + d_row, col + d_col)
            inside = 0 <= neighbor[0] < grid.rows and 0 <= neighbor[1] < grid.cols
            if neighbor in seen or not inside or obstacle[neighbor]:
                continue
            seen.add(neighbor)
            parent[neighbor] = current
            queue.append(neighbor)
    return None


def _simplify(
    grid: OccupancyGrid, origin: GroundPoint, points: list[GroundPoint], *, inflated: bool
) -> list[GroundPoint]:
    """Keeps only the turns: from each kept point, jumps to the farthest point in straight sight."""
    kept: list[GroundPoint] = []
    current = origin
    index = 0
    while index < len(points):
        farthest = index
        for candidate in range(len(points) - 1, index, -1):
            if grid.segment_is_free(current, points[candidate], inflated=inflated):
                farthest = candidate
                break
        current = points[farthest]
        kept.append(current)
        index = farthest + 1
    return kept
