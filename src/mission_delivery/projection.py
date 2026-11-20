"""Pixel to ground projection for the downward camera, assuming flat ground."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mission_delivery.config import CameraConfig

_MIN_DOWNWARD = 1e-9


@dataclass(frozen=True, slots=True)
class GroundPoint:
    """A point on the ground, in meters from the NED origin (HOME)."""

    north: float
    east: float


@dataclass(frozen=True, slots=True)
class Pose:
    """Drone pose when the frame was captured.

    `height` is above the ground, not MSL. `yaw` is the heading in radians
    (0 = north, positive towards east). `roll` and `pitch` follow the
    aerospace convention (positive = right wing down / nose up); leave them at
    zero when the camera is on a gimbal.
    """

    north: float
    east: float
    height: float
    yaw: float
    roll: float = 0.0
    pitch: float = 0.0


class Camera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.focal = config.width / (2.0 * math.tan(config.hfov / 2.0))
        self.cx = config.width / 2.0
        self.cy = config.height / 2.0

    def ray_body(self, u: float, v: float) -> NDArray[np.float64]:
        """Ray through pixel (u, v) in the body FRD frame (not normalized)."""
        right = (u - self.cx) / self.focal
        back = (v - self.cy) / self.focal
        return np.array([-back, right, 1.0])


def body_to_ned(pose: Pose) -> NDArray[np.float64]:
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return np.asarray(rz @ ry @ rx, dtype=np.float64)


def pixel_to_ground(u: float, v: float, camera: Camera, pose: Pose) -> GroundPoint | None:
    """Ground point seen at pixel (u, v), or None if the ray does not hit the ground."""
    rotation = body_to_ned(pose)
    ray = rotation @ camera.ray_body(u, v)
    if ray[2] <= _MIN_DOWNWARD:
        return None
    offset = rotation @ np.array([camera.config.offset_forward, camera.config.offset_right, 0.0])
    lens_down = offset[2] - pose.height
    scale = -lens_down / ray[2]
    if scale <= 0:
        return None
    return GroundPoint(
        north=float(pose.north + offset[0] + scale * ray[0]),
        east=float(pose.east + offset[1] + scale * ray[1]),
    )


def polygon_to_ground(
    corners: Iterable[Iterable[float]], camera: Camera, pose: Pose
) -> tuple[GroundPoint, ...] | None:
    """Projects image corners (x, y in pixels) to the ground.

    Accepts any iterable of (x, y) pairs, including `drone_vision_core.Point`.
    Returns None if any corner does not hit the ground.
    """
    points: list[GroundPoint] = []
    for corner in corners:
        u, v = tuple(corner)
        point = pixel_to_ground(u, v, camera, pose)
        if point is None:
            return None
        points.append(point)
    return tuple(points)
