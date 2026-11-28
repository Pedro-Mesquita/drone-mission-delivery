from __future__ import annotations

import math

import numpy as np
import pytest

from mission_delivery.config import CameraConfig
from mission_delivery.projection import (
    Camera,
    GroundPoint,
    Pose,
    body_to_ned,
    pixel_to_ground,
    polygon_to_ground,
)

CONFIG = CameraConfig(width=640, height=640, hfov=1.2, offset_forward=0.0, offset_right=0.0)
CAMERA = Camera(CONFIG)
HALF = math.tan(CONFIG.hfov / 2.0)


def ground(u: float, v: float, pose: Pose, camera: Camera = CAMERA) -> GroundPoint:
    point = pixel_to_ground(u, v, camera, pose)
    assert point is not None
    return point


def assert_close(point: GroundPoint, north: float, east: float) -> None:
    assert point.north == pytest.approx(north, abs=1e-9)
    assert point.east == pytest.approx(east, abs=1e-9)


def test_image_center_is_below_the_drone() -> None:
    assert_close(ground(320, 320, Pose(north=5.0, east=-2.0, height=3.0, yaw=0.7)), 5.0, -2.0)


def test_top_of_image_is_ahead_of_the_nose() -> None:
    assert_close(ground(320, 0, Pose(0.0, 0.0, 3.0, yaw=0.0)), 3.0 * HALF, 0.0)


def test_right_of_image_is_right_of_the_drone() -> None:
    assert_close(ground(640, 320, Pose(0.0, 0.0, 3.0, yaw=0.0)), 0.0, 3.0 * HALF)


def test_yaw_rotates_the_footprint() -> None:
    pose = Pose(0.0, 0.0, 3.0, yaw=math.pi / 2)
    assert_close(ground(320, 0, pose), 0.0, 3.0 * HALF)
    assert_close(ground(640, 320, pose), -3.0 * HALF, 0.0)


def test_footprint_scales_with_height() -> None:
    low = ground(0, 0, Pose(0.0, 0.0, 3.0, yaw=0.0))
    high = ground(0, 0, Pose(0.0, 0.0, 6.0, yaw=0.0))
    assert high.north == pytest.approx(2 * low.north)
    assert high.east == pytest.approx(2 * low.east)


def test_nose_up_pitch_moves_the_center_forward() -> None:
    pose = Pose(0.0, 0.0, 3.0, yaw=0.0, pitch=math.radians(10))
    assert_close(ground(320, 320, pose), 3.0 * math.tan(math.radians(10)), 0.0)


def test_right_roll_moves_the_center_left() -> None:
    pose = Pose(0.0, 0.0, 3.0, yaw=0.0, roll=math.radians(10))
    assert_close(ground(320, 320, pose), 0.0, -3.0 * math.tan(math.radians(10)))


def test_camera_offset_is_rotated_with_the_drone() -> None:
    config = CameraConfig(width=640, height=640, hfov=1.2, offset_forward=0.2, offset_right=0.1)
    camera = Camera(config)
    assert_close(ground(320, 320, Pose(0.0, 0.0, 3.0, yaw=math.pi / 2), camera), -0.1, 0.2)


def test_ray_above_the_horizon_does_not_hit_the_ground() -> None:
    pose = Pose(0.0, 0.0, 3.0, yaw=0.0, pitch=math.radians(80))
    assert pixel_to_ground(320, 0, CAMERA, pose) is None


def test_projection_inverts_a_pinhole_camera() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        pose = Pose(
            north=float(rng.uniform(-20, 20)),
            east=float(rng.uniform(-20, 20)),
            height=float(rng.uniform(2, 15)),
            yaw=float(rng.uniform(-math.pi, math.pi)),
            roll=float(rng.uniform(-0.25, 0.25)),
            pitch=float(rng.uniform(-0.25, 0.25)),
        )
        u, v = float(rng.uniform(0, 640)), float(rng.uniform(0, 640))
        point = ground(u, v, pose)
        relative = np.array([point.north - pose.north, point.east - pose.east, pose.height])
        forward, right, down = body_to_ned(pose).T @ relative
        assert 320 + CAMERA.focal * right / down == pytest.approx(u)
        assert 320 - CAMERA.focal * forward / down == pytest.approx(v)


def test_polygon_projects_every_corner() -> None:
    corners = [(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)]
    polygon = polygon_to_ground(corners, CAMERA, Pose(0.0, 0.0, 3.0, yaw=0.0))
    assert polygon is not None
    reach = 3.0 * HALF
    expected = [(reach, -reach), (reach, reach), (-reach, reach), (-reach, -reach)]
    for point, (north, east) in zip(polygon, expected, strict=True):
        assert_close(point, north, east)


def test_polygon_accepts_vision_core_points() -> None:
    from drone_vision_core import Point

    polygon = polygon_to_ground([Point(320, 320)] * 4, CAMERA, Pose(1.0, 2.0, 3.0, yaw=0.0))
    assert polygon is not None
    assert_close(polygon[0], 1.0, 2.0)


def test_polygon_is_none_when_a_corner_misses_the_ground() -> None:
    pose = Pose(0.0, 0.0, 3.0, yaw=0.0, pitch=math.radians(80))
    assert polygon_to_ground([(320, 320), (320, 0)], CAMERA, pose) is None
