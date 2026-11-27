"""One stop of the mission: photos, detection, and the map update."""

from __future__ import annotations

import logging
import math
from typing import Protocol

import cv2
import numpy as np
from drone_mission_core import Drone
from drone_vision_core import ObbDetection
from numpy.typing import NDArray

from mission_delivery.config import MissionConfig
from mission_delivery.occupancy import OccupancyGrid
from mission_delivery.projection import Camera, GroundPoint, Pose, polygon_to_ground

_logger = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, image: NDArray[np.uint8]) -> ObbDetection: ...


class PhotoError(RuntimeError):
    """No usable photo after every attempt."""


def current_pose(drone: Drone) -> Pose:
    """Pose of a drone that is stopped, so level (the UAV API gives no attitude)."""
    position = drone.get_ned_position()
    heading = drone.get_heading()
    return Pose(
        north=position.north,
        east=position.east,
        height=position.altitude,
        yaw=math.radians(heading),
    )


def observe(
    drone: Drone,
    detector: Detector,
    camera: Camera,
    grid: OccupancyGrid,
    config: MissionConfig,
) -> int:
    """Takes the photos of one stop and records what they show in `grid`.

    Every obstacle box is an observation; the ground the photo covers becomes
    seen once all photos succeeded. Returns the number of boxes found.
    Raises `PhotoError` when a photo fails `config.photo_attempts` times.
    """
    boxes = 0
    first_pose: Pose | None = None
    for index in range(config.photos_per_stop):
        image, detection = _photo(drone, detector, config, index)
        # Read right after each photo: a drone still drifting would put every
        # photo's boxes at a different place if one pose served the whole stop.
        pose = current_pose(drone)
        first_pose = first_pose or pose
        _logger.info("photo %d at north=%.2f east=%.2f heading=%.0f: %d boxes", index + 1,
                     pose.north, pose.east, math.degrees(pose.yaw), len(detection.boxes))
        scale_x = camera.config.width / image.shape[1]
        scale_y = camera.config.height / image.shape[0]
        detected = []
        for box in detection.boxes:
            corners = [(x * scale_x, y * scale_y) for x, y in box.corners]
            polygon = polygon_to_ground(corners, camera, pose)
            if polygon is not None:
                grid.add_observation(polygon)
                detected.append(polygon)
                boxes += 1
        clear_area = _inner_area(camera, pose, config.clear_border)
        if clear_area is not None:
            grid.clear_unconfirmed(clear_area, detected)
    assert first_pose is not None  # photos_per_stop >= 1
    pose = first_pose
    width, height = camera.config.width, camera.config.height
    footprint = polygon_to_ground([(0, 0), (width, 0), (width, height), (0, height)], camera, pose)
    if footprint is not None:
        grid.add_coverage(footprint)
    _logger.info(
        "stop at north=%.2f east=%.2f heading=%.0f: %d boxes in %d photos",
        pose.north,
        pose.east,
        math.degrees(pose.yaw),
        boxes,
        config.photos_per_stop,
    )
    return boxes


def _inner_area(camera: Camera, pose: Pose, border: float) -> list[GroundPoint] | None:
    """Ground shown by the photo, minus `border` meters from each edge."""
    margin = border * camera.focal / pose.height
    width, height = camera.config.width, camera.config.height
    if 2 * margin >= min(width, height):
        return None
    left, top, right, bottom = margin, margin, width - margin, height - margin
    area = polygon_to_ground([(left, top), (right, top), (right, bottom), (left, bottom)],
                             camera, pose)
    return None if area is None else list(area)


def _photo(
    drone: Drone, detector: Detector, config: MissionConfig, index: int
) -> tuple[NDArray[np.uint8], ObbDetection]:
    for attempt in range(1, config.photo_attempts + 1):
        try:
            content = drone.take_photo(
                command=config.camera_command, resolution=config.camera_resolution
            )
            image = _decode(content)
            return image, detector.detect(image)
        except Exception as error:
            _logger.warning(
                "photo %d, attempt %d/%d failed: %s: %s",
                index + 1,
                attempt,
                config.photo_attempts,
                type(error).__name__,
                error,
            )
    raise PhotoError(f"photo {index + 1} failed {config.photo_attempts} times")


def _decode(content: bytes) -> NDArray[np.uint8]:
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("the camera returned bytes that are not an image")
    return np.asarray(decoded, dtype=np.uint8)
