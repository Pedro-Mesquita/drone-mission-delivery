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
from mission_delivery.projection import Camera, Pose, polygon_to_ground

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
    pose = current_pose(drone)
    boxes = 0
    for index in range(config.photos_per_stop):
        image, detection = _photo(drone, detector, config, index)
        scale_x = camera.config.width / image.shape[1]
        scale_y = camera.config.height / image.shape[0]
        for box in detection.boxes:
            corners = [(x * scale_x, y * scale_y) for x, y in box.corners]
            polygon = polygon_to_ground(corners, camera, pose)
            if polygon is not None:
                grid.add_observation(polygon)
                boxes += 1
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
