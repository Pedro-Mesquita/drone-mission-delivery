"""Mission parameters.

Values marked provisional are placeholders until the real hardware is known
(drone size, camera mounting). Change them here, not in the modules that use them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Downward-facing pinhole camera.

    Mounting: optical axis along the drone's body down axis, image top towards
    the drone's nose and image right towards its right side. `offset_forward`
    and `offset_right` place the lens relative to the drone's center, in meters.
    """

    # Raspberry Pi Camera Module 3 (IMX708, standard lens), captured by /take_photo
    # (rpicam-still) at its default resolution. Check on the drone that this resolution
    # keeps the full field of view.
    width: int = 1280
    height: int = 720
    hfov: float = math.radians(66.0)
    offset_forward: float = -0.12  # measured: lens 12 cm behind the center
    offset_right: float = 0.015  # measured: lens 1.5 cm right of the center


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Occupancy grid and path planner."""

    cell_size: float = 0.1  # meters
    # Measured: half the 56 cm propeller tip-to-tip diagonal; the gripper stays inside it.
    drone_radius: float = 0.28
    safety_margin: float = 0.5  # provisional: covers GPS drift and projection error
    # Observations of a cell before it counts as an obstacle: 2 of the
    # `MissionConfig.photos_per_stop` photos of a stop.
    confirm_hits: int = 2

    @property
    def inflation_radius(self) -> float:
        return self.drone_radius + self.safety_margin


@dataclass(frozen=True, slots=True)
class MissionConfig:
    """Flight rules of the photo-by-photo mission.

    The drone stops, takes `photos_per_stop` photos, updates the map, plans
    with A* and flies only over ground already seen; then stops again.
    """

    altitude: float = 4.0  # meters above HOME, for flight and photos
    photos_per_stop: int = 3
    photo_attempts: int = 5  # tries per photo before landing
    # Track (geofence) in meters from the takeoff point. Provisional: 30 x 10 m.
    track: tuple[tuple[float, float], ...] = ((-2.0, -5.0), (32.0, -5.0), (32.0, 5.0), (-2.0, 5.0))
    waypoint_tolerance: float = 0.3  # horizontal meters
    arrival_timeout: float = 60.0  # seconds to reach a waypoint before landing
    # The drone counts as stopped once its ground speed stays at or below
    # `settle_speed` for `settle_hold` seconds in a row: right after arriving it is
    # still braking, tilted and drifting to the waypoint, which moves the photos.
    settle_speed: float = 0.05  # m/s
    settle_hold: float = 1.0  # seconds
    settle_timeout: float = 10.0  # seconds before taking the photos anyway
    # Meters on the ground, from each edge of the photo, where a missing detection
    # does not clear a pending cell: a cartolina cut by the edge is often missed.
    clear_border: float = 0.7
    min_progress: float = 0.05  # meters: a shorter flight means the drone is stuck
    poll_interval: float = 0.2  # seconds
    gripper_channel: int = 9
    gripper_open_pwm: int = 1400
    gripper_open_wait: float = 1.0  # seconds after opening, before leaving
    camera_command: str = "rpicam-still"
    camera_resolution: str = "1280x720"
    model_path: str = "models/ground-obstacle-v2.onnx"
