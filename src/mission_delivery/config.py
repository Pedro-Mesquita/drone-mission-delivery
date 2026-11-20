"""Mission parameters.

Values marked provisional are placeholders until the real hardware is known
(camera model, drone size). Change them here, not in the modules that use them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Downward-facing pinhole camera.

    Mounting: optical axis along the drone's body down axis, image top towards
    the drone's nose and image right towards its right side. `offset_forward`
    and `offset_right` place the lens relative to the drone's center, in meters.
    """

    width: int = 640  # provisional: same as the synthetic dataset
    height: int = 640  # provisional
    hfov: float = 1.2  # provisional, radians
    offset_forward: float = 0.0  # provisional
    offset_right: float = 0.0  # provisional


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Occupancy grid and path planner."""

    cell_size: float = 0.1  # meters
    drone_radius: float = 0.4  # provisional: half the tip-to-tip width with the gripper
    safety_margin: float = 0.5  # provisional: covers GPS drift and projection error
    confirm_hits: int = 3  # observations of a cell before it counts as an obstacle

    @property
    def inflation_radius(self) -> float:
        return self.drone_radius + self.safety_margin
