import math
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest
from drone_mission_core import NedPosition, NedTelemetry, NedVelocity
from drone_vision_core import ObbDetection, OrientedBox, Point
from numpy.typing import NDArray

from mission_delivery.config import CameraConfig, MissionConfig, PlannerConfig
from mission_delivery.mission import HOME, Mission, run
from mission_delivery.projection import Camera, GroundPoint

CAMERA = Camera(CameraConfig())
FAST = MissionConfig(poll_interval=0.0, gripper_open_wait=0.0, settle_timeout=0.0)
TARGET = GroundPoint(north=28.0, east=0.0)
CARTOLINA = (0.5, 0.66)
_, _JPEG = cv2.imencode(".jpg", np.zeros((720, 1280, 3), dtype=np.uint8))
JPEG = _JPEG.tobytes()

Polygon = list[GroundPoint]


def cartolina(north: float, east: float, yaw_deg: float = 0.0) -> Polygon:
    half_n, half_e = CARTOLINA[1] / 2, CARTOLINA[0] / 2
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [
        GroundPoint(north + dn * c - de * s, east + dn * s + de * c)
        for dn, de in ((-half_n, -half_e), (half_n, -half_e), (half_n, half_e), (-half_n, half_e))
    ]


REFERENCE = [
    cartolina(8.0, 1.0),  # right of the straight line
    cartolina(15.0, -1.0, 30.0),  # left
    cartolina(22.0, -2.2),  # two leaving a single passage
    cartolina(22.0, 2.2),
]


@dataclass
class FakeDrone:
    heading: float = 0.0
    position: GroundPoint = HOME
    altitude: float = 0.0
    flights: list[tuple[GroundPoint, GroundPoint]] = field(default_factory=list)
    servo: list[tuple[int, int]] = field(default_factory=list)
    landed_at: GroundPoint | None = None
    photo_failures: int = 0
    photos: int = 0
    fail_go_to: bool = False

    def arm(self) -> None:
        pass

    def takeoff(self, altitude: int) -> None:
        self.altitude = float(altitude)

    def land(self) -> None:
        self.landed_at = self.position
        self.altitude = 0.0

    def set_servo(self, channel: int, pwm: int) -> None:
        self.servo.append((channel, pwm))

    def take_photo(self, *, command: str, resolution: str) -> bytes:
        self.photos += 1
        if self.photo_failures:
            self.photo_failures -= 1
            raise RuntimeError("TAKE_PHOTO_FAIL: no camera")
        return JPEG

    def go_to_ned(self, target: NedPosition) -> None:
        if self.fail_go_to:
            raise RuntimeError("link lost")
        end = GroundPoint(target.north, target.east)
        self.flights.append((self.position, end))
        self.position = end

    def get_ned_position(self) -> NedPosition:
        return NedPosition(north=self.position.north, east=self.position.east,
                           altitude=self.altitude)

    def get_ned_telemetry(self) -> NedTelemetry:
        return NedTelemetry(position=self.get_ned_position(), velocity=NedVelocity(0.0, 0.0, 0.0))

    def get_heading(self) -> float:
        return self.heading


@dataclass
class FakeDetector:
    """Sees the cartolinas of the scene that fit whole in the drone's photo."""

    drone: FakeDrone
    scene: list[Polygon]
    failures: int = 0

    def detect(self, image: NDArray[np.uint8]) -> ObbDetection:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("detector crashed")
        boxes = []
        for polygon in self.scene:
            pixels = [self._pixel(point) for point in polygon]
            if all(0 <= u < CAMERA.config.width and 0 <= v < CAMERA.config.height
                   for u, v in pixels):
                corners = tuple(Point(u, v) for u, v in pixels)
                boxes.append(OrientedBox(corners, label="cartolina", confidence=0.9))  # type: ignore[arg-type]
        return ObbDetection(boxes=tuple(boxes))

    def _pixel(self, point: GroundPoint) -> tuple[float, float]:
        drone = self.drone
        yaw = math.radians(drone.heading)
        d_north, d_east = point.north - drone.position.north, point.east - drone.position.east
        forward = d_north * math.cos(yaw) + d_east * math.sin(yaw)
        right = -d_north * math.sin(yaw) + d_east * math.cos(yaw)
        return (
            CAMERA.cx + CAMERA.focal * right / drone.altitude,
            CAMERA.cy - CAMERA.focal * forward / drone.altitude,
        )


def fly_mission(
    scene: list[Polygon],
    *,
    target: GroundPoint = TARGET,
    heading: float = 0.0,
    setup: Callable[[FakeDrone, FakeDetector], None] | None = None,
) -> tuple[FakeDrone, Mission]:
    drone = FakeDrone(heading=heading)
    detector = FakeDetector(drone, scene)
    if setup is not None:
        setup(drone, detector)
    mission = Mission(target=target, detector=detector, config=FAST)
    run(drone, mission)  # type: ignore[arg-type]
    return drone, mission


def clearance(flights: list[tuple[GroundPoint, GroundPoint]], scene: list[Polygon]) -> float:
    """Smallest distance between the flown segments and the cartolinas."""
    smallest = math.inf
    for start, end in flights:
        steps = max(1, math.ceil(math.hypot(end.north - start.north, end.east - start.east) / 0.02))
        for step in range(steps + 1):
            t = step / steps
            point = (start.north + t * (end.north - start.north),
                     start.east + t * (end.east - start.east))
            for polygon in scene:
                smallest = min(smallest, _distance_to_polygon(point, polygon))
    return smallest


def _distance_to_polygon(point: tuple[float, float], polygon: Polygon) -> float:
    contour = np.array([[p.east, p.north] for p in polygon], dtype=np.float32)
    signed = cv2.pointPolygonTest(contour, (point[1], point[0]), measureDist=True)
    return max(0.0, -signed)


def test_reference_track_is_flown_without_crossing_a_cartolina() -> None:
    drone, mission = fly_mission(REFERENCE)

    assert drone.servo == [(9, 1400)]
    assert drone.landed_at is not None
    assert math.hypot(drone.landed_at.north, drone.landed_at.east) <= FAST.waypoint_tolerance
    assert mission.returning
    assert mission.landing_reason == "back at HOME"
    assert clearance(drone.flights, REFERENCE) > PlannerConfig().drone_radius
    dropped_at = drone.flights[next(i for i, (_, end) in enumerate(drone.flights)
                                    if math.hypot(end.north - TARGET.north,
                                                  end.east - TARGET.east) <= 0.3)][1]
    assert math.hypot(dropped_at.north - TARGET.north, dropped_at.east - TARGET.east) <= 0.3


def test_the_heading_is_taken_into_account() -> None:
    drone, _ = fly_mission(REFERENCE, heading=37.0)

    assert drone.servo == [(9, 1400)]
    assert clearance(drone.flights, REFERENCE) > PlannerConfig().drone_radius


def test_the_drone_only_flies_over_ground_already_seen() -> None:
    drone, mission = fly_mission([])

    forward_reach = 4.0 * (CAMERA.config.height / 2) / CAMERA.focal
    arrival = next(i for i, (_, end) in enumerate(drone.flights)
                   if math.hypot(end.north - TARGET.north, end.east - TARGET.east) <= 0.3)
    outbound = drone.flights[: arrival + 1]
    longest = max(math.hypot(end.north - start.north, end.east - start.east)
                  for start, end in outbound)
    assert longest < forward_reach
    assert mission.stops > TARGET.north / forward_reach


def test_no_free_path_lands_where_it_is() -> None:
    wall = [cartolina(10.0, east, 90.0) for east in np.arange(-5.0, 5.01, 0.6)]

    drone, mission = fly_mission(wall)

    assert drone.servo == []
    assert mission.landing_reason.startswith("no free path")
    assert drone.landed_at is not None and drone.landed_at.north < 10.0
    assert clearance(drone.flights, wall) > PlannerConfig().drone_radius


def test_a_failed_photo_is_taken_again() -> None:
    def failing_camera(drone: FakeDrone, detector: FakeDetector) -> None:
        drone.photo_failures = 2
        detector.failures = 2  # 4 failed attempts on the first photo, then success

    drone, mission = fly_mission(REFERENCE, setup=failing_camera)

    assert drone.servo == [(9, 1400)]
    assert mission.landing_reason == "back at HOME"


def test_a_photo_failing_every_attempt_lands() -> None:
    def broken_camera(drone: FakeDrone, detector: FakeDetector) -> None:
        drone.photo_failures = 10_000

    drone, mission = fly_mission(REFERENCE, setup=broken_camera)

    assert drone.photos == FAST.photo_attempts
    assert drone.flights == []
    assert drone.landed_at == HOME
    assert mission.landing_reason == f"photo 1 failed {FAST.photo_attempts} times"


def test_an_unexpected_error_lands_and_is_raised() -> None:
    def broken_link(drone: FakeDrone, detector: FakeDetector) -> None:
        drone.fail_go_to = True

    with pytest.raises(RuntimeError, match="link lost"):
        fly_mission(REFERENCE, setup=broken_link)


def test_a_target_outside_the_track_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the track"):
        Mission(target=GroundPoint(north=10.0, east=8.0), detector=FakeDetector(FakeDrone(), []))
