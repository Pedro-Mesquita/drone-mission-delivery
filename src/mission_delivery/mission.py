"""Delivery mission: fly to a NED target avoiding cartolinas, drop the object, come back.

Photo by photo: the drone stops at the mission altitude, takes several
photos, marks the cartolinas it sees on the map, plans with A* and flies
only over ground already seen; then stops again. At the target it opens
the gripper and returns to HOME the same way, starting from an empty map,
where it lands. With no free path, or no usable photo, it lands where it is.

Run: python -m mission_delivery.mission --target NORTH EAST [ALTITUDE]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from drone_mission_core import DEFAULT_BASE_URL, Drone, NedPosition, StateMachine

from mission_delivery.config import CameraConfig, MissionConfig, PlannerConfig
from mission_delivery.occupancy import OccupancyGrid
from mission_delivery.perception import Detector, PhotoError, observe
from mission_delivery.planner import plan_path, seen_prefix
from mission_delivery.projection import Camera, GroundPoint

_logger = logging.getLogger(__name__)

HOME = GroundPoint(north=0.0, east=0.0)


class ArrivalTimeoutError(RuntimeError):
    """The drone did not reach a waypoint in time."""


@dataclass
class Mission:
    target: GroundPoint
    detector: Detector
    config: MissionConfig = field(default_factory=MissionConfig)
    camera: Camera = field(default_factory=lambda: Camera(CameraConfig()))
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    returning: bool = False
    segment: list[GroundPoint] = field(default_factory=list)
    stops: int = 0
    landing_reason: str = ""

    def __post_init__(self) -> None:
        self.grid = self.empty_grid()
        if not self.grid.is_free(self.target, inflated=False):
            raise ValueError(f"target {self.target} is outside the track")
        # The planner never ends a path inside the inflated margin of the track edge:
        # such a target, or HOME for the way back, would land the drone at the first plan.
        for name, point in (("target", self.target), ("HOME", HOME)):
            if not self.grid.is_free(point):
                raise ValueError(
                    f"{name} {point} is too close to the track edge: keep it "
                    f"{self.grid.inflation:.2f} m or more inside"
                )

    def empty_grid(self) -> OccupancyGrid:
        track = [GroundPoint(north, east) for north, east in self.config.track]
        return OccupancyGrid(track, self.planner)

    @property
    def goal(self) -> GroundPoint:
        return HOME if self.returning else self.target


def takeoff(mission: Mission, drone: Drone) -> str:
    drone.arm()
    drone.takeoff(altitude=round(mission.config.altitude))
    return "OBSERVE"


def observe_stop(mission: Mission, drone: Drone) -> str:
    _wait_until_stopped(drone, mission.config)
    try:
        observe(drone, mission.detector, mission.camera, mission.grid, mission.config)
    except PhotoError as error:
        mission.landing_reason = str(error)
        return "LAND"
    mission.stops += 1
    return "PLAN"


def plan(mission: Mission, drone: Drone) -> str:
    here = _ground(drone.get_ned_position())
    if _distance(here, mission.goal) <= mission.config.waypoint_tolerance:
        return "ARRIVED"
    path = plan_path(mission.grid, here, mission.goal)
    if path is None:
        mission.landing_reason = f"no free path to {mission.goal}"
        return "LAND"
    segment = seen_prefix(mission.grid, here, path)
    if not segment or _length(here, segment) < mission.config.min_progress:
        mission.landing_reason = "no seen ground to advance over"
        return "LAND"
    mission.segment = segment
    return "FLY"


def fly(mission: Mission, drone: Drone) -> str:
    for waypoint in mission.segment:
        _fly_to(drone, waypoint, mission.config)
    return "OBSERVE"


def arrived(mission: Mission, drone: Drone) -> str:
    if mission.returning:
        mission.landing_reason = "back at HOME"
        return "LAND"
    drone.set_servo(mission.config.gripper_channel, mission.config.gripper_open_pwm)
    time.sleep(mission.config.gripper_open_wait)
    # The way back is the same mission towards HOME, starting from an empty
    # map: nothing seen on the way out counts, so it is photographed again.
    mission.returning = True
    mission.grid = mission.empty_grid()
    return "OBSERVE"


def land(mission: Mission, drone: Drone) -> str:
    _logger.info("landing: %s", mission.landing_reason)
    drone.land()
    return "DONE"


def run(drone: Drone, mission: Mission) -> None:
    """Flies `mission`. On an unexpected error it tries to land, then re-raises."""
    machine: StateMachine[Mission] = StateMachine(mission, drone)
    for name, function in (
        ("TAKEOFF", takeoff),
        ("OBSERVE", observe_stop),
        ("PLAN", plan),
        ("FLY", fly),
        ("ARRIVED", arrived),
        ("LAND", land),
    ):
        machine.create_state(name, function)
    machine.set_initial_state("TAKEOFF")
    try:
        machine.run_state_machine()
    except BaseException:
        _logger.exception("mission failed in state %s; landing", machine.current_state)
        try:
            drone.land()
        except Exception:
            _logger.exception("landing after the failure also failed")
        raise


def _wait_until_stopped(drone: Drone, config: MissionConfig) -> None:
    """Returns once the ground speed stayed at or below `settle_speed` for
    `settle_hold` seconds in a row, or after `settle_timeout`."""
    deadline = time.monotonic() + config.settle_timeout
    still_since: float | None = None
    while True:
        velocity = drone.get_ned_telemetry().velocity
        now = time.monotonic()
        if math.hypot(velocity.north, velocity.east) <= config.settle_speed:
            still_since = now if still_since is None else still_since
            if now - still_since >= config.settle_hold:
                return
        else:
            still_since = None
        if now >= deadline:
            _logger.warning("still moving after %.0f s; taking the photos anyway",
                            config.settle_timeout)
            return
        time.sleep(config.poll_interval)


def _fly_to(drone: Drone, waypoint: GroundPoint, config: MissionConfig) -> None:
    target = NedPosition(north=waypoint.north, east=waypoint.east, altitude=config.altitude)
    drone.go_to_ned(target)
    deadline = time.monotonic() + config.arrival_timeout
    # Ground distance only: altitude does not decide arrival. The mission core
    # writes the real position to the flight log on arrival.
    while not drone.has_arrived_ned(target, config.waypoint_tolerance, horizontal=True):
        if time.monotonic() >= deadline:
            raise ArrivalTimeoutError(f"did not reach {waypoint} in {config.arrival_timeout:.0f} s")
        time.sleep(config.poll_interval)


def _ground(position: NedPosition) -> GroundPoint:
    return GroundPoint(north=position.north, east=position.east)


def _distance(a: GroundPoint, b: GroundPoint) -> float:
    return math.hypot(a.north - b.north, a.east - b.east)


def _length(start: GroundPoint, waypoints: Sequence[GroundPoint]) -> float:
    total, previous = 0.0, start
    for waypoint in waypoints:
        total += _distance(previous, waypoint)
        previous = waypoint
    return total


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--target",
        nargs="+",
        type=float,
        required=True,
        metavar="METERS",
        help="NORTH EAST [ALTITUDE] from HOME; the altitude is ignored",
    )
    parser.add_argument(
        "--track",
        nargs="+",
        type=float,
        metavar="METERS",
        help="geofence as NORTH EAST pairs from HOME, 3 or more corners "
        "(default: MissionConfig.track)",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="UAV API address")
    parser.add_argument("--model", default=MissionConfig().model_path, help="YOLO-OBB weights")
    args = parser.parse_args(argv)
    if len(args.target) not in (2, 3):
        parser.error("--target takes NORTH EAST [ALTITUDE]")
    config = MissionConfig(model_path=args.model)
    if args.track is not None:
        if len(args.track) < 6 or len(args.track) % 2:
            parser.error("--track takes 3 or more NORTH EAST pairs")
        corners = tuple(zip(args.track[::2], args.track[1::2], strict=True))
        config = dataclasses.replace(config, track=corners)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from drone_vision_core import ObbDetector

    mission = Mission(
        target=GroundPoint(north=args.target[0], east=args.target[1]),
        detector=ObbDetector(args.model),
        config=config,
    )
    with Drone(args.base_url) as drone:
        run(drone, mission)


if __name__ == "__main__":
    main()
