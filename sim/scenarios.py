"""Simulated tracks for the delivery mission: fixed scenarios and a seeded random one.

Every scenario is a straight track starting behind the takeoff point, with
the target near its far end. Cartolinas, distractor shapes (not obstacles)
and positions are in NED meters from the takeoff point.

The "theoretical minimum" width is the narrowest (in steps of `WIDTH_STEP`)
for which the mission's own planner, with the mission's `PlannerConfig`,
finds a path from HOME to the target once every cartolina is known exactly.
It leaves no room for detection error, so the scenarios that must deliver
add `SLACK` to it (a few centimeters of error already close the way at the
exact minimum). `zigzag_exact` keeps the exact minimum: it may deliver or land.
Scenarios meant to have no path are checked the same way, without slack.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mission_delivery.config import PlannerConfig
from mission_delivery.occupancy import OccupancyGrid
from mission_delivery.planner import plan_path
from mission_delivery.projection import GroundPoint

BEHIND = 1.5  # meters of track behind the takeoff point
LENGTH = 15.0  # meters from the takeoff point to the target
AHEAD = 1.5  # meters of track past the target
WIDTH_STEP = 0.1
SLACK = 0.2  # meters added to the minimum width or gap for detection error
MAX_WIDTH = 12.0

# Cartolina size range in meters (the real size is not known yet).
CARTOLINA_LONG = (0.5, 0.7)
CARTOLINA_SHORT = (0.4, 0.5)
GROUNDS = ["grass", "dirt", "asphalt", "concrete", "gravel", "tiles"]
CARTOLINA_FAMILIES = ["solid", "solid", "paper", "gradient"]
DISTRACTOR_KINDS = ["ellipse", "triangle", "ngon", "parallelogram", "trapezoid", "irregular"]


@dataclass(frozen=True)
class Cartolina:
    north: float
    east: float
    yaw: float  # degrees clockwise from north
    length: float  # along its own north axis
    width: float
    texture_seed: int
    family: str
    rgb: tuple[float, float, float] | None = None  # solid color instead of a texture

    def corners(self) -> list[GroundPoint]:
        yaw = math.radians(self.yaw)
        cos, sin = math.cos(yaw), math.sin(yaw)
        half_l, half_w = self.length / 2, self.width / 2
        points = []
        for along, across in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w),
                              (-half_l, half_w)):
            points.append(GroundPoint(self.north + along * cos - across * sin,
                                      self.east + along * sin + across * cos))
        return points


@dataclass(frozen=True)
class Distractor:
    kind: str
    north: float
    east: float
    yaw: float
    size: float  # longest side, meters
    shape_seed: int


@dataclass
class Scenario:
    name: str
    description: str
    expect: str  # "deliver", "land" or "uncertain" (exact minimum, no slack)
    half_width: float
    cartolinas: list[Cartolina]
    distractors: list[Distractor] = field(default_factory=list)
    ground: str | None = None  # ground texture family; None keeps the plain runway
    ground_seed: int = 0
    target_east: float = 0.0
    # Direction the sunlight travels (ENU); None keeps the world's default.
    sun: tuple[float, float, float] | None = None

    @property
    def target(self) -> GroundPoint:
        return GroundPoint(LENGTH, self.target_east)

    @property
    def track(self) -> list[GroundPoint]:
        w = self.half_width
        return [GroundPoint(-BEHIND, -w), GroundPoint(LENGTH + AHEAD, -w),
                GroundPoint(LENGTH + AHEAD, w), GroundPoint(-BEHIND, w)]

    def has_path(self, planner: PlannerConfig | None = None) -> bool:
        planner = planner or PlannerConfig()
        grid = OccupancyGrid(self.track, planner)
        for cartolina in self.cartolinas:
            for _ in range(planner.confirm_hits):
                grid.add_observation(cartolina.corners())
        return plan_path(grid, GroundPoint(0.0, 0.0), self.target) is not None


Layout = Callable[[float], Scenario]
"""Builds a scenario for a given half width, so the width can be searched."""


def narrowest(layout: Layout, planner: PlannerConfig | None = None, *,
              at_least: float = 0.0) -> Scenario:
    """The scenario at the smallest width (step `WIDTH_STEP`) that has a path,
    with half width no less than `at_least`."""
    # Below twice the inflated margin not even an empty track has a path.
    planner = planner or PlannerConfig()
    first = math.floor(max(2 * planner.inflation_radius, 2 * at_least) / WIDTH_STEP)
    for step in range(first, round(MAX_WIDTH / WIDTH_STEP) + 1):
        scenario = layout(step * WIDTH_STEP / 2)
        if scenario.has_path(planner):
            return scenario
    raise ValueError(f"no width up to {MAX_WIDTH} m has a path")


# ---------------------------------------------------------------- fixed scenarios


def _cartolina(rng: np.random.Generator, north: float, east: float,
               yaw: float | None = None) -> Cartolina:
    return Cartolina(
        north=north,
        east=east,
        yaw=float(rng.uniform(-40, 40)) if yaw is None else yaw,
        length=float(rng.uniform(*CARTOLINA_LONG)),
        width=float(rng.uniform(*CARTOLINA_SHORT)),
        texture_seed=int(rng.integers(1 << 30)),
        family=str(rng.choice(CARTOLINA_FAMILIES)),
    )


def _distractors(rng: np.random.Generator, half_width: float, count: int) -> list[Distractor]:
    return [
        Distractor(
            kind=str(rng.choice(DISTRACTOR_KINDS)),
            north=float(rng.uniform(1.0, LENGTH - 1.0)),
            east=float(rng.uniform(-half_width + 0.3, half_width - 0.3)),
            yaw=float(rng.uniform(0, 360)),
            size=float(rng.uniform(0.4, 0.7)),
            shape_seed=int(rng.integers(1 << 30)),
        )
        for _ in range(count)
    ]


def zigzag(half_width: float) -> Scenario:
    """Cartolinas touching the left and right edges in turn, every 2 m."""
    rng = np.random.default_rng(1)
    cartolinas = []
    for index, north in enumerate(np.arange(2.0, LENGTH - 1.0, 2.0)):
        side = 1 if index % 2 == 0 else -1
        cartolinas.append(_cartolina(rng, float(north), side * (half_width - 0.25), yaw=0.0))
    return Scenario("zigzag", "cartolinas nas bordas, alternando lado a cada 2 m", "deliver",
                    half_width, cartolinas, _distractors(rng, half_width, 3), "grass", 1)


def corridor(half_width: float) -> Scenario:
    """Pairs of cartolinas on both edges: a straight corridor down the middle."""
    rng = np.random.default_rng(2)
    cartolinas = []
    for north in np.arange(2.0, LENGTH - 1.0, 1.8):
        for side in (1, -1):
            cartolinas.append(_cartolina(rng, float(north), side * (half_width - 0.25), yaw=0.0))
    return Scenario("corridor", "corredor reto entre cartolinas nas duas bordas", "deliver",
                    half_width, cartolinas, _distractors(rng, half_width, 3), "concrete", 2)


def dense(half_width: float) -> Scenario:
    """Rows of cartolinas 1.5-2.5 m apart, each at a different place across the track."""
    rng = np.random.default_rng(3)
    cartolinas = []
    north = 2.0
    while north < LENGTH - 1.5:
        across = float(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
        cartolinas.append(_cartolina(rng, north, across * (half_width - 0.25)))
        north += float(rng.uniform(1.5, 2.5))
    return Scenario("dense", "uma cartolina a cada 1,5-2,5 m, em posições variadas", "deliver",
                    half_width, cartolinas, _distractors(rng, half_width, 4), "dirt", 3)


def _wall(rng: np.random.Generator, north: float, half_width: float, gap: float | None,
          gap_east: float = 0.0) -> list[Cartolina]:
    """Cartolinas side by side across the track, edge to edge, leaving one gap of
    exactly `gap` meters between the cartolina edges (or none)."""
    spans = [(-half_width, half_width)] if gap is None else [
        (-half_width, gap_east - gap / 2), (gap_east + gap / 2, half_width)]
    cartolinas = []
    for start, end in spans:
        count = max(1, math.ceil((end - start) / 0.5 - 1e-9))
        width = (end - start) / count
        for index in range(count):
            east = start + width * (index + 0.5)
            cartolinas.append(Cartolina(north, east, 0.0, 0.7, width,
                                        int(rng.integers(1 << 30)), "solid"))
    return cartolinas


def gap_layout(gap: float, half_width: float = 3.0) -> Scenario:
    rng = np.random.default_rng(4)
    cartolinas = [_cartolina(rng, 3.0, 1.0), *_wall(rng, 8.0, half_width, gap, gap_east=-0.8),
                  _cartolina(rng, 12.0, -1.2)]
    return Scenario("narrow_gap", f"parede de cartolinas com uma passagem de {gap:.2f} m",
                    "deliver", half_width, cartolinas, _distractors(rng, half_width, 3),
                    "asphalt", 4)


def narrowest_gap(planner: PlannerConfig | None = None) -> tuple[float, Scenario]:
    """The narrowest gap (step 0.05 m) the planner still goes through."""
    for step in range(1, 120):
        gap = round(step * 0.05, 2)
        scenario = gap_layout(gap)
        if scenario.has_path(planner):
            return gap, scenario
    raise ValueError("no gap up to 6 m has a path")


def closed(half_width: float = 3.0) -> Scenario:
    """A wall of cartolinas across the whole track: no path, the drone must land."""
    rng = np.random.default_rng(5)
    cartolinas = [_cartolina(rng, 3.5, -0.8), *_wall(rng, 8.0, half_width, None)]
    return Scenario("closed", "parede de cartolinas fechando a pista (sem caminho)", "land",
                    half_width, cartolinas, _distractors(rng, half_width, 2), "gravel", 5)


def with_slack(layout: Layout, planner: PlannerConfig | None = None) -> Scenario:
    """The narrowest scenario with a path, widened by `SLACK`."""
    return layout(narrowest(layout, planner).half_width + SLACK / 2)


def serpentine(half_width: float) -> Scenario:
    """Walls of 4 cartolinas side by side ([][][][]), each against one edge and
    alternating sides every 3.5 m: the straight line is blocked, the drone has
    to go around every wall."""
    rng = np.random.default_rng(7)
    cartolinas = []
    for index, north in enumerate((3.5, 7.0, 10.5)):
        side = 1 if index % 2 == 0 else -1
        for slot in range(4):
            east = side * (half_width - 0.25 - 0.5 * slot)
            cartolinas.append(Cartolina(north, east, 0.0, 0.7, 0.5,
                                        int(rng.integers(1 << 30)), "solid"))
    return Scenario("serpentine", "paredes de 4 cartolinas ([][][][]) presas a bordas "
                    "alternadas: sem linha reta, contorna cada parede", "deliver",
                    half_width, cartolinas, _distractors(rng, half_width, 3), "grass", 7)


# ---------------------------------------------------------------- focused scenarios
# Each one targets one point of attention, or a layout expected on the day.


def _solid(north: float, east: float, yaw: float = 0.0,
           rgb: tuple[float, float, float] | None = None, seed: int = 0) -> Cartolina:
    """A 0.6 x 0.45 m cartolina."""
    return Cartolina(north, east, yaw, 0.6, 0.45, seed, "solid", rgb)


def edge_entry(half_width: float) -> Scenario:
    """A wall of 4 cartolinas right in front of the drone: it enters the photo
    through its top edge, cut, before it is seen whole."""
    rng = np.random.default_rng(11)
    cartolinas = [Cartolina(7.0, -half_width + 0.25 + 0.5 * slot, 0.0, 0.7, 0.5,
                            int(rng.integers(1 << 30)), "solid") for slot in range(4)]
    return Scenario("edge_entry", "parede de 4 cartolinas em frente, entrando pela borda "
                    "da foto", "deliver", half_width, cartolinas, [], "concrete", 11)


def low_contrast(half_width: float) -> Scenario:
    """Green and olive cartolinas on grass."""
    greens = [(0.22, 0.42, 0.12), (0.35, 0.45, 0.18), (0.18, 0.35, 0.10), (0.30, 0.40, 0.15)]
    cartolinas = [_solid(north, side * (half_width - 0.6), 15.0 * side, rgb)
                  for north, side, rgb in zip((3.5, 6.5, 9.5, 12.0), (1, -1, 1, -1), greens,
                                              strict=True)]
    return Scenario("low_contrast", "cartolinas verdes e oliva sobre grama", "deliver",
                    half_width, cartolinas, [], "grass", 12)


def distractors_on_path(half_width: float) -> Scenario:
    """Shapes that are not cartolinas on the straight line; one real cartolina."""
    rng = np.random.default_rng(13)
    distractors = [Distractor(kind, north, 0.0, float(rng.uniform(0, 360)), 0.6,
                              int(rng.integers(1 << 30)))
                   for kind, north in (("parallelogram", 3.5), ("ellipse", 6.0),
                                       ("triangle", 8.5), ("trapezoid", 11.0))]
    cartolinas = [_solid(13.0, half_width - 0.5, seed=13)]
    return Scenario("distractors_on_path", "formas que não são cartolina na linha reta "
                    "(paralelogramo, elipse, triângulo, trapézio)", "deliver", half_width,
                    cartolinas, distractors, "asphalt", 13)


def rotated_45(half_width: float) -> Scenario:
    """Cartolinas at 45 degrees, staggered: the way through is diagonal."""
    cartolinas = [_solid(north, side * (half_width - 0.6), 45.0, seed=14 + index)
                  for index, (north, side) in enumerate(((4.0, 1), (6.0, -1), (8.0, 1),
                                                         (10.0, -1)))]
    return Scenario("rotated_45", "cartolinas a 45° desencontradas: passagem diagonal",
                    "deliver", half_width, cartolinas, [], "dirt", 14)


def near_target(half_width: float) -> Scenario:
    """A cartolina 1.6 m before the target, on the straight line."""
    cartolinas = [_solid(LENGTH - 1.6, 0.0, seed=15)]
    return Scenario("near_target", "cartolina 1,6 m antes do alvo, na linha reta", "deliver",
                    half_width, cartolinas, [], "tiles", 15)


def low_sun(half_width: float) -> Scenario:
    """Low sun from the side: the drone casts a long, sharp shadow on plain ground."""
    cartolinas = [_solid(5.0, half_width - 0.6, seed=16), _solid(10.0, -half_width + 0.6,
                                                                 seed=17)]
    return Scenario("low_sun", "sol baixo de lado: sombra longa e nítida do drone", "deliver",
                    half_width, cartolinas, [], None, 16, sun=(-0.95, 0.1, -0.3))


def single_ahead(half_width: float) -> Scenario:
    """The most common case: one cartolina in the middle of the way."""
    return Scenario("single_ahead", "uma cartolina no meio do caminho", "deliver", half_width,
                    [_solid(7.5, 0.0, seed=18)], [], "grass", 18)


def staggered_pair(half_width: float) -> Scenario:
    """Two cartolinas, right then left."""
    cartolinas = [_solid(5.0, 0.6, seed=19), _solid(10.0, -0.6, seed=20)]
    return Scenario("staggered_pair", "duas cartolinas desencontradas (direita, depois "
                    "esquerda)", "deliver", half_width, cartolinas, [], "concrete", 19)


def offset_target(half_width: float) -> Scenario:
    """Target 2 m to the side: the drone flies diagonally, with one cartolina on the way."""
    return Scenario("offset_target", "alvo deslocado 2 m para leste, voo em diagonal",
                    "deliver", half_width, [_solid(7.5, 1.0, seed=21)], [], "dirt", 21,
                    target_east=2.0)


FOCUSED: list[Layout] = [edge_entry, low_contrast, distractors_on_path, rotated_45,
                         near_target, low_sun, single_ahead, staggered_pair, offset_target]


def focused_scenarios(planner: PlannerConfig | None = None) -> list[Scenario]:
    return [with_slack(layout, planner) for layout in FOCUSED]


def fixed_scenarios(planner: PlannerConfig | None = None) -> list[Scenario]:
    gap, _ = narrowest_gap(planner)
    through = gap_layout(round(gap + SLACK, 2))
    exact = narrowest(zigzag, planner)
    exact.name = "zigzag_exact"
    exact.expect = "uncertain"
    exact.description += ", na largura mínima exata (sem folga)"
    too_narrow = gap_layout(round(gap - 0.1, 2))
    too_narrow.name = "too_narrow_gap"
    too_narrow.expect = "land"
    too_narrow.description = (f"passagem de {gap - 0.1:.2f} m, 0,1 m abaixo da mínima "
                              f"({gap:.2f} m): sem caminho")
    too_narrow.ground, too_narrow.ground_seed = "tiles", 6
    return [*focused_scenarios(planner), with_slack(zigzag, planner),
            with_slack(serpentine, planner), with_slack(corridor, planner),
            with_slack(dense, planner), through, too_narrow, closed(), exact]


# ---------------------------------------------------------------- random scenario


def random_layout(seed: int, count: int) -> tuple[Layout, float]:
    """`count` cartolinas at seeded places, at least 1.5 m apart along the track and
    up to 1.5 m to either side of its center line. Returns the layout and the
    smallest half width that keeps every cartolina inside the track."""
    # Kept clear of HOME and of the target, which the planner needs free.
    first, last, spacing = 2.5, LENGTH - 2.5, 1.5
    if count < 1 or (count - 1) * spacing > last - first:
        raise ValueError(f"count must be 1-{int((last - first) / spacing) + 1}")
    rng = np.random.default_rng(seed)
    # Spare room beyond the minimum spacing, split at random between the gaps.
    spare = (last - first) - (count - 1) * spacing
    offsets = np.sort(rng.uniform(0.0, spare, count))
    norths = first + offsets + spacing * np.arange(count)
    easts = rng.uniform(-1.5, 1.5, count)
    cartolinas = [
        Cartolina(float(north), float(east), float(rng.uniform(-45, 45)),
                  float(rng.uniform(*CARTOLINA_LONG)), float(rng.uniform(*CARTOLINA_SHORT)),
                  int(rng.integers(1 << 30)), str(rng.choice(CARTOLINA_FAMILIES)))
        for north, east in zip(norths, easts, strict=True)
    ]
    ground = str(rng.choice(GROUNDS))
    distractor_seed = int(rng.integers(1 << 30))
    inside = max(abs(c.east) + max(c.length, c.width) / 2 for c in cartolinas)

    def layout(half_width: float) -> Scenario:
        distractors = _distractors(np.random.default_rng(distractor_seed), half_width, 3)
        return Scenario(f"random_{seed}", f"{len(cartolinas)} cartolinas aleatórias "
                        f"(semente {seed})", "deliver", half_width, list(cartolinas),
                        distractors, ground, seed)

    return layout, inside


def random_scenario(seed: int, count: int, *, expect: str = "deliver",
                    planner: PlannerConfig | None = None) -> Scenario:
    """At the narrowest width with a path plus `SLACK`; with `expect="land"`, one
    step narrower than the exact minimum, where the planner finds no path."""
    layout, inside = random_layout(seed, count)
    scenario = narrowest(layout, planner, at_least=inside)
    if expect == "deliver":
        return layout(scenario.half_width + SLACK / 2)
    narrower = layout(scenario.half_width - WIDTH_STEP / 2)
    if narrower.half_width < inside or narrower.has_path(planner):
        raise ValueError(f"seed {seed}: every width that holds the cartolinas has a path")
    narrower.name = f"random_{seed}_land"
    narrower.expect = "land"
    narrower.description += ", sem caminho"
    return narrower
