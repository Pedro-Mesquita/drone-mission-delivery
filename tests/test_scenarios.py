"""The simulated scenarios have, or lack, a path as they say (sim/scenarios.py)."""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))

from scenarios import (
    WIDTH_STEP,
    Scenario,
    fixed_scenarios,
    narrowest,
    random_layout,
    random_scenario,
    zigzag,
)

FIXED = fixed_scenarios()


@pytest.mark.parametrize("scenario", FIXED, ids=[s.name for s in FIXED])
def test_fixed_scenario_matches_its_expectation(scenario: Scenario) -> None:
    """The exact-minimum scenario has a path only if detection is perfect."""
    assert scenario.has_path() is (scenario.expect in ("deliver", "uncertain"))


@pytest.mark.parametrize("scenario", [s for s in FIXED if s.expect == "deliver"],
                         ids=[s.name for s in FIXED if s.expect == "deliver"])
def test_delivering_scenarios_survive_detection_error(scenario: Scenario) -> None:
    """Every cartolina 5 cm bigger on each side still leaves a path."""
    grown = [dataclasses.replace(c, length=c.length + 0.1, width=c.width + 0.1)
             for c in scenario.cartolinas]
    assert dataclasses.replace(scenario, cartolinas=grown).has_path()


def test_the_width_is_the_narrowest_with_a_path() -> None:
    scenario = narrowest(zigzag)

    assert scenario.has_path()
    assert not zigzag(scenario.half_width - WIDTH_STEP / 2).has_path()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_scenarios_deliver_and_land(seed: int) -> None:
    assert random_scenario(seed, 7).has_path()
    assert not random_scenario(seed, 7, expect="land").has_path()


def test_random_scenario_is_repeatable() -> None:
    assert random_scenario(5, 7).cartolinas == random_scenario(5, 7).cartolinas


def test_too_many_cartolinas_for_the_track_are_rejected() -> None:
    with pytest.raises(ValueError, match="count must be"):
        random_layout(1, 50)
