"""Shared pytest fixtures: synthetic boundary-condition records."""
from __future__ import annotations

import pytest

from dfs.config import COVARIATE_NAMES
from dfs.schema import TrialBoundaryCondition


def _make(trial_id: str, **overrides) -> TrialBoundaryCondition:
    anchor = {n: v for n, v in zip(
        COVARIATE_NAMES,
        (55.0, 60.0, 70.0, 4.3, 0.3, 1.0, 0.9),
    )}
    anchor.update(overrides.pop("anchor_overrides", {}))
    defaults = dict(
        trial_id=trial_id,
        drug="spironolactone",
        mr_occupancy_equivalent=1.0,
        anchor_covariates=anchor,
        covariate_ranges={
            "lvef": (45.0, 75.0), "egfr": (30.0, 120.0),
            "age": (55.0, 85.0), "baseline_k": (3.5, 5.0),
            "dm_fraction": (0.0, 1.0), "mr_occupancy": (0.0, 2.0),
            "adherence_proxy": (0.0, 1.0),
        },
        outcomes={
            "primary_composite": {"log_hr": -0.15, "se": 0.09, "baseline_prop": 1.0},
            "acm": {"log_hr": -0.08, "se": 0.08, "baseline_prop": 1.0},
            "cv_death": {"log_hr": -0.12, "se": 0.10, "baseline_prop": 0.55},
            "non_cv_death": {"log_hr": -0.02, "se": 0.12, "baseline_prop": 0.45},
        },
        safety={"delta_k": {"value": 0.2, "se": 0.04}},
        design_priors={"placebo_rate_per_yr": 0.15, "ltfu_fraction": 0.08,
                       "adherence_proxy": 0.9},
    )
    defaults.update(overrides)
    return TrialBoundaryCondition(**defaults)


@pytest.fixture
def synth_trial_a() -> TrialBoundaryCondition:
    return _make("SYNTH-A")


@pytest.fixture
def synth_trial_b() -> TrialBoundaryCondition:
    return _make(
        "SYNTH-B",
        anchor_overrides={"lvef": 65.0, "adherence_proxy": 0.6},
    )


@pytest.fixture
def synth_trial_pair_same_pop() -> tuple[TrialBoundaryCondition, TrialBoundaryCondition]:
    """Two trials identical except adherence-proxy — for dissonance-resolution test."""
    a = _make("PAIR-A", anchor_overrides={"adherence_proxy": 0.95})
    b = _make(
        "PAIR-B",
        anchor_overrides={"adherence_proxy": 0.40},
        outcomes={
            "primary_composite": {"log_hr": 0.05, "se": 0.10, "baseline_prop": 1.0},
            "acm": {"log_hr": 0.10, "se": 0.09, "baseline_prop": 1.0},
            "cv_death": {"log_hr": 0.08, "se": 0.11, "baseline_prop": 0.55},
            "non_cv_death": {"log_hr": 0.12, "se": 0.13, "baseline_prop": 0.45},
        },
    )
    return a, b
