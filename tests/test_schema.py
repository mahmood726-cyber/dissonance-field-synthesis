"""TrialBoundaryCondition roundtrips JSON and enforces required fields."""
import json
from pathlib import Path

import pytest


def test_minimal_record_roundtrips(tmp_path: Path) -> None:
    from dfs.schema import TrialBoundaryCondition

    record = TrialBoundaryCondition(
        trial_id="TEST-1",
        drug="spironolactone",
        mr_occupancy_equivalent=1.0,
        anchor_covariates={
            "lvef": 57.0, "egfr": 66.0, "age": 72.0, "baseline_k": 4.3,
            "dm_fraction": 0.32, "mr_occupancy": 1.0, "adherence_proxy": 0.85,
        },
        covariate_ranges={"lvef": (45.0, 75.0), "egfr": (30.0, 120.0)},
        outcomes={
            "primary_composite": {"log_hr": -0.186, "se": 0.080, "baseline_prop": 1.0},
            "acm": {"log_hr": -0.08, "se": 0.07, "baseline_prop": 1.0},
        },
        safety={"delta_k": {"value": 0.21, "se": 0.03}},
        design_priors={"placebo_rate_per_yr": 0.18, "ltfu_fraction": 0.09,
                       "adherence_proxy": 0.85},
    )
    path = tmp_path / "trial.json"
    record.to_json(path)
    roundtripped = TrialBoundaryCondition.from_json(path)
    assert roundtripped == record


def test_missing_covariate_raises() -> None:
    from dfs.schema import TrialBoundaryCondition

    with pytest.raises(KeyError, match="lvef"):
        TrialBoundaryCondition(
            trial_id="BAD",
            drug="spironolactone",
            mr_occupancy_equivalent=1.0,
            anchor_covariates={"egfr": 66.0},  # lvef missing
            covariate_ranges={},
            outcomes={},
            safety={},
            design_priors={},
        )


def test_silent_failure_sentinel_refused() -> None:
    """Schema must never accept 'unknown' or None for required numeric fields."""
    from dfs.schema import TrialBoundaryCondition

    with pytest.raises((TypeError, ValueError)):
        TrialBoundaryCondition(
            trial_id="BAD",
            drug="spironolactone",
            mr_occupancy_equivalent=None,  # type: ignore[arg-type]
            anchor_covariates={n: 0.0 for n in (
                "lvef", "egfr", "age", "baseline_k",
                "dm_fraction", "mr_occupancy", "adherence_proxy",
            )},
            covariate_ranges={},
            outcomes={},
            safety={},
            design_priors={},
        )
