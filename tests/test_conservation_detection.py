import numpy as np
import pytest

from dfs.config import CONSERVATION_VIOLATION_SIGMA
from dfs.schema import TrialBoundaryCondition
from dfs.config import COVARIATE_NAMES


def _make_inconsistent_trial() -> TrialBoundaryCondition:
    """ACM HR deliberately inconsistent with CV + non-CV decomposition."""
    return TrialBoundaryCondition(
        trial_id="INCONSISTENT",
        drug="spironolactone",
        mr_occupancy_equivalent=1.0,
        anchor_covariates={n: 0.0 for n in COVARIATE_NAMES},
        covariate_ranges={},
        outcomes={
            "primary_composite": {"log_hr": -0.20, "se": 0.05, "baseline_prop": 1.0},
            "acm":               {"log_hr":  0.40, "se": 0.05, "baseline_prop": 1.0},   # WRONG
            "cv_death":          {"log_hr": -0.30, "se": 0.06, "baseline_prop": 0.55},
            "non_cv_death":      {"log_hr": -0.10, "se": 0.07, "baseline_prop": 0.45},
        },
        safety={},
        design_priors={"placebo_rate_per_yr": 0.15, "ltfu_fraction": 0.05,
                       "adherence_proxy": 0.9},
    )


def test_detects_inconsistent_mortality_decomposition() -> None:
    """A trial with HR(ACM) not matching p·HR(CV)+q·HR(nonCV) must be flagged."""
    from dfs.diagnostics import detect_conservation_violations

    trial = _make_inconsistent_trial()
    violations = detect_conservation_violations([trial])
    assert len(violations) >= 1
    v0 = violations[0]
    assert v0.law_name == "mortality_decomposition"
    assert v0.trial_id == "INCONSISTENT"
    assert v0.sigma_magnitude > CONSERVATION_VIOLATION_SIGMA


def test_consistent_trial_has_no_violations(synth_trial_a) -> None:
    """A trial where ACM log_hr is consistent with CV+nonCV decomposition
    should produce no violations."""
    from dfs.diagnostics import detect_conservation_violations

    # synth_trial_a has: acm=-0.08, cv=-0.12 (prop=0.55), non_cv=-0.02 (prop=0.45)
    # Expected: 0.55 * -0.12 + 0.45 * -0.02 = -0.066 + -0.009 = -0.075 ≈ -0.08.
    violations = detect_conservation_violations([synth_trial_a])
    assert violations == []
