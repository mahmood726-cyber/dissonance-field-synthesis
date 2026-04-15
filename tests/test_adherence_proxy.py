"""Tests for dfs.adherence_proxy."""
from __future__ import annotations


def test_high_adherence_trial_scores_high(synth_trial_a) -> None:
    from dfs.adherence_proxy import adherence_proxy
    score = adherence_proxy(synth_trial_a)
    assert 0.0 <= score <= 1.0
    assert score >= 0.6


def test_zero_placebo_event_rate_penalised() -> None:
    """A trial reporting zero placebo events in a high-risk population
    is the TOPCAT-Russia signal — adherence must be penalised."""
    from dfs.adherence_proxy import adherence_proxy
    from dfs.schema import TrialBoundaryCondition
    from dfs.config import COVARIATE_NAMES

    bad = TrialBoundaryCondition(
        trial_id="BAD-ADHERENCE",
        drug="spironolactone",
        mr_occupancy_equivalent=1.0,
        anchor_covariates={n: 0.0 for n in COVARIATE_NAMES},
        covariate_ranges={},
        outcomes={"primary_composite": {"log_hr": 0.0, "se": 0.1, "baseline_prop": 1.0}},
        safety={},
        design_priors={
            "placebo_rate_per_yr": 0.005,
            "ltfu_fraction": 0.25,
            "adherence_proxy": 0.3,
        },
    )
    assert adherence_proxy(bad) < 0.5


def test_output_bounded_in_unit_interval(synth_trial_a) -> None:
    from dfs.adherence_proxy import adherence_proxy
    score = adherence_proxy(synth_trial_a)
    assert 0.0 <= score <= 1.0
