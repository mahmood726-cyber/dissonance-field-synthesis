import math

import pytest


def test_dissonance_formula(synth_trial_a, synth_trial_b) -> None:
    from dfs.dissonance import pairwise_dissonance

    pairs = pairwise_dissonance([synth_trial_a, synth_trial_b], outcome="primary_composite")
    assert len(pairs) == 1
    p = pairs[0]
    # d_ij = |log-HR_a - log-HR_b| / sqrt(SE_a^2 + SE_b^2)
    # Both synth trials use the default primary_composite log_hr=-0.15, se=0.09
    la, lb = -0.15, -0.15
    sa, sb = 0.09, 0.09
    expected_d = abs(la - lb) / math.sqrt(sa**2 + sb**2)
    assert p.d == pytest.approx(expected_d, abs=1e-9)
    assert p.trial_ids == ("SYNTH-A", "SYNTH-B")


def test_covariate_distance_vector(synth_trial_a, synth_trial_b) -> None:
    from dfs.dissonance import pairwise_dissonance

    pairs = pairwise_dissonance([synth_trial_a, synth_trial_b], outcome="primary_composite")
    p = pairs[0]
    # synth_trial_b has lvef=65 vs a=55, adherence=0.6 vs 0.9, all else equal.
    assert p.covariate_delta["lvef"] == pytest.approx(10.0)
    assert p.covariate_delta["adherence_proxy"] == pytest.approx(-0.3)


def test_all_pairs_counted() -> None:
    # 6 trials -> C(6,2) = 15 pairs
    pytest.skip("Combinatorial count test covered in integration (Task 12)")


def test_log_hr_delta_sign_convention(synth_trial_a, synth_trial_b) -> None:
    """log_hr_delta uses (trial_b - trial_a) direction, matching covariate_delta."""
    from dfs.dissonance import pairwise_dissonance

    # Build trials where a's log_hr is more negative than b's.
    from dfs.schema import TrialBoundaryCondition
    from dfs.config import COVARIATE_NAMES

    def _make_with_log_hr(tid: str, log_hr: float) -> TrialBoundaryCondition:
        return TrialBoundaryCondition(
            trial_id=tid,
            drug="spironolactone",
            mr_occupancy_equivalent=1.0,
            anchor_covariates={n: 0.0 for n in COVARIATE_NAMES},
            covariate_ranges={},
            outcomes={"primary_composite": {"log_hr": log_hr, "se": 0.1, "baseline_prop": 1.0}},
            safety={}, design_priors={},
        )

    a = _make_with_log_hr("A", -0.3)
    b = _make_with_log_hr("B", -0.1)
    pairs = pairwise_dissonance([a, b], outcome="primary_composite")
    p = pairs[0]
    assert p.trial_ids == ("A", "B")
    # b_minus_a = -0.1 - (-0.3) = +0.2
    assert p.log_hr_delta == pytest.approx(0.2, abs=1e-9)
