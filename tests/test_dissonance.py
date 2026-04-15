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
