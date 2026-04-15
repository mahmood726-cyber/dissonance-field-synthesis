from dfs.dissonance import pairwise_dissonance


def test_same_population_diff_adherence_not_pooled(synth_trial_pair_same_pop) -> None:
    """Two trials identical except adherence_proxy are reported as separate
    boundary conditions with non-trivial dissonance that's fully accounted
    for by the adherence covariate."""
    a, b = synth_trial_pair_same_pop
    pairs = pairwise_dissonance([a, b], outcome="primary_composite")
    assert len(pairs) == 1
    p = pairs[0]
    # Dissonance should be measurable (not zero).
    assert p.d > 1.0
    # Covariate delta is entirely in adherence_proxy.
    cov_non_adherence = {k: v for k, v in p.covariate_delta.items() if k != "adherence_proxy"}
    assert all(abs(v) < 1e-9 for v in cov_non_adherence.values())
    assert abs(p.covariate_delta["adherence_proxy"]) > 0.1
