import numpy as np


def test_end_to_end_synthetic_pipeline_no_silent_sentinels(
    synth_trial_a, synth_trial_b
) -> None:
    from dfs.dissonance import pairwise_dissonance
    from dfs.adherence_proxy import adherence_proxy
    from dfs.decisions import DECISION_THRESHOLDS, PER_PATIENT_VAR
    from dfs.mind_change import mind_change_price
    from dfs.feasibility import feasibility_region

    trials = [synth_trial_a, synth_trial_b]

    # 1. Dissonance runs on real outcome field names.
    pairs = pairwise_dissonance(trials, outcome="primary_composite")
    assert pairs, "pairwise_dissonance returned empty"
    for p in pairs:
        assert p.d is not None
        assert not (isinstance(p.d, str) and p.d.startswith("unknown"))

    # 2. Adherence proxy produces valid scalars for every trial.
    for t in trials:
        score = adherence_proxy(t)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # 3. Decisions dict covers endpoint used by dissonance.
    assert "primary_composite" in DECISION_THRESHOLDS
    assert "primary_composite" in PER_PATIENT_VAR

    # 4. MCP computes a finite or +inf value — never None or string.
    thresholds = DECISION_THRESHOLDS["primary_composite"]
    t_cross = thresholds["recommend_if_log_hr_below"]
    mu = synth_trial_a.outcomes["primary_composite"]["log_hr"]
    var = synth_trial_a.outcomes["primary_composite"]["se"] ** 2
    n = mind_change_price(
        posterior_mean=mu, posterior_var=var,
        t_cross=t_cross, t_obs=0.0,
        per_patient_var=PER_PATIENT_VAR["primary_composite"],
    )
    assert isinstance(n, float)
    assert not (isinstance(n, float) and np.isnan(n))

    # 5. Feasibility region output is boolean array.
    mu_arr = np.array([mu])
    var_arr = np.array([var])
    mask = feasibility_region(mu_arr, var_arr, threshold=t_cross, ci_level=0.95, direction="below")
    assert mask.dtype == bool


def test_covariate_name_contract_stable() -> None:
    """All modules share the 7-D covariate vocabulary via config.COVARIATE_NAMES."""
    from dfs.config import COVARIATE_NAMES

    assert set(COVARIATE_NAMES) == {
        "lvef", "egfr", "age", "baseline_k", "dm_fraction",
        "mr_occupancy", "adherence_proxy",
    }


def test_outcome_name_contract_stable() -> None:
    from dfs.config import OUTCOME_NAMES
    expected = {
        "primary_composite", "acm", "cv_death", "non_cv_death",
        "hf_hosp", "sudden_death", "pump_failure", "mi", "stroke",
    }
    assert set(OUTCOME_NAMES) == expected
