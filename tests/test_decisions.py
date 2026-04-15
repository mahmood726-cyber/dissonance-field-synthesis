def test_every_primary_endpoint_has_thresholds() -> None:
    from dfs.decisions import DECISION_THRESHOLDS

    for endpoint in ("primary_composite", "acm", "cv_death", "hf_hosp"):
        assert endpoint in DECISION_THRESHOLDS


def test_threshold_shape() -> None:
    from dfs.decisions import DECISION_THRESHOLDS

    for endpoint, thresholds in DECISION_THRESHOLDS.items():
        assert "recommend_if_log_hr_below" in thresholds
        assert "do_not_recommend_if_log_hr_above" in thresholds
        assert thresholds["recommend_if_log_hr_below"] < thresholds["do_not_recommend_if_log_hr_above"]


def test_per_patient_variance_provided() -> None:
    from dfs.decisions import PER_PATIENT_VAR

    for endpoint, v in PER_PATIENT_VAR.items():
        assert 0.1 <= v <= 10.0
