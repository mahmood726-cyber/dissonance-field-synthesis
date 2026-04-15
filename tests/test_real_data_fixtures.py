"""Tests for real-data MRA-HFpEF trial boundary-condition fixtures (Task 13)."""
from pathlib import Path

import pytest

from dfs.manifest import load_trials


MANIFEST_PATH = Path("data/mra_hfpef/MANIFEST.json")
EXPECTED_IDS = {
    "TOPCAT-Americas", "TOPCAT-Russia-Georgia", "FINEARTS-HF",
    "FIDELIO-DKD-HF-subgroup", "FIGARO-DKD-HF-subgroup", "Aldo-DHF",
}


def test_manifest_lists_six_trials() -> None:
    trials = load_trials(MANIFEST_PATH)
    assert {t.trial_id for t in trials} == EXPECTED_IDS


def test_every_trial_has_primary_composite() -> None:
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        assert "primary_composite" in t.outcomes


def test_every_trial_has_mortality_decomposition() -> None:
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        for key in ("acm", "cv_death", "non_cv_death"):
            assert key in t.outcomes, f"{t.trial_id} missing {key}"


def test_every_trial_has_source_citation() -> None:
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        for outcome_name, outcome in t.outcomes.items():
            assert "source" in outcome, (
                f"{t.trial_id}.outcomes[{outcome_name}] missing 'source'"
            )


def test_design_priors_populated() -> None:
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        assert "placebo_rate_per_yr" in t.design_priors
        assert "ltfu_fraction" in t.design_priors


def test_all_seven_anchor_covariates_present() -> None:
    """Every trial must have all 7 required covariate keys."""
    from dfs.config import COVARIATE_NAMES
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        for name in COVARIATE_NAMES:
            assert name in t.anchor_covariates, (
                f"{t.trial_id} missing anchor covariate '{name}'"
            )


def test_mr_occupancy_equivalent_range() -> None:
    """MR-occupancy-equivalent must be positive and plausible (0.1-3.0)."""
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        assert 0.1 <= t.mr_occupancy_equivalent <= 3.0, (
            f"{t.trial_id}: mr_occupancy_equivalent {t.mr_occupancy_equivalent} out of range"
        )


def test_topcat_adherence_proxy_split() -> None:
    """Americas adherence must be substantially higher than Russia/Georgia."""
    trials = {t.trial_id: t for t in load_trials(MANIFEST_PATH)}
    americas = trials["TOPCAT-Americas"]
    russia = trials["TOPCAT-Russia-Georgia"]
    assert americas.anchor_covariates["adherence_proxy"] >= 0.75
    assert russia.anchor_covariates["adherence_proxy"] <= 0.50


def test_topcat_russia_georgia_low_placebo_rate() -> None:
    """Russia/Georgia placebo event rate must be implausibly low (non-adherence signal)."""
    trials = {t.trial_id: t for t in load_trials(MANIFEST_PATH)}
    russia = trials["TOPCAT-Russia-Georgia"]
    americas = trials["TOPCAT-Americas"]
    # Russia/Georgia placebo event rate must be < 1/4 of Americas rate
    assert russia.design_priors["placebo_rate_per_yr"] < (
        americas.design_priors["placebo_rate_per_yr"] / 4
    ), (
        "Russia/Georgia placebo rate should be much lower than Americas (non-adherence signal)"
    )


def test_finerenone_trials_higher_mr_occupancy() -> None:
    """Finerenone trials must have mr_occupancy_equivalent > 1.0 (more selective MR binding)."""
    finerenone_ids = {"FINEARTS-HF", "FIDELIO-DKD-HF-subgroup", "FIGARO-DKD-HF-subgroup"}
    trials = {t.trial_id: t for t in load_trials(MANIFEST_PATH)}
    for tid in finerenone_ids:
        assert trials[tid].mr_occupancy_equivalent > 1.0, (
            f"{tid}: finerenone should have mr_occupancy_equivalent > 1.0"
        )


def test_baseline_prop_non_negative() -> None:
    """baseline_prop for all outcomes must be non-negative."""
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        for name, outcome in t.outcomes.items():
            bp = outcome.get("baseline_prop", None)
            assert bp is not None, f"{t.trial_id}.outcomes[{name}] missing baseline_prop"
            assert bp >= 0.0, f"{t.trial_id}.outcomes[{name}].baseline_prop = {bp} < 0"


def test_mortality_fraction_sums_to_one() -> None:
    """cv_death.baseline_prop + non_cv_death.baseline_prop should sum to ~1.0."""
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        cv_frac = t.outcomes["cv_death"]["baseline_prop"]
        ncv_frac = t.outcomes["non_cv_death"]["baseline_prop"]
        total = cv_frac + ncv_frac
        assert abs(total - 1.0) < 0.01, (
            f"{t.trial_id}: cv_death.baseline_prop ({cv_frac}) + "
            f"non_cv_death.baseline_prop ({ncv_frac}) = {total}, expected ~1.0"
        )


def test_safety_keys_present() -> None:
    """All three safety endpoints must be present."""
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        for key in ("delta_k", "delta_sbp", "delta_egfr"):
            assert key in t.safety, f"{t.trial_id} missing safety key '{key}'"


def test_round_trip_json_fidelity() -> None:
    """to_json -> from_json must preserve all outcome log_hr values exactly."""
    import tempfile
    trials = load_trials(MANIFEST_PATH)
    for t in trials:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            tmp = Path(fh.name)
        t.to_json(tmp)
        t2 = type(t).from_json(tmp)
        for name, outcome in t.outcomes.items():
            assert abs(t2.outcomes[name]["log_hr"] - outcome["log_hr"]) < 1e-9, (
                f"{t.trial_id}.outcomes[{name}].log_hr changed on round-trip"
            )
        tmp.unlink(missing_ok=True)
