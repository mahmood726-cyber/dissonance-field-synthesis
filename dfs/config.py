"""Shared numeric constants. All magic numbers live here."""

MA_EQUIVALENCE_TOL: float = 1e-3
CONSERVATION_VIOLATION_SIGMA: float = 2.0
VIRTUAL_GRID_SIZE: int = 200
QP_SOLVER_ATOL: float = 1e-6
SE_COMBINED_FLOOR: float = 1e-12

COVARIATE_NAMES: tuple[str, ...] = (
    "lvef", "egfr", "age", "baseline_k", "dm_fraction",
    "mr_occupancy", "adherence_proxy",
)
assert len(COVARIATE_NAMES) == 7, "Covariate space is 7-D by design"

OUTCOME_NAMES: tuple[str, ...] = (
    "primary_composite", "acm", "cv_death", "non_cv_death",
    "hf_hosp", "sudden_death", "pump_failure", "mi", "stroke",
)
assert len(OUTCOME_NAMES) == 9, "1 composite + 8 decomposed outcomes"

SAFETY_NAMES: tuple[str, ...] = ("delta_k", "delta_sbp", "delta_egfr")
assert len(SAFETY_NAMES) == 3, "K+, SBP, eGFR surfaces"
