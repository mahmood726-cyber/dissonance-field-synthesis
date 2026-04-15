"""Conservation-law violation diagnostics: fail-closed detection with σ magnitude."""
from __future__ import annotations

import math
from dataclasses import dataclass

from dfs.config import CONSERVATION_VIOLATION_SIGMA
from dfs.schema import TrialBoundaryCondition


@dataclass(frozen=True)
class ConservationViolation:
    trial_id: str
    law_name: str
    sigma_magnitude: float
    detail: str


def detect_conservation_violations(
    trials: list[TrialBoundaryCondition],
    sigma_threshold: float = CONSERVATION_VIOLATION_SIGMA,
) -> list[ConservationViolation]:
    violations: list[ConservationViolation] = []
    for t in trials:
        o = t.outcomes
        if not {"acm", "cv_death", "non_cv_death"}.issubset(o):
            continue
        p = o["cv_death"]["baseline_prop"]
        q = o["non_cv_death"]["baseline_prop"]
        # Hazard-difference approximation: log(HR_ACM) ≈ p*log(HR_CV) + q*log(HR_nonCV)
        predicted = p * o["cv_death"]["log_hr"] + q * o["non_cv_death"]["log_hr"]
        observed = o["acm"]["log_hr"]
        diff = observed - predicted
        se_pred = math.sqrt(
            p**2 * o["cv_death"]["se"]**2 + q**2 * o["non_cv_death"]["se"]**2
        )
        se_diff = math.sqrt(se_pred**2 + o["acm"]["se"]**2)
        sigma = abs(diff) / se_diff if se_diff > 0 else float("inf")
        if sigma > sigma_threshold:
            violations.append(ConservationViolation(
                trial_id=t.trial_id,
                law_name="mortality_decomposition",
                sigma_magnitude=sigma,
                detail=(
                    f"Reported log-HR(ACM)={observed:.3f}, "
                    f"predicted {predicted:.3f} from p={p:.2f}·CV + q={q:.2f}·nonCV. "
                    f"Discrepancy {diff:+.3f} ({sigma:.2f}σ). "
                    "Likely causes: (a) transcription error, "
                    "(b) differential follow-up, (c) outcome-specific censoring."
                ),
            ))
    return violations
