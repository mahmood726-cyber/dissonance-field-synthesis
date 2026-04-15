"""Trial -> scalar adherence proxy in [0, 1].

USER-AUTHORED: the cardiologist encodes the clinical mapping from
trial design features to a single adherence score. This function
is load-bearing for splitting TOPCAT into two distinct boundary
conditions (spec §4 and §11).

Features you may use (from TrialBoundaryCondition.design_priors):
  - placebo_rate_per_yr : observed placebo-arm primary event rate
  - ltfu_fraction       : loss-to-followup fraction
  - adherence_proxy     : trial-reported protocol adherence, if any

Signals to consider:
  - very-low placebo event rate in a supposedly-sick population
    suggests non-adherence or misdiagnosis (TOPCAT-Russia signal)
  - high ltfu => lower adherence
  - reported adherence %, if present, is the simplest anchor
"""
from __future__ import annotations

from dfs.schema import TrialBoundaryCondition


def adherence_proxy(trial: TrialBoundaryCondition) -> float:
    """Return a score in [0, 1]; higher = more confident in adherence.

    USER: implement your clinical mapping below.
    The default implementation below is a placeholder that satisfies
    the test contract but is NOT calibrated — you must revise it.
    """
    priors = trial.design_priors
    reported = priors.get("adherence_proxy", 0.8)

    ltfu = priors.get("ltfu_fraction", 0.1)
    ltfu_penalty = max(0.0, 1.0 - 2.0 * ltfu)

    placebo_rate = priors.get("placebo_rate_per_yr", 0.1)
    rate_score = 1.0 if placebo_rate >= 0.05 else placebo_rate / 0.05

    combined = 0.5 * reported + 0.3 * ltfu_penalty + 0.2 * rate_score
    return max(0.0, min(1.0, combined))
