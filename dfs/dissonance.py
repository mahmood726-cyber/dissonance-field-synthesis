"""Pairwise dissonance extraction across trial boundary-condition records."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from dfs.schema import TrialBoundaryCondition


@dataclass(frozen=True)
class DissonancePair:
    """Pairwise dissonance between two trials.

    Convention: `log_hr_delta` and `covariate_delta` entries are
    computed as (trial_b - trial_a) where (trial_a, trial_b) is the
    pair tuple produced by itertools.combinations. The scalar `d`
    uses abs() so is direction-invariant.
    """

    trial_ids: tuple[str, str]
    outcome: str
    d: float
    log_hr_delta: float
    covariate_delta: dict[str, float]


def pairwise_dissonance(
    trials: list[TrialBoundaryCondition],
    outcome: str,
) -> list[DissonancePair]:
    pairs: list[DissonancePair] = []
    for a, b in itertools.combinations(trials, 2):
        if outcome not in a.outcomes or outcome not in b.outcomes:
            raise KeyError(
                f"Outcome {outcome!r} missing from trial "
                f"{a.trial_id!r} or {b.trial_id!r}"
            )
        la = a.outcomes[outcome]["log_hr"]
        lb = b.outcomes[outcome]["log_hr"]
        sa = a.outcomes[outcome]["se"]
        sb = b.outcomes[outcome]["se"]
        denom = math.sqrt(sa**2 + sb**2)
        if denom < 1e-12:
            raise ZeroDivisionError(
                f"Near-zero combined SE for {outcome!r} "
                f"({a.trial_id!r} SE={sa}, {b.trial_id!r} SE={sb}); "
                "cannot compute dissonance"
            )
        d = abs(la - lb) / denom
        if a.anchor_covariates.keys() != b.anchor_covariates.keys():
            raise KeyError(
                f"Covariate key mismatch between {a.trial_id!r} and {b.trial_id!r}: "
                f"{sorted(set(a.anchor_covariates) ^ set(b.anchor_covariates))}"
            )
        cov_delta = {
            k: b.anchor_covariates[k] - a.anchor_covariates[k]
            for k in a.anchor_covariates
        }
        pairs.append(DissonancePair(
            trial_ids=(a.trial_id, b.trial_id),
            outcome=outcome,
            d=d,
            log_hr_delta=lb - la,
            covariate_delta=cov_delta,
        ))
    return pairs
