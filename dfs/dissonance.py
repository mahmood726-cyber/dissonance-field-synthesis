"""Pairwise dissonance extraction across trial boundary-condition records."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from dfs.schema import TrialBoundaryCondition


@dataclass(frozen=True)
class DissonancePair:
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
        if denom == 0.0:
            raise ZeroDivisionError(
                f"Both trials have zero SE for {outcome!r}; cannot compute dissonance"
            )
        d = abs(la - lb) / denom
        cov_delta = {
            k: b.anchor_covariates[k] - a.anchor_covariates[k]
            for k in a.anchor_covariates
        }
        pairs.append(DissonancePair(
            trial_ids=(a.trial_id, b.trial_id),
            outcome=outcome,
            d=d,
            log_hr_delta=la - lb,
            covariate_delta=cov_delta,
        ))
    return pairs
