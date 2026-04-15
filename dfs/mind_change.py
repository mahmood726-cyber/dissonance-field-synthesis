"""Mind-change price: how much new evidence to flip a recommendation.

Closed-form derivation in spec §7.3:
    n_eff = per_patient_var / posterior_var
    MCP = n_eff * (mu - t_cross) / (t_cross - t_obs)
"""
from __future__ import annotations

import math


def mind_change_price(
    posterior_mean: float,
    posterior_var: float,
    t_cross: float,
    t_obs: float,
    per_patient_var: float,
) -> float:
    """Smallest hypothetical-trial N that moves posterior mean past t_cross.

    Returns 0.0 if the update direction would push us FURTHER past t_cross
    (i.e. we're already past in the relevant direction).
    Returns +inf if t_obs == t_cross (update cannot cross a boundary when
    new observations sit on the boundary).
    """
    if per_patient_var <= 0.0 or posterior_var <= 0.0:
        raise ValueError("Variances must be positive")

    denom = t_cross - t_obs
    if denom == 0.0:
        return math.inf

    n_eff = per_patient_var / posterior_var

    # If the posterior mean has already crossed t_cross in the direction
    # opposite to t_obs (i.e. posterior_mean > t_cross while t_obs <
    # t_cross), the recommendation has already flipped — the "mind" is
    # already changed and no further evidence is required.
    if posterior_mean > t_cross and t_obs < t_cross:
        return 0.0

    numerator = posterior_mean - t_cross
    n = n_eff * numerator / denom
    return max(n, 0.0)
