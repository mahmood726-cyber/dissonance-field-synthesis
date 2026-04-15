"""Feasibility region: points where the CrI excludes a decision threshold."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def feasibility_region(
    posterior_mean: NDArray[np.float64],
    posterior_var: NDArray[np.float64],
    threshold: float,
    ci_level: float,
    direction: Literal["below", "above"],
) -> NDArray[np.bool_]:
    """Return bool array: True where CrI at ci_level entirely on the stated side of threshold."""
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0,1); got {ci_level}")
    alpha = 1.0 - ci_level
    z = norm.ppf(1.0 - alpha / 2.0)
    sd = np.sqrt(np.clip(posterior_var, 0.0, None))
    lo = posterior_mean - z * sd
    hi = posterior_mean + z * sd
    if direction == "below":
        return hi < threshold
    return lo > threshold
