"""Validation §8.1: unconstrained DFS with intercept-only covariate must match
inverse-variance pooling (fixed-effect MA limit) to MA_EQUIVALENCE_TOL."""

import numpy as np
import pytest

from dfs.config import MA_EQUIVALENCE_TOL
from dfs.field_unconstrained import fit_unconstrained_gp


def _inverse_variance_pool(y: np.ndarray, v: np.ndarray) -> float:
    """Standard fixed-effect inverse-variance pool."""
    w = 1.0 / v
    return float(np.sum(w * y) / np.sum(w))


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_ma_equivalence_intercept_only(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k = 8
    true_effect = -0.2
    y = true_effect + rng.standard_normal(k) * 0.05
    v = np.full(k, 0.05**2)

    # DFS setup: all anchors at origin (intercept-only) -> kernel at zero
    # distance = sigma2. With sigma2 >> noise_var, posterior reduces to
    # inverse-variance pooling on the intercept.
    x = np.zeros((k, 1))
    gp = fit_unconstrained_gp(
        x, y, v,
        sigma2=1.0,  # large prior variance so posterior is data-dominated
        length_scales=np.array([1.0]),
    )
    mu, _ = gp.predict(np.zeros((1, 1)))
    pooled = _inverse_variance_pool(y, v)
    assert abs(mu[0] - pooled) < MA_EQUIVALENCE_TOL, (
        f"DFS intercept-only posterior {mu[0]} vs MA pooled {pooled} "
        f"differ by more than {MA_EQUIVALENCE_TOL}"
    )
