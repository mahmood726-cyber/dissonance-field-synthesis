"""ARD Matérn 5/2 kernel."""
from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray


def ard_matern_52(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma2: float,
    length_scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Matérn ν=5/2 with per-dimension length-scales.

    k(x, y) = σ² * (1 + √5·r + (5/3)·r²) * exp(-√5·r)
    where r = sqrt(sum((x_i - y_i)² / ℓ_i²)).
    """
    if x.shape[1] != y.shape[1] or x.shape[1] != len(length_scales):
        raise ValueError(
            f"Dimension mismatch: x={x.shape}, y={y.shape}, "
            f"length_scales={length_scales.shape}"
        )
    scaled_x = x / length_scales[np.newaxis, :]
    scaled_y = y / length_scales[np.newaxis, :]
    sq = (
        np.sum(scaled_x**2, axis=1)[:, np.newaxis]
        + np.sum(scaled_y**2, axis=1)[np.newaxis, :]
        - 2.0 * scaled_x @ scaled_y.T
    )
    sq = np.clip(sq, 0.0, None)
    r = np.sqrt(sq)
    sqrt5_r = math.sqrt(5.0) * r
    return sigma2 * (1.0 + sqrt5_r + (5.0 / 3.0) * sq) * np.exp(-sqrt5_r)
