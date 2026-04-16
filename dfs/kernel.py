"""ARD stationary kernels (Matérn-5/2 default; Matérn-3/2 and RBF for sensitivity)."""
from __future__ import annotations

import math
from typing import Callable
import numpy as np
from numpy.typing import NDArray


KernelFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]],
    NDArray[np.float64],
]


def _scaled_sq(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    length_scales: NDArray[np.float64],
) -> NDArray[np.float64]:
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
    return np.clip(sq, 0.0, None)


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
    sq = _scaled_sq(x, y, length_scales)
    r = np.sqrt(sq)
    sqrt5_r = math.sqrt(5.0) * r
    return sigma2 * (1.0 + sqrt5_r + (5.0 / 3.0) * sq) * np.exp(-sqrt5_r)


def ard_matern_32(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma2: float,
    length_scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Matérn ν=3/2 with per-dimension length-scales.

    k(x, y) = σ² * (1 + √3·r) * exp(-√3·r)
    Once mean-square differentiable — rougher surfaces than ν=5/2.
    """
    sq = _scaled_sq(x, y, length_scales)
    sqrt3_r = math.sqrt(3.0) * np.sqrt(sq)
    return sigma2 * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)


def ard_rbf(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma2: float,
    length_scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Squared-exponential (RBF) kernel — the ν→∞ limit of the Matérn family.

    k(x, y) = σ² * exp(-½ * sum((x_i - y_i)² / ℓ_i²))
    Infinitely differentiable; assumes very smooth outcome surfaces.
    """
    sq = _scaled_sq(x, y, length_scales)
    return sigma2 * np.exp(-0.5 * sq)


KERNELS: dict[str, KernelFn] = {
    "matern52": ard_matern_52,
    "matern32": ard_matern_32,
    "rbf": ard_rbf,
}
