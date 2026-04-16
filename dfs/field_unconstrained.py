"""Unconstrained GP posterior (closed-form)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from dfs.kernel import KernelFn, ard_matern_52


@dataclass
class UnconstrainedGP:
    x_train: NDArray[np.float64]
    alpha: NDArray[np.float64]
    L_and_lower: tuple
    sigma2: float
    length_scales: NDArray[np.float64]
    kernel_fn: KernelFn = field(default=ard_matern_52)

    def predict(self, x_star: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        k_star = self.kernel_fn(
            x_star, self.x_train, self.sigma2, self.length_scales,
        )
        mu = k_star @ self.alpha
        v = cho_solve(self.L_and_lower, k_star.T)
        k_ss_diag = np.full(x_star.shape[0], self.sigma2)
        var = k_ss_diag - np.einsum("ij,ji->i", k_star, v)
        var = np.clip(var, 0.0, None)
        return mu, var


def fit_unconstrained_gp(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    noise_var: NDArray[np.float64],
    sigma2: float,
    length_scales: NDArray[np.float64],
    kernel_fn: KernelFn = ard_matern_52,
) -> UnconstrainedGP:
    K = kernel_fn(x_train, x_train, sigma2, length_scales)
    K = K + np.diag(noise_var)
    L_and_lower = cho_factor(K, lower=True)
    alpha = cho_solve(L_and_lower, y_train)
    return UnconstrainedGP(
        x_train=x_train, alpha=alpha, L_and_lower=L_and_lower,
        sigma2=sigma2, length_scales=length_scales,
        kernel_fn=kernel_fn,
    )
