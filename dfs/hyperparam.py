"""Type-II marginal-likelihood (ML-II) hyperparameter fit for ARD-Matérn GPs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from dfs.kernel import ard_matern_52


@dataclass
class FittedHyperparameters:
    sigma2: float
    length_scales: NDArray[np.float64]
    neg_log_marginal_likelihood: float


def _neg_log_marginal_likelihood(
    log_theta: NDArray[np.float64],
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    noise_var: NDArray[np.float64],
) -> float:
    """Negative log marginal likelihood under ARD-Matérn 5/2."""
    n_dim = x_train.shape[1]
    sigma2 = float(np.exp(log_theta[0]))
    length_scales = np.exp(log_theta[1:1 + n_dim])

    K = ard_matern_52(x_train, x_train, sigma2, length_scales)
    K = K + np.diag(noise_var) + 1e-8 * np.eye(K.shape[0])
    try:
        L_and_lower = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e12
    L = L_and_lower[0]
    alpha = cho_solve(L_and_lower, y_train)

    n = y_train.shape[0]
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    nll = 0.5 * float(y_train @ alpha) + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
    return nll


def fit_hyperparameters(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    noise_var: NDArray[np.float64],
    initial_sigma2: float = 1.0,
    initial_length_scales: Optional[NDArray[np.float64]] = None,
    n_restarts: int = 5,
    seed: int = 0,
) -> FittedHyperparameters:
    """Fit (sigma2, length_scales) by maximizing log marginal likelihood.

    Uses L-BFGS-B over log-parameters with multiple random restarts.

    Parameters
    ----------
    x_train:
        Training inputs, shape (n, d).
    y_train:
        Training targets, shape (n,).
    noise_var:
        Per-observation noise variances (fixed heteroscedastic), shape (n,).
    initial_sigma2:
        Starting point for signal variance.
    initial_length_scales:
        Starting point for ARD length-scales; defaults to 0.5 for each dim.
    n_restarts:
        Number of additional random restarts beyond the deterministic start.
    seed:
        RNG seed for reproducible restarts.

    Returns
    -------
    FittedHyperparameters
        Best-fit sigma2, length_scales, and the negative log marginal
        likelihood at the optimum.

    Raises
    ------
    RuntimeError
        If all restarts fail to converge.
    """
    n_dim = x_train.shape[1]
    if initial_length_scales is None:
        initial_length_scales = np.full(n_dim, 0.5)
    rng = np.random.default_rng(seed)

    best_result = None
    best_nll = np.inf
    starting_points = [
        np.concatenate([[np.log(initial_sigma2)], np.log(initial_length_scales)])
    ]
    for _ in range(n_restarts):
        x0 = np.concatenate([
            [np.log(initial_sigma2) + rng.normal(0, 1.0)],
            np.log(initial_length_scales) + rng.normal(0, 1.0, size=n_dim),
        ])
        starting_points.append(x0)

    for x0 in starting_points:
        try:
            result = minimize(
                _neg_log_marginal_likelihood,
                x0=x0,
                args=(x_train, y_train, noise_var),
                method="L-BFGS-B",
                bounds=[(-10, 10)] * (1 + n_dim),
            )
            if result.success and result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("ML-II optimisation failed across all restarts")

    sigma2 = float(np.exp(best_result.x[0]))
    length_scales = np.exp(best_result.x[1:1 + n_dim])
    return FittedHyperparameters(
        sigma2=sigma2,
        length_scales=length_scales,
        neg_log_marginal_likelihood=float(best_result.fun),
    )
