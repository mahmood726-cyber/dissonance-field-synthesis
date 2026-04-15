"""Constrained GP: virtual-observation projection via QP (CVXPY).

Reference: Da Veiga & Marrel (2020) – inequality constraints are enforced at
a discrete grid of virtual observation points by solving a QP that combines
the GP data-fit term with the GP prior regulariser.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from dfs.config import QP_SOLVER_ATOL
from dfs.kernel import ard_matern_52


class InequalityConstraint(TypedDict):
    matrix: NDArray[np.float64]
    bound: NDArray[np.float64]
    direction: Literal["geq", "leq"]
    grid: NDArray[np.float64]


@dataclass
class ConstrainedGP:
    x_train: NDArray[np.float64]
    y_posterior: NDArray[np.float64]
    sigma2: float
    length_scales: NDArray[np.float64]
    x_joint: NDArray[np.float64]

    def predict(
        self, x_star: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Posterior mean and variance at x_star.

        The QP solution y_posterior lives at x_joint.  We treat it as a
        noiseless "observation" and do standard GP conditioning.
        """
        k_star = ard_matern_52(
            x_star, self.x_joint, self.sigma2, self.length_scales
        )
        K_joint = ard_matern_52(
            self.x_joint, self.x_joint, self.sigma2, self.length_scales
        )
        K_joint = K_joint + 1e-8 * np.eye(K_joint.shape[0])
        alpha = np.linalg.solve(K_joint, self.y_posterior)
        mu = k_star @ alpha
        V = np.linalg.solve(K_joint, k_star.T)
        # σ² diagonal only valid for stationary kernel with zero prior mean;
        # update if a mean function is added (Task 5 review carry-over).
        var_prior = np.full(x_star.shape[0], self.sigma2)
        var = var_prior - np.einsum("ij,ji->i", k_star, V)
        return mu, np.clip(var, 0.0, None)


def fit_constrained_gp(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    noise_var: NDArray[np.float64],
    sigma2: float,
    length_scales: NDArray[np.float64],
    inequality_constraints: list[InequalityConstraint],
) -> ConstrainedGP:
    """Fit a GP with optional inequality constraints via CVXPY QP.

    When inequality_constraints is empty, the QP solution equals the
    standard GP posterior mean (up to the 1e-8 nugget) and predict()
    produces the same result as UnconstrainedGP.predict().

    Parameters
    ----------
    x_train:
        Training inputs, shape (n, d).
    y_train:
        Training targets, shape (n,).
    noise_var:
        Per-observation noise variances, shape (n,).
    sigma2:
        GP signal variance (prior amplitude²).
    length_scales:
        ARD length-scale per input dimension, shape (d,).
    inequality_constraints:
        List of InequalityConstraint dicts.  Each constraint applies
        ``matrix @ f[grid_indices] >= bound`` (or <= for "leq").
        All grid points must be contained in the union of supplied grids.

    Returns
    -------
    ConstrainedGP
        Fitted model.  Call .predict(x_star) for posterior mean and variance.

    Raises
    ------
    ValueError
        If a constraint grid point cannot be matched to the joint index set.
    RuntimeError
        If the QP solver returns a non-optimal status.
    """
    # --- Build x_joint = train points ∪ virtual constraint grid points ---
    if inequality_constraints:
        virtual_grids = [c["grid"] for c in inequality_constraints]
        # Deduplicate virtual points (preserves spatial coverage, removes
        # identical grid points shared across multiple constraints).
        x_virtual = np.unique(np.vstack(virtual_grids), axis=0)
    else:
        x_virtual = np.zeros((0, x_train.shape[1]))

    x_joint = np.vstack([x_train, x_virtual]) if x_virtual.size else x_train.copy()
    n_train = x_train.shape[0]
    n_joint = x_joint.shape[0]

    # --- Prior covariance on joint set (nugget for numerical stability) ---
    K = ard_matern_52(x_joint, x_joint, sigma2, length_scales)
    K_reg = K + 1e-8 * np.eye(n_joint)

    # --- QP decision variable: f = posterior mean at joint points ---
    f = cp.Variable(n_joint)

    # Data-fit term: sum_i (f_i - y_i)^2 / noise_i  (for train indices only)
    residuals = f[:n_train] - y_train
    data_term = cp.sum(cp.multiply(1.0 / noise_var, cp.square(residuals)))

    # GP prior regulariser: f^T K_reg^{-1} f = ||z||^2 where f = L z
    # Use Cholesky so CVXPY sees a sum-of-squares (convex QP form).
    L = np.linalg.cholesky(K_reg)
    z = cp.Variable(n_joint)
    prior_term = cp.sum_squares(z)

    # Enforce f = L z (links regulariser to decision variable)
    qp_constraints = [L @ z == f]

    # --- Inequality constraints via grid-point index lookup ---
    for ic in inequality_constraints:
        grid = ic["grid"]
        idx = []
        # Exact float match relies on x_virtual rows being the original grid
        # rows verbatim (np.unique preserves bit patterns). If a caller passes
        # grid points computed via arithmetic (e.g. x_base + 0.1), this may
        # silently ValueError. Harden to cdist-with-tolerance if needed.
        for g in grid:
            # Find row in x_joint that matches this grid point exactly.
            # np.unique may have changed order; row-wise equality check is safe
            # for float arrays built from np.linspace (exact bit patterns).
            matches = np.where(np.all(x_joint == g, axis=1))[0]
            if matches.size == 0:
                raise ValueError(
                    f"Constraint grid point {g} not found in x_joint. "
                    "Ensure all constraint grid points are within the "
                    "virtual grids passed to fit_constrained_gp."
                )
            idx.append(int(matches[0]))
        idx_arr = np.array(idx, dtype=int)

        M = ic["matrix"]
        b = ic["bound"]
        expr = M @ f[idx_arr]
        if ic["direction"] == "geq":
            qp_constraints.append(expr >= b)
        elif ic["direction"] == "leq":
            qp_constraints.append(expr <= b)
        else:
            raise ValueError(
                f"Unknown constraint direction: {ic['direction']!r}. "
                "Use 'geq' or 'leq'."
            )

    # --- Solve ---
    prob = cp.Problem(cp.Minimize(data_term + prior_term), qp_constraints)
    # CLARABEL uses tol_gap_abs/tol_gap_rel/tol_feas instead of OSQP's
    # eps_abs/eps_rel; map QP_SOLVER_ATOL to all three gap/feasibility knobs.
    prob.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=QP_SOLVER_ATOL,
        tol_gap_rel=QP_SOLVER_ATOL,
        tol_feas=QP_SOLVER_ATOL,
    )
    if prob.status not in (cp.OPTIMAL,):
        raise RuntimeError(
            f"QP solver returned non-optimal status: {prob.status!r}. "
            "Problem may be infeasible or numerically ill-conditioned."
        )

    y_posterior = np.asarray(f.value, dtype=np.float64)

    return ConstrainedGP(
        x_train=x_train,
        y_posterior=y_posterior,
        sigma2=sigma2,
        length_scales=length_scales,
        x_joint=x_joint,
    )
