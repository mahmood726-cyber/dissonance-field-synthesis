"""Safety-field fitter: constrained GP for per-trial safety outcomes.

Currently wired for the k_sign conservation law (ΔK⁺ >= 0 wherever
MR occupancy > 0).  The other five laws are deferred to Phase-2; see
docs/conservation_law_wiring_status.md for architectural blockers.

Usage
-----
from dfs.safety import fit_safety_field

result = fit_safety_field(trials, safety_key="delta_k", constraint="k_sign")
gp       = result["gp"]           # ConstrainedGP
report   = result["report"]       # dict with constraint details
vgrid    = result["virtual_grid"] # (50, d) normalised covariate array
mu, var  = gp.predict(vgrid)
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats.qmc import LatinHypercube

from dfs.config import COVARIATE_NAMES
from dfs.field_constrained import InequalityConstraint, fit_constrained_gp
from dfs.hyperparam import fit_hyperparameters
from dfs.schema import TrialBoundaryCondition

# ------------------------------------------------------------------ constants

# Number of virtual observation points for the constraint grid.
# 50 is enough for 7-D with LatinHypercube (per deviation log).
N_VIRTUAL = 50

# How close (in natural units) a posterior mean must be to 0 for the
# constraint to be considered "binding".
BINDING_THRESHOLD = 0.05

# SE inflation factor applied to safety values whose source string indicates
# a "Derived" value (increased uncertainty about indirect estimates).
DERIVED_SE_INFLATION = 2.0


# ------------------------------------------------------------------ helpers

def _is_derived(source: str) -> bool:
    """Return True if a source string starts with 'Derived' (case-insensitive)."""
    return source.strip().lower().startswith("derived")


def _build_training_data(
    trials: list[TrialBoundaryCondition],
    safety_key: str,
) -> tuple[
    NDArray[np.float64],  # x_raw  (n_included, d)
    NDArray[np.float64],  # y      (n_included,)
    NDArray[np.float64],  # noise  (n_included,)
    list[dict[str, Any]], # per_trial report rows
    int,                  # n_skipped
]:
    """Extract (x, y, noise_var) from trials for a given safety key.

    Trials missing the safety key or with a None value are skipped with a
    note in the per_trial report.  Trials marked "Derived" receive a 2×
    SE inflation before variance is computed.

    Parameters
    ----------
    trials:
        Full list of TrialBoundaryCondition records.
    safety_key:
        Key inside trial.safety, e.g. "delta_k".

    Returns
    -------
    x_raw:
        Covariate matrix (un-normalised) for included trials.
    y:
        Safety values.
    noise_var:
        Per-observation noise variances (= (inflated SE)²).
    per_trial:
        One dict per trial with inclusion/exclusion details and the
        raw observed value (for the constraint report).
    n_skipped:
        Count of skipped (missing-key) trials.
    """
    x_list: list[list[float]] = []
    y_list: list[float] = []
    noise_list: list[float] = []
    per_trial: list[dict[str, Any]] = []
    n_skipped = 0

    for t in trials:
        entry = t.safety.get(safety_key)
        if entry is None:
            per_trial.append({
                "trial_id": t.trial_id,
                "included": False,
                "skip_reason": f"safety key '{safety_key}' absent",
                "observed_delta_k": None,
            })
            n_skipped += 1
            continue

        value = entry.get("value")
        se = entry.get("se")
        source = entry.get("source", "")

        if value is None or se is None:
            per_trial.append({
                "trial_id": t.trial_id,
                "included": False,
                "skip_reason": "value or se is None",
                "observed_delta_k": value,
            })
            n_skipped += 1
            continue

        # Apply SE inflation for derived values
        se_eff = se * DERIVED_SE_INFLATION if _is_derived(source) else se
        # Guard against zero/negative SE
        se_eff = max(se_eff, 1e-6)

        x_list.append([t.anchor_covariates[c] for c in COVARIATE_NAMES])
        y_list.append(float(value))
        noise_list.append(float(se_eff) ** 2)

        per_trial.append({
            "trial_id": t.trial_id,
            "included": True,
            "skip_reason": None,
            "observed_delta_k": float(value),
            "se_used": float(se_eff),
            "derived": _is_derived(source),
            "satisfies_k_sign": float(value) >= 0.0,
        })

    x_raw = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    noise_var = np.array(noise_list, dtype=np.float64)
    return x_raw, y, noise_var, per_trial, n_skipped


def _build_virtual_grid_mr_active(n_points: int, seed: int = 42) -> NDArray[np.float64]:
    """Sample a Latin-hypercube virtual grid in normalised [0,1]^d covariate space.

    The k_sign law applies wherever MR occupancy > 0.  We sample from [0,1]^d
    (the full normalised covariate space) but clamp the mr_occupancy dimension
    to [0.1, 1.0] in normalised coordinates so that all virtual points satisfy
    "MR occupancy > 0" per the conservation-law spec.

    Parameters
    ----------
    n_points:
        Number of virtual observation points.
    seed:
        RNG seed for reproducible grids.

    Returns
    -------
    NDArray of shape (n_points, len(COVARIATE_NAMES)) in [0,1]^d normalised space.
    """
    d = len(COVARIATE_NAMES)
    sampler = LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n_points)  # shape (n_points, d)

    mr_idx = COVARIATE_NAMES.index("mr_occupancy")
    # Remap the MR-occupancy dimension from [0,1] to [0.1, 1.0] so every
    # virtual point is in the "MR occupancy > 0" region.
    sample[:, mr_idx] = 0.1 + sample[:, mr_idx] * 0.9

    return sample.astype(np.float64)


# ------------------------------------------------------------------ public API

def fit_safety_field(
    trials: list[TrialBoundaryCondition],
    safety_key: str = "delta_k",
    constraint: Literal["k_sign"] = "k_sign",
    n_virtual: int = N_VIRTUAL,
    seed: int = 42,
    n_restarts: int = 5,
) -> dict[str, Any]:
    """Fit a constrained GP on a trial-level safety outcome surface.

    Currently only the ``k_sign`` constraint (``g_K(x) >= 0``) is supported,
    matching the single-output value-inequality API of ``fit_constrained_gp``.

    Parameters
    ----------
    trials:
        List of TrialBoundaryCondition records loaded from the manifest.
    safety_key:
        Key inside ``trial.safety`` to use as observations, e.g. ``"delta_k"``.
    constraint:
        Name of the conservation law to enforce; currently only ``"k_sign"``.
    n_virtual:
        Number of Latin-hypercube virtual observation points for the constraint.
    seed:
        RNG seed forwarded to both the LHC sampler and ``fit_hyperparameters``.
    n_restarts:
        Number of ML-II restart attempts.

    Returns
    -------
    dict with keys:
        ``"gp"``           — fitted :class:`~dfs.field_constrained.ConstrainedGP`
        ``"virtual_grid"`` — normalised covariate array (n_virtual, d)
        ``"report"``       — constraint details (binding status, per-trial check)

    Raises
    ------
    ValueError
        If fewer than 2 trials contain the requested safety key (GP is
        ill-defined with a single point).
    ValueError
        If ``constraint`` is not ``"k_sign"``.
    """
    if constraint != "k_sign":
        raise ValueError(
            f"Unsupported constraint: {constraint!r}.  "
            "Only 'k_sign' is wired in Phase-1b.  "
            "See docs/conservation_law_wiring_status.md for Phase-2 blockers."
        )

    # ---------------------------------------------------------------- data
    x_raw, y, noise_var, per_trial, n_skipped = _build_training_data(trials, safety_key)

    n_included = x_raw.shape[0]
    if n_included < 2:
        raise ValueError(
            f"Only {n_included} trial(s) have safety key '{safety_key}'. "
            "Need at least 2 for a well-defined GP fit.  "
            "Check that trials include the safety entry."
        )

    # ------------------------------------------ normalise to [0,1]^d
    mins = x_raw.min(axis=0)
    maxs = x_raw.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    x_norm = (x_raw - mins) / ranges

    # ------------------------------------------ ML-II hyperparameter fit
    hp = fit_hyperparameters(
        x_norm, y, noise_var,
        n_restarts=n_restarts,
        seed=seed,
    )

    # ------------------------------------------ virtual constraint grid
    virtual_grid = _build_virtual_grid_mr_active(n_virtual, seed=seed)

    # ------------------------------------------ build inequality constraint
    # k_sign: g_K(x) >= 0 for all x with MR occupancy > 0
    # Encoded as: I · f[grid] >= 0
    ineq: InequalityConstraint = {
        "matrix": np.eye(n_virtual),
        "bound": np.zeros(n_virtual),
        "direction": "geq",
        "grid": virtual_grid,
    }

    # ------------------------------------------ fit constrained GP
    gp = fit_constrained_gp(
        x_norm, y, noise_var,
        sigma2=hp.sigma2,
        length_scales=hp.length_scales,
        inequality_constraints=[ineq],
    )

    # ------------------------------------------ report
    mu_virtual, _ = gp.predict(virtual_grid)
    min_posterior = float(np.min(mu_virtual))

    # "Binding" means the constraint was *active* — i.e. the QP was forced to
    # pull the posterior up from a negative unconstrained value.  We detect
    # this by fitting an *unconstrained* GP with the same hyperparameters and
    # checking where the unconstrained posterior would have been negative.
    gp_unconstrained = fit_constrained_gp(
        x_norm, y, noise_var,
        sigma2=hp.sigma2,
        length_scales=hp.length_scales,
        inequality_constraints=[],
    )
    mu_unc, _ = gp_unconstrained.predict(virtual_grid)
    # The constraint is binding at points where the unconstrained posterior
    # was below the boundary (< 0) — the QP had to enforce the floor there.
    binding_mask = mu_unc < 0.0
    any_binding = bool(np.any(binding_mask))
    n_binding = int(np.sum(binding_mask))

    report: dict[str, Any] = {
        "constraint": "k_sign",
        "safety_key": safety_key,
        "n_virtual_obs": n_virtual,
        "any_binding": any_binding,
        "n_binding_virtual_obs": n_binding,
        "min_posterior_mean_on_grid": min_posterior,
        "n_included_trials": n_included,
        "n_skipped_trials": n_skipped,
        "per_trial": per_trial,
        "hyperparameters": {
            "sigma2": float(hp.sigma2),
            "length_scales": hp.length_scales.tolist(),
            "neg_log_mll": float(hp.neg_log_marginal_likelihood),
        },
    }

    return {
        "gp": gp,
        "virtual_grid": virtual_grid,
        "report": report,
    }
