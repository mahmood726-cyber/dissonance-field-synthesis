from pathlib import Path

import numpy as np

from dfs.config import COVARIATE_NAMES
from dfs.manifest import load_trials
from dfs.field_constrained import fit_constrained_gp
from dfs.hyperparam import fit_hyperparameters


MANIFEST_PATH = Path("data/mra_hfpef/MANIFEST.json")


def test_loo_fineartshf_runs_and_reports() -> None:
    """Hold out FINEARTS-HF; fit hyperparams by ML-II; predict its anchor; report."""
    trials = load_trials(MANIFEST_PATH)
    held_out = next(t for t in trials if t.trial_id == "FINEARTS-HF")
    kept = [t for t in trials if t.trial_id != "FINEARTS-HF"]

    x = np.array([
        [t.anchor_covariates[c] for c in COVARIATE_NAMES] for t in kept
    ])
    y = np.array([t.outcomes["primary_composite"]["log_hr"] for t in kept])
    noise = np.array([t.outcomes["primary_composite"]["se"] ** 2 for t in kept])

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    x_norm = (x - mins) / ranges
    anchor = np.array([held_out.anchor_covariates[c] for c in COVARIATE_NAMES])
    anchor_norm = ((anchor - mins) / ranges).reshape(1, -1)

    # --- ML-II hyperparameter fit ---
    hp = fit_hyperparameters(x_norm, y, noise, n_restarts=5, seed=42)
    assert np.isfinite(hp.sigma2), "sigma2 must be finite after ML-II fit"
    assert np.all(np.isfinite(hp.length_scales)), "all length_scales must be finite"
    print(
        f"\nML-II fitted hyperparameters:\n"
        f"  sigma2={hp.sigma2:.4f}\n"
        f"  length_scales={np.array2string(hp.length_scales, precision=4)}\n"
        f"  neg_log_mll={hp.neg_log_marginal_likelihood:.4f}\n"
    )

    gp = fit_constrained_gp(
        x_norm, y, noise,
        sigma2=hp.sigma2, length_scales=hp.length_scales,
        inequality_constraints=[],
    )
    mu, var = gp.predict(anchor_norm)
    observed = held_out.outcomes["primary_composite"]["log_hr"]
    se_observed = held_out.outcomes["primary_composite"]["se"]

    # LOO predictive variance: GP posterior variance + held-out observation noise.
    # This is the proper leave-one-out predictive interval: we are predicting
    # a *noisy* observation, so we must add the noise variance of the new point.
    var_pred = float(var[0]) + float(se_observed ** 2)
    se_pred = float(np.sqrt(var_pred))
    lo = mu[0] - 1.96 * se_pred
    hi = mu[0] + 1.96 * se_pred
    width = hi - lo

    inside = lo <= observed <= hi
    print(
        f"\nLOO FINEARTS-HF report (ML-II fitted hyperparams):\n"
        f"  Predicted log-HR mean:     {mu[0]:+.3f}\n"
        f"  GP posterior std:          {float(np.sqrt(var[0])):.4f}\n"
        f"  Obs noise (se_held_out):   {se_observed:.4f}\n"
        f"  Predictive std (combined): {se_pred:.4f}\n"
        f"  Predicted 95% CrI:        [{lo:+.3f}, {hi:+.3f}]\n"
        f"  CrI width:                 {width:.4f} log-HR units\n"
        f"  Observed log-HR:           {observed:+.3f}\n"
        f"  Inside CrI: {inside}\n"
    )

    assert np.isfinite(mu[0])
    assert np.isfinite(var[0])
    # CrI must tighten relative to hard-coded baseline (~1.96 units)
    assert width <= 1.5, (
        f"CrI width {width:.4f} exceeds 1.5 log-HR units — ML-II fit did not tighten"
    )
    # Observed value must remain covered by the LOO predictive interval
    assert inside, (
        f"Observed log-HR {observed:+.3f} is outside LOO predictive CrI [{lo:+.3f}, {hi:+.3f}]"
    )
