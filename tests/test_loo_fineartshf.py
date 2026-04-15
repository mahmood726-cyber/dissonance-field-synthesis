from pathlib import Path

import numpy as np

from dfs.config import COVARIATE_NAMES
from dfs.manifest import load_trials
from dfs.field_constrained import fit_constrained_gp


MANIFEST_PATH = Path("data/mra_hfpef/MANIFEST.json")


def test_loo_fineartshf_runs_and_reports() -> None:
    """Hold out FINEARTS-HF; fit on 5 trials; predict its anchor; report."""
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

    gp = fit_constrained_gp(
        x_norm, y, noise,
        sigma2=0.25, length_scales=np.full(7, 0.5),
        inequality_constraints=[],
    )
    mu, var = gp.predict(anchor_norm)
    observed = held_out.outcomes["primary_composite"]["log_hr"]
    se_pred = float(np.sqrt(var[0]))
    lo = mu[0] - 1.96 * se_pred
    hi = mu[0] + 1.96 * se_pred

    inside = lo <= observed <= hi
    print(
        f"\nLOO FINEARTS-HF report:\n"
        f"  Predicted log-HR mean: {mu[0]:+.3f}\n"
        f"  Predicted 95% CrI:     [{lo:+.3f}, {hi:+.3f}]\n"
        f"  Observed log-HR:       {observed:+.3f}\n"
        f"  Inside CrI: {inside}\n"
    )
    # Test PASSES regardless of inside/outside; only pipeline errors fail it.
    assert np.isfinite(mu[0])
    assert np.isfinite(var[0])
