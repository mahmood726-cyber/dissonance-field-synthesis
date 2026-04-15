"""End-to-end DFS pipeline on the manifest-defined trial set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dfs.config import COVARIATE_NAMES
from dfs.decisions import DECISION_THRESHOLDS, PER_PATIENT_VAR
from dfs.diagnostics import detect_conservation_violations
from dfs.dissonance import pairwise_dissonance
from dfs.feasibility import feasibility_region
from dfs.field_constrained import fit_constrained_gp
from dfs.manifest import load_trials
from dfs.mind_change import mind_change_price
from dfs.outputs import plot_field_slice, write_dissonance_table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--outcome", default="primary_composite")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    trials = load_trials(args.manifest)
    outcome = args.outcome

    # 1. Dissonance
    pairs = pairwise_dissonance(trials, outcome=outcome)
    write_dissonance_table(pairs, args.out / "dissonance.csv")

    # 2. Conservation diagnostics
    violations = detect_conservation_violations(trials)
    (args.out / "conservation_diagnostics.json").write_text(
        json.dumps(
            [{"trial_id": v.trial_id, "law": v.law_name,
              "sigma": v.sigma_magnitude, "detail": v.detail} for v in violations],
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3. Effect field (unconstrained for POC; constrained extension phase-1b)
    x = np.array([[t.anchor_covariates[c] for c in COVARIATE_NAMES] for t in trials])
    y = np.array([t.outcomes[outcome]["log_hr"] for t in trials])
    noise = np.array([t.outcomes[outcome]["se"] ** 2 for t in trials])

    mins, maxs = x.min(axis=0), x.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    x_norm = (x - mins) / ranges

    gp = fit_constrained_gp(
        x_norm, y, noise,
        sigma2=0.25, length_scales=np.full(7, 0.5),
        inequality_constraints=[],
    )

    # 4. Field slice: LVEF × eGFR at cohort median of other covariates
    lvef_ax = np.linspace(0, 1, 40)
    egfr_ax = np.linspace(0, 1, 40)
    LV, EG = np.meshgrid(lvef_ax, egfr_ax, indexing="xy")
    fixed = np.median(x_norm, axis=0)
    grid = np.tile(fixed, (LV.size, 1))
    grid[:, COVARIATE_NAMES.index("lvef")] = LV.ravel()
    grid[:, COVARIATE_NAMES.index("egfr")] = EG.ravel()
    mu, var = gp.predict(grid)
    mu_img = mu.reshape(LV.shape)
    var_img = var.reshape(LV.shape)
    plot_field_slice(
        lvef_ax, egfr_ax, mu_img, var_img,
        args.out / "field_lvef_egfr.png",
        x_label="LVEF (norm)", y_label="eGFR (norm)",
    )

    # 5. Mind-change price at the same slice
    thr = DECISION_THRESHOLDS[outcome]["recommend_if_log_hr_below"]
    per_pt = PER_PATIENT_VAR[outcome]
    mcp = np.array([
        mind_change_price(
            posterior_mean=float(m), posterior_var=float(v),
            t_cross=thr, t_obs=0.0, per_patient_var=per_pt,
        ) for m, v in zip(mu, var)
    ])
    mcp_img = mcp.reshape(LV.shape)
    with (args.out / "mind_change_price.csv").open("w", encoding="utf-8") as f:
        f.write("lvef_norm,egfr_norm,mind_change_price\n")
        for (i, j), v in np.ndenumerate(mcp_img):
            f.write(f"{lvef_ax[j]:.3f},{egfr_ax[i]:.3f},{v:.3f}\n")

    # 6. Feasibility mask
    mask = feasibility_region(
        mu, var, threshold=thr, ci_level=0.95, direction="below",
    ).reshape(LV.shape).astype(int)
    with (args.out / "feasibility_mask.csv").open("w", encoding="utf-8") as f:
        f.write("lvef_norm,egfr_norm,in_region\n")
        for (i, j), v in np.ndenumerate(mask):
            f.write(f"{lvef_ax[j]:.3f},{egfr_ax[i]:.3f},{v}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
