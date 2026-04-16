"""Phase-2a: swap FIDELIO-DKD parent-trial anchor for HF-subgroup-specific HR.

Background
----------
The Phase-1b DFS analysis used the FIDELIO-DKD whole-trial CV composite HR
(0.86 [0.75-0.99]; log-HR = -0.151, SE = 0.072) as the FIDELIO-DKD
HF-subgroup primary anchor, because the AACT snapshot did not separately
report the HF-subgroup CV composite. The dedicated subgroup paper
(Filippatos et al. Eur J Heart Fail 2022;24:996-1005, PMID:35119760,
DOI:10.1002/ejhf.2469) reports for n=436 with history of HF at baseline:

  CV composite HR = 0.73 (95% CI 0.50-1.06)
  -> log-HR = -0.3147, SE = 0.1917

This script re-runs the LOO FINEARTS-HF pipeline with the FIDELIO anchor
swapped for the subgroup-specific value, holding all other trials at their
Phase-1b values, and reports the change in:

  - ML-II adherence_proxy length-scale
  - LOO FINEARTS-HF predicted log-HR mean / 95% CrI / inside-CrI bool
  - Pairwise dissonance scalar d for TOPCAT-Americas vs TOPCAT-Russia/Georgia

The FIGARO-DKD HF-subgroup CV composite HR is not separately reported in
the main text of the FIGARO HF-subgroup paper (PMID:34775784); it lives
in the supplement (Figure S3). Until that value is extracted, FIGARO is
held at its Phase-1b parent-trial anchor and this script demonstrates the
single-trial swap as a Phase-2a methodological proof-of-concept.

Usage
-----
  python scripts/phase2_subgroup_anchor_swap.py \
      --manifest data/mra_hfpef/MANIFEST.json \
      --results manuscript/phase2_subgroup_swap_results.csv
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dfs.config import COVARIATE_NAMES
from dfs.dissonance import pairwise_dissonance
from dfs.field_constrained import fit_constrained_gp
from dfs.hyperparam import fit_hyperparameters
from dfs.manifest import load_trials
from dfs.schema import TrialBoundaryCondition


# --- FIDELIO HF-subgroup anchor (Filippatos 2022 EJHF, PMID:35119760) ----

FIDELIO_HF_SUBGROUP_LOG_HR: float = -0.314711   # log(0.73)
FIDELIO_HF_SUBGROUP_SE: float = 0.191688         # (log(1.06) - log(0.50)) / 3.92
FIDELIO_HF_SUBGROUP_SOURCE: str = (
    "Filippatos 2022 Eur J Heart Fail 24:996-1005 (PMID:35119760, "
    "DOI:10.1002/ejhf.2469); n=436 with history of HF at baseline; "
    "CV composite HR 0.73, 95% CI 0.50-1.06"
)

# --- FIGARO HF-subgroup anchor (PROXY only — see Phase-2b notes) --------
# The dedicated FIGARO HF-subgroup CV composite HR is in supplementary
# Figure S3 of Filippatos 2022 Circulation (PMID:34775784) and was not
# extractable in this session (5 web sources tried). For Phase-2b, we use
# the FIDELIO HF-subgroup HR (0.73, SE inflated to reflect FIGARO's
# slightly larger n=571 vs 436 by sqrt(436/571) = 0.875) as a proxy and
# label the result as a hypothetical sensitivity test, not a true swap.

FIGARO_HF_SUBGROUP_PROXY_LOG_HR: float = -0.314711   # = FIDELIO HF subgroup
FIGARO_HF_SUBGROUP_PROXY_SE: float = 0.191688 * 0.875  # ~0.168 (n-scaled)
FIGARO_HF_SUBGROUP_PROXY_SOURCE: str = (
    "PROXY (Phase-2b sensitivity): FIDELIO HF-subgroup HR 0.73 used in "
    "place of FIGARO HF-subgroup CV composite HR (which is in supplement "
    "Figure S3 of Filippatos 2022 Circulation, PMID:34775784, not "
    "extractable in this session). SE scaled by sqrt(436/571)=0.875 to "
    "approximate FIGARO's larger HF-subgroup n. Result is a sensitivity "
    "test, NOT a citable FIGARO HF-subgroup-anchored fit."
)


CSV_FIELDNAMES: list[str] = [
    "scenario", "fidelio_log_hr", "fidelio_se",
    "figaro_log_hr", "figaro_se",
    "d_topcat", "adherence_ls", "loo_mu", "loo_lo", "loo_hi",
    "loo_width", "loo_inside",
]


def _swap_anchor(
    trials: list[TrialBoundaryCondition],
    trial_id: str,
    log_hr: float,
    se: float,
    note: str,
) -> list[TrialBoundaryCondition]:
    """Return a new trial list with the named trial's primary_composite replaced."""
    result = []
    for t in trials:
        if t.trial_id == trial_id:
            new_outcomes = dict(t.outcomes)
            new_primary = dict(new_outcomes["primary_composite"])
            new_primary["log_hr"] = log_hr
            new_primary["se"] = se
            new_primary["source"] = note
            new_outcomes["primary_composite"] = new_primary
            t = dataclasses.replace(t, outcomes=new_outcomes)
        result.append(t)
    return result


def _swap_fidelio_anchor(trials, log_hr, se, note):
    return _swap_anchor(trials, "FIDELIO-DKD-HF-subgroup", log_hr, se, note)


def _swap_figaro_anchor(trials, log_hr, se, note):
    return _swap_anchor(trials, "FIGARO-DKD-HF-subgroup", log_hr, se, note)


def _dissonance_topcat(trials: list[TrialBoundaryCondition]) -> float:
    pairs = pairwise_dissonance(trials, "primary_composite")
    for p in pairs:
        ids = set(p.trial_ids)
        if "TOPCAT-Americas" in ids and "TOPCAT-Russia-Georgia" in ids:
            return p.d
    raise KeyError("TOPCAT pair not found.")


def _run_loo(trials: list[TrialBoundaryCondition]) -> dict:
    held_out = next(t for t in trials if t.trial_id == "FINEARTS-HF")
    kept = [t for t in trials if t.trial_id != "FINEARTS-HF"]

    x = np.array([[t.anchor_covariates[c] for c in COVARIATE_NAMES] for t in kept])
    y = np.array([t.outcomes["primary_composite"]["log_hr"] for t in kept])
    noise = np.array([t.outcomes["primary_composite"]["se"] ** 2 for t in kept])

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    x_norm = (x - mins) / ranges

    anchor = np.array([held_out.anchor_covariates[c] for c in COVARIATE_NAMES])
    anchor_norm = ((anchor - mins) / ranges).reshape(1, -1)

    hp = fit_hyperparameters(x_norm, y, noise, n_restarts=5, seed=0)

    adh_idx = list(COVARIATE_NAMES).index("adherence_proxy")
    adherence_ls = float(hp.length_scales[adh_idx])

    gp = fit_constrained_gp(
        x_norm, y, noise,
        sigma2=hp.sigma2,
        length_scales=hp.length_scales,
        inequality_constraints=[],
    )
    mu, var = gp.predict(anchor_norm)

    observed = held_out.outcomes["primary_composite"]["log_hr"]
    se_observed = held_out.outcomes["primary_composite"]["se"]
    var_pred = float(var[0]) + float(se_observed ** 2)
    se_pred = float(np.sqrt(max(var_pred, 0.0)))
    lo = float(mu[0]) - 1.96 * se_pred
    hi = float(mu[0]) + 1.96 * se_pred

    return {
        "adherence_ls": adherence_ls,
        "loo_mu": float(mu[0]),
        "loo_lo": lo,
        "loo_hi": hi,
        "loo_width": hi - lo,
        "loo_inside": bool(lo <= observed <= hi),
    }


def _trial_anchor(
    trials: list[TrialBoundaryCondition], trial_id: str,
) -> tuple[float, float]:
    f = next(t for t in trials if t.trial_id == trial_id)
    return (
        float(f.outcomes["primary_composite"]["log_hr"]),
        float(f.outcomes["primary_composite"]["se"]),
    )


def _fidelio_anchor(trials):
    return _trial_anchor(trials, "FIDELIO-DKD-HF-subgroup")


def _figaro_anchor(trials):
    return _trial_anchor(trials, "FIGARO-DKD-HF-subgroup")


def run(base_trials: list[TrialBoundaryCondition]) -> list[dict]:
    """Run baseline + swapped scenarios; return rows for CSV."""
    rows: list[dict] = []

    # --- Scenario 1: baseline (parent-trial anchor) ---
    print("\n=== Scenario 1: baseline (FIDELIO parent-trial anchor) ===")
    baseline_log_hr, baseline_se = _fidelio_anchor(base_trials)
    print(f"  FIDELIO log-HR = {baseline_log_hr:+.4f}, SE = {baseline_se:.4f}")
    d_baseline = _dissonance_topcat(base_trials)
    loo_baseline = _run_loo(base_trials)
    figaro_baseline_log_hr, figaro_baseline_se = _figaro_anchor(base_trials)
    rows.append({
        "scenario": "baseline_parent_trial_anchor",
        "fidelio_log_hr": baseline_log_hr,
        "fidelio_se": baseline_se,
        "figaro_log_hr": figaro_baseline_log_hr,
        "figaro_se": figaro_baseline_se,
        "d_topcat": d_baseline,
        **loo_baseline,
    })
    print(f"  d_topcat = {d_baseline:.3f}, adh_ls = {loo_baseline['adherence_ls']:.4f}, "
          f"LOO mu = {loo_baseline['loo_mu']:+.4f} "
          f"[{loo_baseline['loo_lo']:+.4f}, {loo_baseline['loo_hi']:+.4f}], "
          f"inside = {loo_baseline['loo_inside']}")

    # --- Scenario 2: FIDELIO HF-subgroup anchor swapped ---
    print("\n=== Scenario 2: FIDELIO HF-subgroup-specific anchor swap ===")
    print(f"  Replacing FIDELIO primary_composite with HF-subgroup HR 0.73")
    print(f"  Source: {FIDELIO_HF_SUBGROUP_SOURCE}")
    print(f"  New log-HR = {FIDELIO_HF_SUBGROUP_LOG_HR:+.4f}, "
          f"SE = {FIDELIO_HF_SUBGROUP_SE:.4f}")
    swapped = _swap_fidelio_anchor(
        base_trials,
        FIDELIO_HF_SUBGROUP_LOG_HR,
        FIDELIO_HF_SUBGROUP_SE,
        FIDELIO_HF_SUBGROUP_SOURCE,
    )
    d_swap = _dissonance_topcat(swapped)
    loo_swap = _run_loo(swapped)
    rows.append({
        "scenario": "fidelio_hf_subgroup_anchor",
        "fidelio_log_hr": FIDELIO_HF_SUBGROUP_LOG_HR,
        "fidelio_se": FIDELIO_HF_SUBGROUP_SE,
        "figaro_log_hr": figaro_baseline_log_hr,
        "figaro_se": figaro_baseline_se,
        "d_topcat": d_swap,
        **loo_swap,
    })

    # --- Scenario 3 (Phase-2b): both FIDELIO and FIGARO swapped (PROXY) ---
    print("\n=== Scenario 3 (Phase-2b): FIDELIO + FIGARO HF-subgroup proxy swap ===")
    print(f"  FIGARO swap is HYPOTHETICAL — uses FIDELIO HF-subgroup HR (0.73)")
    print(f"  Source: {FIGARO_HF_SUBGROUP_PROXY_SOURCE}")
    swapped_both = _swap_figaro_anchor(
        swapped,
        FIGARO_HF_SUBGROUP_PROXY_LOG_HR,
        FIGARO_HF_SUBGROUP_PROXY_SE,
        FIGARO_HF_SUBGROUP_PROXY_SOURCE,
    )
    d_both = _dissonance_topcat(swapped_both)
    loo_both = _run_loo(swapped_both)
    rows.append({
        "scenario": "phase2b_proxy_both_swapped",
        "fidelio_log_hr": FIDELIO_HF_SUBGROUP_LOG_HR,
        "fidelio_se": FIDELIO_HF_SUBGROUP_SE,
        "figaro_log_hr": FIGARO_HF_SUBGROUP_PROXY_LOG_HR,
        "figaro_se": FIGARO_HF_SUBGROUP_PROXY_SE,
        "d_topcat": d_both,
        **loo_both,
    })
    print(f"  d_topcat = {d_both:.3f}, adh_ls = {loo_both['adherence_ls']:.4f}, "
          f"LOO mu = {loo_both['loo_mu']:+.4f} "
          f"[{loo_both['loo_lo']:+.4f}, {loo_both['loo_hi']:+.4f}], "
          f"inside = {loo_both['loo_inside']}")
    print(f"  d_topcat = {d_swap:.3f}, adh_ls = {loo_swap['adherence_ls']:.4f}, "
          f"LOO mu = {loo_swap['loo_mu']:+.4f} "
          f"[{loo_swap['loo_lo']:+.4f}, {loo_swap['loo_hi']:+.4f}], "
          f"inside = {loo_swap['loo_inside']}")

    # --- Comparison ---
    print("\n--- Phase-2a comparison ---")
    print(f"FIDELIO anchor shift:        log-HR {baseline_log_hr:+.4f} -> "
          f"{FIDELIO_HF_SUBGROUP_LOG_HR:+.4f} (delta = "
          f"{FIDELIO_HF_SUBGROUP_LOG_HR - baseline_log_hr:+.4f})")
    print(f"FIDELIO anchor SE inflation: SE     {baseline_se:.4f} -> "
          f"{FIDELIO_HF_SUBGROUP_SE:.4f} ({FIDELIO_HF_SUBGROUP_SE / baseline_se:.2f}x)")
    print(f"Adherence length-scale:      {loo_baseline['adherence_ls']:.4f} -> "
          f"{loo_swap['adherence_ls']:.4f} "
          f"(delta = {loo_swap['adherence_ls'] - loo_baseline['adherence_ls']:+.4f})")
    print(f"LOO predicted mean:          {loo_baseline['loo_mu']:+.4f} -> "
          f"{loo_swap['loo_mu']:+.4f} "
          f"(delta = {loo_swap['loo_mu'] - loo_baseline['loo_mu']:+.4f})")
    print(f"LOO 95% CrI width:           {loo_baseline['loo_width']:.4f} -> "
          f"{loo_swap['loo_width']:.4f} "
          f"(delta = {loo_swap['loo_width'] - loo_baseline['loo_width']:+.4f})")
    print(f"LOO inside CrI:              {loo_baseline['loo_inside']} -> {loo_swap['loo_inside']}")
    print(f"TOPCAT pair dissonance d:    {d_baseline:.3f} -> {d_swap:.3f} "
          f"(invariant: depends only on TOPCAT pair, not FIDELIO)")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            out = {}
            for k in CSV_FIELDNAMES:
                val = row[k]
                if isinstance(val, float):
                    out[k] = f"{val:.6f}" if math.isfinite(val) else str(val)
                else:
                    out[k] = val
            writer.writerow(out)
    print(f"\nResults written to: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-2a FIDELIO HF-subgroup anchor swap.")
    p.add_argument("--manifest", type=Path,
                   default=Path("data/mra_hfpef/MANIFEST.json"))
    p.add_argument("--results", type=Path,
                   default=Path("manuscript/phase2_subgroup_swap_results.csv"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Loading manifest: {args.manifest}")
    trials = load_trials(args.manifest)
    print(f"  Loaded {len(trials)} trials: {[t.trial_id for t in trials]}")
    rows = run(trials)
    write_csv(rows, args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
