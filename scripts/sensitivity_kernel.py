"""Kernel-choice sensitivity analysis for the DFS POC.

Re-fits the leave-one-out FINEARTS-HF GP under three ARD stationary kernels:

  - Matern-5/2  (nu=2.5, current default; twice mean-square differentiable)
  - Matern-3/2  (nu=1.5, rougher;      once  mean-square differentiable)
  - RBF / squared-exponential (nu->inf; infinitely differentiable)

For each kernel, reports:
  - ML-II adherence_proxy length-scale
  - ML-II negative log marginal likelihood (lower = better fit)
  - LOO FINEARTS-HF: predicted log-HR mean, 95% CrI, inside-CrI bool
  - dissonance scalar d for the TOPCAT-Americas vs TOPCAT-Russia-Georgia pair

Baseline (unperturbed) adherence values are used throughout; this sweep is
orthogonal to sensitivity_adherence.py.

Usage
-----
  python scripts/sensitivity_kernel.py \
      --manifest data/mra_hfpef/MANIFEST.json \
      --results manuscript/sensitivity_kernel_results.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dfs.config import COVARIATE_NAMES
from dfs.diagnostics import detect_conservation_violations
from dfs.dissonance import pairwise_dissonance
from dfs.field_constrained import fit_constrained_gp
from dfs.hyperparam import fit_hyperparameters
from dfs.kernel import KERNELS
from dfs.manifest import load_trials
from dfs.schema import TrialBoundaryCondition


KERNEL_ORDER: list[str] = ["matern52", "matern32", "rbf"]

CSV_FIELDNAMES: list[str] = [
    "kernel",
    "nll",
    "adherence_ls",
    "d_topcat",
    "loo_mu",
    "loo_lo",
    "loo_hi",
    "loo_width",
    "loo_inside",
    "n_violations",
]


def _dissonance_topcat(trials: list[TrialBoundaryCondition]) -> float:
    pairs = pairwise_dissonance(trials, "primary_composite")
    for p in pairs:
        ids = set(p.trial_ids)
        if "TOPCAT-Americas" in ids and "TOPCAT-Russia-Georgia" in ids:
            return p.d
    raise KeyError("TOPCAT pair not found in pairwise_dissonance output.")


def _run_loo_with_kernel(
    trials: list[TrialBoundaryCondition],
    kernel_name: str,
) -> dict:
    """Hold out FINEARTS-HF; fit ML-II GP with the named kernel on the other 5."""
    kernel_fn = KERNELS[kernel_name]

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

    hp = fit_hyperparameters(x_norm, y, noise, n_restarts=5, seed=0, kernel_fn=kernel_fn)

    adh_idx = list(COVARIATE_NAMES).index("adherence_proxy")
    adherence_ls = float(hp.length_scales[adh_idx])

    gp = fit_constrained_gp(
        x_norm, y, noise,
        sigma2=hp.sigma2,
        length_scales=hp.length_scales,
        inequality_constraints=[],
        kernel_fn=kernel_fn,
    )
    mu, var = gp.predict(anchor_norm)

    observed = held_out.outcomes["primary_composite"]["log_hr"]
    se_observed = held_out.outcomes["primary_composite"]["se"]
    var_pred = float(var[0]) + float(se_observed ** 2)
    se_pred = float(np.sqrt(max(var_pred, 0.0)))
    lo = float(mu[0]) - 1.96 * se_pred
    hi = float(mu[0]) + 1.96 * se_pred
    inside = bool(lo <= observed <= hi)

    return {
        "kernel": kernel_name,
        "nll": float(hp.neg_log_marginal_likelihood),
        "adherence_ls": adherence_ls,
        "loo_mu": float(mu[0]),
        "loo_lo": lo,
        "loo_hi": hi,
        "loo_width": hi - lo,
        "loo_inside": inside,
    }


def run(trials: list[TrialBoundaryCondition]) -> list[dict]:
    rows: list[dict] = []
    d_topcat_baseline = _dissonance_topcat(trials)
    violations = detect_conservation_violations(trials)
    print(f"\n=== Kernel sweep over {KERNEL_ORDER} ===")
    print(f"  (baseline d_topcat={d_topcat_baseline:.3f}, "
          f"conservation violations={len(violations)})\n")

    for name in KERNEL_ORDER:
        print(f"  kernel={name:<9} ... ", end="", flush=True)
        res = _run_loo_with_kernel(trials, name)
        res["d_topcat"] = d_topcat_baseline
        res["n_violations"] = len(violations)
        rows.append(res)
        print(
            f"nll={res['nll']:+.3f}  ls={res['adherence_ls']:.3f}  "
            f"mu={res['loo_mu']:+.3f}  [{res['loo_lo']:+.3f},{res['loo_hi']:+.3f}]  "
            f"inside={res['loo_inside']}"
        )
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
    parser = argparse.ArgumentParser(description="Kernel-choice sensitivity for DFS POC.")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/mra_hfpef/MANIFEST.json"),
        help="Path to MANIFEST.json (default: data/mra_hfpef/MANIFEST.json)",
    )
    parser.add_argument(
        "--results", type=Path,
        default=Path("manuscript/sensitivity_kernel_results.csv"),
        help="Path for the results CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Loading manifest: {args.manifest}")
    trials = load_trials(args.manifest)
    print(f"  Loaded {len(trials)} trials: {[t.trial_id for t in trials]}")

    rows = run(trials)
    write_csv(rows, args.results)

    print("\n--- Summary ---")
    inside = sum(1 for r in rows if r["loo_inside"])
    print(f"LOO inside CrI: {inside}/{len(rows)} kernels")
    best = min(rows, key=lambda r: r["nll"])
    print(f"Best ML-II fit: kernel={best['kernel']} (nll={best['nll']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
