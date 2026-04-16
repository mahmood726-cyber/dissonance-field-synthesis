"""ML-II optimiser-restart stability analysis for the DFS POC.

Re-fits the LOO FINEARTS-HF hyperparameters across a grid of
(n_restarts, seed) combinations with the default Matern-5/2 kernel.
For each combination we record the best ML-II NLL and the full fitted
length-scale vector, then summarise dispersion across seeds at each
n_restarts level.

The purpose is to answer the reviewer question: "how do you know the
single-seed n_restarts=5 fit used in the primary analysis is a global
optimum rather than a local minimum?"

Grid
----
  n_restarts in {5, 20, 50}
  seed       in {0, 1, ..., 9}
  -> 30 total ML-II fits

Usage
-----
  python scripts/sensitivity_restarts.py \
      --manifest data/mra_hfpef/MANIFEST.json \
      --results manuscript/sensitivity_restarts_results.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dfs.config import COVARIATE_NAMES
from dfs.hyperparam import fit_hyperparameters
from dfs.manifest import load_trials


N_RESTARTS_GRID: list[int] = [5, 20, 50]
SEED_GRID: list[int] = list(range(10))

# Reference: primary analysis uses (n_restarts=5, seed=0)
REF_N_RESTARTS: int = 5
REF_SEED: int = 0

CSV_FIELDNAMES: list[str] = [
    "n_restarts", "seed", "nll", "sigma2",
    "ls_lvef", "ls_egfr", "ls_age", "ls_baseline_k",
    "ls_dm_fraction", "ls_mr_occupancy", "ls_adherence_proxy",
]


def _fit_one(
    x_norm: np.ndarray,
    y: np.ndarray,
    noise: np.ndarray,
    n_restarts: int,
    seed: int,
) -> dict:
    hp = fit_hyperparameters(
        x_norm, y, noise, n_restarts=n_restarts, seed=seed,
    )
    row = {
        "n_restarts": n_restarts,
        "seed": seed,
        "nll": float(hp.neg_log_marginal_likelihood),
        "sigma2": float(hp.sigma2),
    }
    for i, name in enumerate(COVARIATE_NAMES):
        row[f"ls_{name}"] = float(hp.length_scales[i])
    return row


def run(trials) -> list[dict]:
    # Build the same training matrix the primary LOO uses: hold out FINEARTS-HF
    held_out = next(t for t in trials if t.trial_id == "FINEARTS-HF")
    kept = [t for t in trials if t.trial_id != "FINEARTS-HF"]

    x = np.array([[t.anchor_covariates[c] for c in COVARIATE_NAMES] for t in kept])
    y = np.array([t.outcomes["primary_composite"]["log_hr"] for t in kept])
    noise = np.array([t.outcomes["primary_composite"]["se"] ** 2 for t in kept])

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    x_norm = (x - mins) / ranges

    rows: list[dict] = []
    total = len(N_RESTARTS_GRID) * len(SEED_GRID)
    idx = 0
    for nr in N_RESTARTS_GRID:
        for seed in SEED_GRID:
            idx += 1
            print(f"  [{idx:2d}/{total}] n_restarts={nr:<3d} seed={seed} ... ",
                  end="", flush=True)
            row = _fit_one(x_norm, y, noise, nr, seed)
            rows.append(row)
            print(f"nll={row['nll']:+.6f}  adh_ls={row['ls_adherence_proxy']:.4f}")
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


def _summarise(rows: list[dict]) -> None:
    print("\n--- Per-n_restarts dispersion across 10 seeds ---")
    print(f"{'n_restarts':>12}  {'NLL mean':>10}  {'NLL std':>10}  "
          f"{'NLL range':>12}  {'AdhLS mean':>11}  {'AdhLS std':>10}  {'AdhLS range':>12}")
    ref_nll = None
    for nr in N_RESTARTS_GRID:
        sub = [r for r in rows if r["n_restarts"] == nr]
        nll_vals = [r["nll"] for r in sub]
        adh_vals = [r["ls_adherence_proxy"] for r in sub]
        nll_mean = mean(nll_vals)
        nll_std = pstdev(nll_vals)
        nll_rng = max(nll_vals) - min(nll_vals)
        adh_mean = mean(adh_vals)
        adh_std = pstdev(adh_vals)
        adh_rng = max(adh_vals) - min(adh_vals)
        print(f"{nr:>12d}  {nll_mean:+10.6f}  {nll_std:10.6f}  {nll_rng:12.6f}  "
              f"{adh_mean:11.6f}  {adh_std:10.6f}  {adh_rng:12.6f}")
        if nr == REF_N_RESTARTS:
            ref = next(r for r in sub if r["seed"] == REF_SEED)
            ref_nll = ref["nll"]
            print(f"   (reference: n_restarts={REF_N_RESTARTS}, seed={REF_SEED} -> "
                  f"nll={ref_nll:+.6f}, adh_ls={ref['ls_adherence_proxy']:.6f})")

    # Best overall vs reference gap
    best = min(rows, key=lambda r: r["nll"])
    print(f"\nBest NLL across all 30 fits: {best['nll']:+.6f} "
          f"(n_restarts={best['n_restarts']}, seed={best['seed']})")
    if ref_nll is not None:
        gap = ref_nll - best["nll"]
        print(f"Reference fit is {gap:+.6f} above the best NLL seen "
              f"(lower NLL = better fit; positive gap = reference is worse).")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML-II restart-stability sweep.")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/mra_hfpef/MANIFEST.json"),
    )
    parser.add_argument(
        "--results", type=Path,
        default=Path("manuscript/sensitivity_restarts_results.csv"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Loading manifest: {args.manifest}")
    trials = load_trials(args.manifest)
    print(f"  Loaded {len(trials)} trials: {[t.trial_id for t in trials]}")
    print(f"\n=== Restart-stability sweep: "
          f"n_restarts in {N_RESTARTS_GRID}, seeds in {SEED_GRID} ===\n")
    rows = run(trials)
    write_csv(rows, args.results)
    _summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
