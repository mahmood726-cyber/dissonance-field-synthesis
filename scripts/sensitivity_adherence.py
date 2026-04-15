"""Adherence-proxy sensitivity analysis for the DFS POC.

Runs two sweeps:

  Sweep 1 — TOPCAT-Russia/Georgia adherence_proxy in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
  Sweep 2 — FINEARTS-HF adherence_proxy in [0.70, 0.80, 0.85, 0.90, 0.95]

For each scenario, reports:
  - dissonance scalar d for the TOPCAT-Americas vs TOPCAT-Russia/Georgia pair
  - fitted ML-II adherence_proxy length-scale
  - LOO FINEARTS-HF: predicted log-HR mean, 95% CrI lower/upper/width
  - LOO FINEARTS-HF: observed inside CrI? (bool)
  - number of conservation-law violations

Usage
-----
  python scripts/sensitivity_adherence.py \
      --manifest data/mra_hfpef/MANIFEST.json \
      --out outputs/ \
      --results manuscript/sensitivity_results.csv
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is importable when the script is run directly
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dfs.config import COVARIATE_NAMES
from dfs.diagnostics import detect_conservation_violations
from dfs.dissonance import pairwise_dissonance
from dfs.field_constrained import fit_constrained_gp
from dfs.hyperparam import fit_hyperparameters
from dfs.manifest import load_trials
from dfs.schema import TrialBoundaryCondition

# ---------------------------------------------------------------------------
# Sweep grids (authoritative definitions)
# ---------------------------------------------------------------------------

SWEEP1_RG_GRID: list[float] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
SWEEP2_FH_GRID: list[float] = [0.70, 0.80, 0.85, 0.90, 0.95]

BASELINE_RG_ADH: float = 0.40   # value in the original JSON
BASELINE_FH_ADH: float = 0.90   # value in the original JSON

CSV_FIELDNAMES: list[str] = [
    "sweep",
    "rg_adh",
    "fh_adh",
    "d_topcat",
    "adherence_ls",
    "loo_mu",
    "loo_lo",
    "loo_hi",
    "loo_width",
    "loo_inside",
    "n_violations",
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _replace_adherence(trial: TrialBoundaryCondition, value: float) -> TrialBoundaryCondition:
    """Return a new frozen TrialBoundaryCondition with adherence_proxy swapped."""
    return dataclasses.replace(
        trial,
        anchor_covariates={**trial.anchor_covariates, "adherence_proxy": value},
    )


def _build_trial_list(
    base_trials: list[TrialBoundaryCondition],
    rg_adh: float,
    fh_adh: float,
) -> list[TrialBoundaryCondition]:
    """Deep-copy-equivalent: return a new list with the two anchors replaced."""
    result = []
    for t in base_trials:
        if t.trial_id == "TOPCAT-Russia-Georgia":
            t = _replace_adherence(t, rg_adh)
        elif t.trial_id == "FINEARTS-HF":
            t = _replace_adherence(t, fh_adh)
        result.append(t)
    return result


def _dissonance_topcat(trials: list[TrialBoundaryCondition]) -> float:
    """Return pairwise-dissonance scalar for TOPCAT-Americas vs TOPCAT-Russia-Georgia."""
    pairs = pairwise_dissonance(trials, "primary_composite")
    for p in pairs:
        ids = set(p.trial_ids)
        if "TOPCAT-Americas" in ids and "TOPCAT-Russia-Georgia" in ids:
            return p.d
    raise KeyError(
        "TOPCAT pair (Americas, Russia-Georgia) not found in pairwise_dissonance output."
    )


def _run_loo(
    trials: list[TrialBoundaryCondition],
) -> dict[str, float | bool]:
    """Hold out FINEARTS-HF; fit ML-II GP on the remaining 5; predict the held-out anchor."""
    held_out = next(t for t in trials if t.trial_id == "FINEARTS-HF")
    kept = [t for t in trials if t.trial_id != "FINEARTS-HF"]

    x = np.array([[t.anchor_covariates[c] for c in COVARIATE_NAMES] for t in kept])
    y = np.array([t.outcomes["primary_composite"]["log_hr"] for t in kept])
    noise = np.array([t.outcomes["primary_composite"]["se"] ** 2 for t in kept])

    # Normalise to [0, 1] on the training data (same convention as test_loo_fineartshf.py)
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    x_norm = (x - mins) / ranges

    anchor = np.array([held_out.anchor_covariates[c] for c in COVARIATE_NAMES])
    anchor_norm = ((anchor - mins) / ranges).reshape(1, -1)

    # ML-II hyperparameter fit
    hp = fit_hyperparameters(x_norm, y, noise, n_restarts=5, seed=0)

    # adherence_proxy is the 7th covariate (index 6 in COVARIATE_NAMES)
    adh_idx = list(COVARIATE_NAMES).index("adherence_proxy")
    adherence_ls = float(hp.length_scales[adh_idx])

    # GP prediction (unconstrained — no inequality constraints here)
    gp = fit_constrained_gp(
        x_norm, y, noise,
        sigma2=hp.sigma2,
        length_scales=hp.length_scales,
        inequality_constraints=[],
    )
    mu, var = gp.predict(anchor_norm)

    observed = held_out.outcomes["primary_composite"]["log_hr"]
    se_observed = held_out.outcomes["primary_composite"]["se"]

    # LOO predictive variance: GP posterior var + held-out observation noise
    var_pred = float(var[0]) + float(se_observed ** 2)
    se_pred = float(np.sqrt(max(var_pred, 0.0)))
    lo = float(mu[0]) - 1.96 * se_pred
    hi = float(mu[0]) + 1.96 * se_pred
    width = hi - lo
    inside = bool(lo <= observed <= hi)

    return {
        "loo_mu": float(mu[0]),
        "loo_lo": lo,
        "loo_hi": hi,
        "loo_width": width,
        "loo_inside": inside,
        "adherence_ls": adherence_ls,
    }


def _run_scenario(
    base_trials: list[TrialBoundaryCondition],
    sweep_label: str,
    rg_adh: float,
    fh_adh: float,
) -> dict:
    """Run a single (rg_adh, fh_adh) scenario; return a flat result dict."""
    trials = _build_trial_list(base_trials, rg_adh, fh_adh)

    try:
        d_topcat = _dissonance_topcat(trials)
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] dissonance failed for rg={rg_adh}, fh={fh_adh}: {exc}", file=sys.stderr)
        d_topcat = float("nan")

    try:
        loo = _run_loo(trials)
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] LOO failed for rg={rg_adh}, fh={fh_adh}: {exc}", file=sys.stderr)
        loo = {
            "loo_mu": float("nan"), "loo_lo": float("nan"), "loo_hi": float("nan"),
            "loo_width": float("nan"), "loo_inside": False, "adherence_ls": float("nan"),
        }

    violations = detect_conservation_violations(trials)
    n_violations = len(violations)

    return {
        "sweep": sweep_label,
        "rg_adh": rg_adh,
        "fh_adh": fh_adh,
        "d_topcat": d_topcat,
        "adherence_ls": loo["adherence_ls"],
        "loo_mu": loo["loo_mu"],
        "loo_lo": loo["loo_lo"],
        "loo_hi": loo["loo_hi"],
        "loo_width": loo["loo_width"],
        "loo_inside": loo["loo_inside"],
        "n_violations": n_violations,
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def run_sweep1(base_trials: list[TrialBoundaryCondition]) -> list[dict]:
    """Sweep 1: vary TOPCAT-Russia/Georgia adherence_proxy; FINEARTS-HF fixed at baseline."""
    rows = []
    print(f"\n=== Sweep 1: TOPCAT-Russia/Georgia adherence_proxy grid {SWEEP1_RG_GRID} ===")
    for rg_adh in SWEEP1_RG_GRID:
        print(f"  rg_adh={rg_adh:.2f} ... ", end="", flush=True)
        row = _run_scenario(base_trials, "sweep1_rg", rg_adh, BASELINE_FH_ADH)
        rows.append(row)
        print(
            f"d={row['d_topcat']:.3f}  ls={row['adherence_ls']:.3f}  "
            f"mu={row['loo_mu']:+.3f}  [{row['loo_lo']:+.3f},{row['loo_hi']:+.3f}]  "
            f"inside={row['loo_inside']}"
        )
    return rows


def run_sweep2(base_trials: list[TrialBoundaryCondition]) -> list[dict]:
    """Sweep 2: vary FINEARTS-HF adherence_proxy; Russia/Georgia fixed at baseline."""
    rows = []
    print(f"\n=== Sweep 2: FINEARTS-HF adherence_proxy grid {SWEEP2_FH_GRID} ===")
    for fh_adh in SWEEP2_FH_GRID:
        print(f"  fh_adh={fh_adh:.2f} ... ", end="", flush=True)
        row = _run_scenario(base_trials, "sweep2_fh", BASELINE_RG_ADH, fh_adh)
        rows.append(row)
        print(
            f"d={row['d_topcat']:.3f}  ls={row['adherence_ls']:.3f}  "
            f"mu={row['loo_mu']:+.3f}  [{row['loo_lo']:+.3f},{row['loo_hi']:+.3f}]  "
            f"inside={row['loo_inside']}"
        )
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            # Format floats to 6 dp; booleans as True/False
            out = {}
            for k in CSV_FIELDNAMES:
                val = row[k]
                if isinstance(val, float):
                    out[k] = f"{val:.6f}" if math.isfinite(val) else str(val)
                else:
                    out[k] = val
            writer.writerow(out)
    print(f"\nResults written to: {path}")


def plot_results(rows: list[dict], out_dir: Path) -> None:
    """4-panel PNG: LOO mean, LOO width, adherence length-scale, dissonance d."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping PNG output.", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Split rows by sweep label
    sweep1 = [r for r in rows if r["sweep"] == "sweep1_rg"]
    sweep2 = [r for r in rows if r["sweep"] == "sweep2_fh"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Adherence-Proxy Sensitivity Analysis", fontsize=13, fontweight="bold")

    def _plot_panel(ax, x_vals, y_vals, xlabel, ylabel, title, inside_flags=None, baseline=None):
        finite = [(x, y) for x, y in zip(x_vals, y_vals) if math.isfinite(y)]
        if not finite:
            ax.set_title(title + "\n(no finite data)")
            return
        xs, ys = zip(*finite)
        ax.plot(xs, ys, "o-", color="#2c7bb6", linewidth=1.8, markersize=6)
        if inside_flags is not None:
            for x, y, ins in zip(x_vals, y_vals, inside_flags):
                if math.isfinite(y):
                    col = "#1a9641" if ins else "#d73027"
                    ax.plot(x, y, "o", color=col, markersize=8, zorder=5)
        if baseline is not None:
            ax.axvline(baseline, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                       label=f"baseline={baseline}")
            ax.legend(fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Panel A: LOO mean vs sweep value (both sweeps)
    ax = axes[0, 0]
    if sweep1:
        xs1 = [r["rg_adh"] for r in sweep1]
        ys1 = [r["loo_mu"] for r in sweep1]
        ins1 = [r["loo_inside"] for r in sweep1]
        ax.plot(xs1, ys1, "o-", color="#2c7bb6", label="Sweep 1 (RG adh)", linewidth=1.8)
        for x, y, ins in zip(xs1, ys1, ins1):
            if math.isfinite(y):
                ax.plot(x, y, "o", color="#1a9641" if ins else "#d73027", markersize=8, zorder=5)
        ax.axvline(BASELINE_RG_ADH, color="#2c7bb6", linestyle="--", alpha=0.5, linewidth=1)
    if sweep2:
        xs2 = [r["fh_adh"] for r in sweep2]
        ys2 = [r["loo_mu"] for r in sweep2]
        ins2 = [r["loo_inside"] for r in sweep2]
        ax.plot(xs2, ys2, "s-", color="#d7191c", label="Sweep 2 (FH adh)", linewidth=1.8)
        for x, y, ins in zip(xs2, ys2, ins2):
            if math.isfinite(y):
                ax.plot(x, y, "s", color="#1a9641" if ins else "#d73027", markersize=8, zorder=5)
        ax.axvline(BASELINE_FH_ADH, color="#d7191c", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(-0.174353, color="black", linestyle=":", linewidth=1, alpha=0.7,
               label="Observed FINEARTS-HF")
    ax.set_xlabel("Adherence proxy value", fontsize=9)
    ax.set_ylabel("LOO predicted log-HR mean", fontsize=9)
    ax.set_title("(A) LOO predicted log-HR mean\ngreen=inside CrI, red=outside", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: LOO CrI width
    ax = axes[0, 1]
    if sweep1:
        xs1 = [r["rg_adh"] for r in sweep1]
        ys1 = [r["loo_width"] for r in sweep1]
        ax.plot(xs1, ys1, "o-", color="#2c7bb6", label="Sweep 1 (RG adh)", linewidth=1.8)
    if sweep2:
        xs2 = [r["fh_adh"] for r in sweep2]
        ys2 = [r["loo_width"] for r in sweep2]
        ax.plot(xs2, ys2, "s-", color="#d7191c", label="Sweep 2 (FH adh)", linewidth=1.8)
    ax.set_xlabel("Adherence proxy value", fontsize=9)
    ax.set_ylabel("LOO 95% CrI width (log-HR)", fontsize=9)
    ax.set_title("(B) LOO CrI width", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel C: Fitted adherence_proxy length-scale
    ax = axes[1, 0]
    if sweep1:
        ax.plot(xs1, [r["adherence_ls"] for r in sweep1],
                "o-", color="#2c7bb6", label="Sweep 1 (RG adh)", linewidth=1.8)
    if sweep2:
        ax.plot(xs2, [r["adherence_ls"] for r in sweep2],
                "s-", color="#d7191c", label="Sweep 2 (FH adh)", linewidth=1.8)
    ax.set_xlabel("Adherence proxy value", fontsize=9)
    ax.set_ylabel("ML-II adherence length-scale", fontsize=9)
    ax.set_title("(C) Fitted adherence length-scale\n(small = adherence dominates GP)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: Dissonance d (TOPCAT pair)
    ax = axes[1, 1]
    if sweep1:
        xs1d = [r["rg_adh"] for r in sweep1]
        ys1d = [r["d_topcat"] for r in sweep1]
        ax.plot(xs1d, ys1d, "o-", color="#2c7bb6", label="Sweep 1 (RG adh)", linewidth=1.8)
        ax.axvline(BASELINE_RG_ADH, color="#2c7bb6", linestyle="--", alpha=0.5, linewidth=1)
    if sweep2:
        xs2d = [r["fh_adh"] for r in sweep2]
        ys2d = [r["d_topcat"] for r in sweep2]
        ax.plot(xs2d, ys2d, "s-", color="#d7191c", label="Sweep 2 (FH adh)", linewidth=1.8)
    ax.set_xlabel("Adherence proxy value", fontsize=9)
    ax.set_ylabel("Dissonance scalar d (TOPCAT pair)", fontsize=9)
    ax.set_title("(D) TOPCAT-Americas vs Russia/Georgia\ndissonance scalar d", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / "sensitivity_adherence.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to: {png_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adherence-proxy sensitivity analysis for DFS POC."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/mra_hfpef/MANIFEST.json"),
        help="Path to MANIFEST.json (default: data/mra_hfpef/MANIFEST.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Output directory for the PNG (default: outputs/)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("manuscript/sensitivity_results.csv"),
        help="Path for the committable results CSV (default: manuscript/sensitivity_results.csv)",
    )
    parser.add_argument(
        "--sweep-size",
        type=int,
        default=0,
        help="Truncate both grids to this many points (0 = full; used by tests for speed).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Loading manifest: {args.manifest}")
    base_trials = load_trials(args.manifest)
    print(f"  Loaded {len(base_trials)} trials: {[t.trial_id for t in base_trials]}")

    # Optionally truncate grids (for subprocess test speed)
    sweep1_grid = SWEEP1_RG_GRID
    sweep2_grid = SWEEP2_FH_GRID
    if args.sweep_size and args.sweep_size > 0:
        sweep1_grid = SWEEP1_RG_GRID[:args.sweep_size]
        sweep2_grid = SWEEP2_FH_GRID[:args.sweep_size]
        print(f"  [--sweep-size {args.sweep_size}] Truncated grids: "
              f"sweep1={sweep1_grid}, sweep2={sweep2_grid}")

    # Run sweeps (using module-level globals, but we need to temporarily shadow
    # the grids if --sweep-size was given — simplest approach: inline loops)
    all_rows: list[dict] = []

    print(f"\n=== Sweep 1: TOPCAT-Russia/Georgia adherence_proxy grid {sweep1_grid} ===")
    for rg_adh in sweep1_grid:
        print(f"  rg_adh={rg_adh:.2f} ... ", end="", flush=True)
        row = _run_scenario(base_trials, "sweep1_rg", rg_adh, BASELINE_FH_ADH)
        all_rows.append(row)
        print(
            f"d={row['d_topcat']:.3f}  ls={row['adherence_ls']:.3f}  "
            f"mu={row['loo_mu']:+.3f}  [{row['loo_lo']:+.3f},{row['loo_hi']:+.3f}]  "
            f"inside={row['loo_inside']}"
        )

    print(f"\n=== Sweep 2: FINEARTS-HF adherence_proxy grid {sweep2_grid} ===")
    for fh_adh in sweep2_grid:
        print(f"  fh_adh={fh_adh:.2f} ... ", end="", flush=True)
        row = _run_scenario(base_trials, "sweep2_fh", BASELINE_RG_ADH, fh_adh)
        all_rows.append(row)
        print(
            f"d={row['d_topcat']:.3f}  ls={row['adherence_ls']:.3f}  "
            f"mu={row['loo_mu']:+.3f}  [{row['loo_lo']:+.3f},{row['loo_hi']:+.3f}]  "
            f"inside={row['loo_inside']}"
        )

    # Write CSV
    write_csv(all_rows, args.results)

    # Write PNG (gitignored)
    plot_results(all_rows, args.out)

    # Print summary
    s1 = [r for r in all_rows if r["sweep"] == "sweep1_rg"]
    s2 = [r for r in all_rows if r["sweep"] == "sweep2_fh"]
    s1_green = sum(1 for r in s1 if r["loo_inside"])
    s2_green = sum(1 for r in s2 if r["loo_inside"])
    print(f"\n--- Summary ---")
    print(f"Sweep 1: {s1_green}/{len(s1)} scenarios LOO inside CrI")
    print(f"Sweep 2: {s2_green}/{len(s2)} scenarios LOO inside CrI")
    s1_ls = [r["adherence_ls"] for r in s1 if math.isfinite(r["adherence_ls"])]
    s2_ls = [r["adherence_ls"] for r in s2 if math.isfinite(r["adherence_ls"])]
    if s1_ls:
        print(f"Sweep 1 adherence LS range: [{min(s1_ls):.3f}, {max(s1_ls):.3f}]")
    if s2_ls:
        print(f"Sweep 2 adherence LS range: [{min(s2_ls):.3f}, {max(s2_ls):.3f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
