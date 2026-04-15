"""Unit tests for adherence_proxy sensitivity analysis.

Uses a 3-point mini-sweep (2 Russia/Georgia values × 2 FINEARTS-HF values,
overlapping at one point for speed) so no I/O to outputs/ is needed.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from dfs.config import COVARIATE_NAMES
from dfs.diagnostics import detect_conservation_violations
from dfs.dissonance import pairwise_dissonance
from dfs.field_constrained import fit_constrained_gp
from dfs.hyperparam import fit_hyperparameters
from dfs.manifest import load_trials
from dfs.schema import TrialBoundaryCondition

MANIFEST_PATH = Path("data/mra_hfpef/MANIFEST.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_adherence(trial: TrialBoundaryCondition, value: float) -> TrialBoundaryCondition:
    """Return a new frozen record with adherence_proxy replaced."""
    new_covariates = {**trial.anchor_covariates, "adherence_proxy": value}
    return dataclasses.replace(trial, anchor_covariates=new_covariates)


def _run_loo_scenario(
    trials: list[TrialBoundaryCondition],
    held_out_id: str,
    outcome: str = "primary_composite",
) -> dict:
    """Run one LOO scenario; return result dict with standard keys."""
    held_out = next(t for t in trials if t.trial_id == held_out_id)
    kept = [t for t in trials if t.trial_id != held_out_id]

    x = np.array([[t.anchor_covariates[c] for c in COVARIATE_NAMES] for t in kept])
    y = np.array([t.outcomes[outcome]["log_hr"] for t in kept])
    noise = np.array([t.outcomes[outcome]["se"] ** 2 for t in kept])

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    x_norm = (x - mins) / ranges

    anchor = np.array([held_out.anchor_covariates[c] for c in COVARIATE_NAMES])
    anchor_norm = ((anchor - mins) / ranges).reshape(1, -1)

    hp = fit_hyperparameters(x_norm, y, noise, n_restarts=5, seed=0)

    # adherence_proxy is the 7th covariate (index 6)
    adh_idx = list(COVARIATE_NAMES).index("adherence_proxy")
    adherence_ls = float(hp.length_scales[adh_idx])

    gp = fit_constrained_gp(
        x_norm, y, noise,
        sigma2=hp.sigma2, length_scales=hp.length_scales,
        inequality_constraints=[],
    )
    mu, var = gp.predict(anchor_norm)

    observed = held_out.outcomes[outcome]["log_hr"]
    se_observed = held_out.outcomes[outcome]["se"]
    var_pred = float(var[0]) + float(se_observed ** 2)
    se_pred = float(np.sqrt(var_pred))
    lo = float(mu[0]) - 1.96 * se_pred
    hi = float(mu[0]) + 1.96 * se_pred
    width = hi - lo
    inside = bool(lo <= observed <= hi)

    return {
        "mu": float(mu[0]),
        "lo": lo,
        "hi": hi,
        "width": width,
        "inside": inside,
        "adherence_ls": adherence_ls,
    }


def _dissonance_topcat_pair(trials: list[TrialBoundaryCondition], outcome: str = "primary_composite") -> float:
    pairs = pairwise_dissonance(trials, outcome)
    for p in pairs:
        ids = set(p.trial_ids)
        if "TOPCAT-Americas" in ids and "TOPCAT-Russia-Georgia" in ids:
            return p.d
    raise KeyError("TOPCAT pair not found")


def _run_sweep(
    rg_adh: float,
    fh_adh: float,
) -> dict:
    """Run one (rg_adh, fh_adh) scenario and return summary dict."""
    trials = load_trials(MANIFEST_PATH)
    modified = []
    for t in trials:
        if t.trial_id == "TOPCAT-Russia-Georgia":
            t = _replace_adherence(t, rg_adh)
        elif t.trial_id == "FINEARTS-HF":
            t = _replace_adherence(t, fh_adh)
        modified.append(t)

    d_topcat = _dissonance_topcat_pair(modified)
    violations = detect_conservation_violations(modified)
    n_violations = len(violations)
    loo = _run_loo_scenario(modified, "FINEARTS-HF")

    return {
        "rg_adh": rg_adh,
        "fh_adh": fh_adh,
        "d_topcat": d_topcat,
        "adherence_ls": loo["adherence_ls"],
        "loo_mu": loo["mu"],
        "loo_lo": loo["lo"],
        "loo_hi": loo["hi"],
        "loo_width": loo["width"],
        "loo_inside": loo["inside"],
        "n_violations": n_violations,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSensitivityMiniSweep:
    """3-point mini-sweep runs without error; structural checks on output."""

    # Mini grid: 3 Russia/Georgia values × 1 FINEARTS-HF value = 3 scenarios
    # (fast but still exercises the sweep logic)
    MINI_RG_GRID = [0.20, 0.40, 0.60]
    MINI_FH_GRID = [0.90]

    @pytest.fixture(scope="class")
    def sweep_results(self):
        rows = []
        for rg in self.MINI_RG_GRID:
            for fh in self.MINI_FH_GRID:
                rows.append(_run_sweep(rg, fh))
        return rows

    def test_row_count(self, sweep_results):
        expected = len(self.MINI_RG_GRID) * len(self.MINI_FH_GRID)
        assert len(sweep_results) == expected, (
            f"Expected {expected} rows, got {len(sweep_results)}"
        )

    def test_required_columns_present(self, sweep_results):
        required = {
            "rg_adh", "fh_adh", "d_topcat", "adherence_ls",
            "loo_mu", "loo_lo", "loo_hi", "loo_width", "loo_inside", "n_violations",
        }
        for row in sweep_results:
            missing = required - set(row.keys())
            assert not missing, f"Missing columns: {missing}"

    def test_all_numeric_columns_finite(self, sweep_results):
        numeric_keys = ["d_topcat", "adherence_ls", "loo_mu", "loo_lo", "loo_hi", "loo_width"]
        for row in sweep_results:
            for k in numeric_keys:
                val = row[k]
                assert val is not None, f"Column {k!r} is None (silent failure)"
                assert not math.isnan(val), f"Column {k!r} is NaN (silent failure)"
                assert math.isfinite(val), f"Column {k!r} is non-finite: {val}"

    def test_loo_inside_cri_at_least_once(self, sweep_results):
        """Sanity check: LOO observation covered in CrI for at least one scenario."""
        assert any(row["loo_inside"] for row in sweep_results), (
            "LOO observed log-HR was outside the predictive CrI for ALL mini-sweep scenarios. "
            "This suggests a pipeline regression, not just sensitivity."
        )

    def test_dissonance_positive(self, sweep_results):
        for row in sweep_results:
            assert row["d_topcat"] > 0, "Dissonance scalar must be positive"

    def test_no_silent_failure_sentinels(self, sweep_results):
        """n_violations is always a non-negative integer, never None."""
        for row in sweep_results:
            assert isinstance(row["n_violations"], int)
            assert row["n_violations"] >= 0

    def test_loo_width_positive(self, sweep_results):
        for row in sweep_results:
            assert row["loo_width"] > 0, "CrI width must be positive"

    def test_adherence_ls_positive(self, sweep_results):
        for row in sweep_results:
            assert row["adherence_ls"] > 0, "Length-scale must be positive"


class TestSensitivitySweep2Mini:
    """Mini sweep 2: vary FINEARTS-HF adherence while Russia/Georgia is fixed at 0.40."""

    MINI_FH_GRID = [0.70, 0.85, 0.95]
    FIXED_RG = 0.40

    @pytest.fixture(scope="class")
    def sweep2_results(self):
        rows = []
        for fh in self.MINI_FH_GRID:
            rows.append(_run_sweep(self.FIXED_RG, fh))
        return rows

    def test_row_count(self, sweep2_results):
        assert len(sweep2_results) == len(self.MINI_FH_GRID)

    def test_all_finite(self, sweep2_results):
        numeric_keys = ["d_topcat", "adherence_ls", "loo_mu", "loo_lo", "loo_hi", "loo_width"]
        for row in sweep2_results:
            for k in numeric_keys:
                assert math.isfinite(row[k]), f"Column {k!r} non-finite in sweep2: {row[k]}"

    def test_loo_inside_at_least_once(self, sweep2_results):
        assert any(row["loo_inside"] for row in sweep2_results), (
            "LOO observed log-HR outside CrI for ALL sweep-2 scenarios."
        )
