"""Tests for dfs/safety.py — k_sign conservation law wiring.

Three test groups:
  1. Synthetic: all-positive ΔK⁺ → constraint non-binding, field non-negative.
  2. Synthetic: one deliberately-negative ΔK⁺ trial → constraint binds, but
     posterior is still pulled up to ≥ 0 everywhere on the virtual grid.
  3. Real-data: load manifest, fit, assert constraint satisfied per-trial and on grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dfs.config import COVARIATE_NAMES
from dfs.schema import TrialBoundaryCondition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_trial(
    trial_id: str,
    delta_k_value: float,
    delta_k_se: float,
    mr_occupancy: float = 1.0,
    delta_k_source: str = "Direct",
) -> TrialBoundaryCondition:
    """Build a minimal synthetic TrialBoundaryCondition for safety tests."""
    anchor = {n: v for n, v in zip(
        COVARIATE_NAMES,
        (55.0, 60.0, 70.0, 4.3, 0.3, mr_occupancy, 0.9),
    )}
    return TrialBoundaryCondition(
        trial_id=trial_id,
        drug="spironolactone",
        mr_occupancy_equivalent=mr_occupancy,
        anchor_covariates=anchor,
        covariate_ranges={
            "lvef": (45.0, 75.0),
            "egfr": (30.0, 120.0),
            "age": (55.0, 85.0),
            "baseline_k": (3.5, 5.0),
            "dm_fraction": (0.0, 1.0),
            "mr_occupancy": (0.5, 1.5),
            "adherence_proxy": (0.0, 1.0),
        },
        outcomes={
            "primary_composite": {"log_hr": -0.15, "se": 0.09, "baseline_prop": 1.0},
        },
        safety={
            "delta_k": {
                "value": delta_k_value,
                "se": delta_k_se,
                "source": delta_k_source,
            }
        },
        design_priors={"placebo_rate_per_yr": 0.15, "ltfu_fraction": 0.08,
                       "adherence_proxy": 0.9},
    )


# ---------------------------------------------------------------------------
# Test 1: all-positive ΔK⁺ — constraint should be non-binding
# ---------------------------------------------------------------------------

def test_all_positive_delta_k_produces_nonnegative_field() -> None:
    """With only positive ΔK⁺ values, the safety GP should be >= 0 everywhere
    on the virtual constraint grid (constraint non-binding, field stays positive).
    """
    from dfs.safety import fit_safety_field

    trials = [
        _synth_trial("S1", delta_k_value=0.20, delta_k_se=0.04),
        _synth_trial("S2", delta_k_value=0.18, delta_k_se=0.03),
        _synth_trial("S3", delta_k_value=0.25, delta_k_se=0.05),
    ]

    result = fit_safety_field(trials, safety_key="delta_k", constraint="k_sign")

    gp = result["gp"]
    virtual_grid = result["virtual_grid"]

    mu, _ = gp.predict(virtual_grid)
    assert np.all(mu >= -1e-3), (
        f"Posterior mean went negative (min={mu.min():.4f}); "
        "constraint should be non-binding with all-positive observations."
    )

    report = result["report"]
    assert report["n_virtual_obs"] == virtual_grid.shape[0]
    assert isinstance(report["any_binding"], bool)
    # With all-positive observations the constraint should be non-binding
    assert report["any_binding"] is False

    # Per-trial check: all observed ΔK⁺ satisfy >= 0
    for row in report["per_trial"]:
        assert row["observed_delta_k"] >= 0.0


# ---------------------------------------------------------------------------
# Test 2: one deliberately-negative ΔK⁺ trial — constraint must bind and
#         pull the posterior back up to >= 0
# ---------------------------------------------------------------------------

def test_negative_delta_k_trial_constraint_binds_and_field_nonneg() -> None:
    """A ΔK⁺ = -0.5 trial violates the k_sign law.  The QP constraint must pull
    the posterior up so it remains >= 0 everywhere on the virtual grid.
    """
    from dfs.safety import fit_safety_field

    trials = [
        _synth_trial("S1", delta_k_value=0.20, delta_k_se=0.04),
        _synth_trial("S2", delta_k_value=-0.50, delta_k_se=0.04),  # violates k_sign
        _synth_trial("S3", delta_k_value=0.15, delta_k_se=0.03),
    ]

    result = fit_safety_field(trials, safety_key="delta_k", constraint="k_sign")

    gp = result["gp"]
    virtual_grid = result["virtual_grid"]

    mu, _ = gp.predict(virtual_grid)
    # The constraint should have pulled the posterior up so it is >= 0
    assert np.all(mu >= -1e-3), (
        f"Constraint should enforce posterior >= 0, but min={mu.min():.4f}."
    )

    report = result["report"]
    # Constraint must be binding because one observation violates the law
    assert report["any_binding"] is True


# ---------------------------------------------------------------------------
# Test 3: real-data test — load manifest, fit, check constraint everywhere
# ---------------------------------------------------------------------------

MANIFEST = Path("data/mra_hfpef/MANIFEST.json")


@pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="Real-data manifest not present — skipped in isolated environments.",
)
def test_real_data_k_sign_constraint_satisfied() -> None:
    """On the real MRA-HFpEF data, every trial has ΔK⁺ >= 0 (mechanistically
    expected from aldosterone blockade), so the constraint should be non-binding
    and the posterior should be >= 0 everywhere on the virtual grid.
    """
    from dfs.manifest import load_trials
    from dfs.safety import fit_safety_field

    trials = load_trials(MANIFEST)
    result = fit_safety_field(trials, safety_key="delta_k", constraint="k_sign")

    gp = result["gp"]
    virtual_grid = result["virtual_grid"]
    report = result["report"]

    # Posterior must be non-negative everywhere on the virtual grid
    mu, _ = gp.predict(virtual_grid)
    assert np.all(mu >= -1e-3), (
        f"Posterior mean went negative (min={mu.min():.4f}) on real data — "
        "unexpected given all MRA trials report ΔK⁺ >= 0."
    )

    # All included trials must have non-negative observed ΔK⁺
    for row in report["per_trial"]:
        assert row["observed_delta_k"] >= 0.0, (
            f"Trial {row['trial_id']}: observed ΔK⁺ = {row['observed_delta_k']:.3f} < 0. "
            "This would be clinically unexpected for an MRA."
        )

    # On real data the constraint should be non-binding (all positive observations)
    assert report["any_binding"] is False, (
        "k_sign constraint is binding on real data — check if a trial has ΔK⁺ near 0."
    )

    # Sanity: at least some trials were included (have delta_k safety entry)
    assert report["n_included_trials"] >= 1
    assert report["n_skipped_trials"] >= 0
