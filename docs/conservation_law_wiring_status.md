# Conservation Law Wiring Status

> DFS POC — Phase-1b implementation record.
> Updated: 2026-04-15.

Six conservation laws are defined in `dfs/conservation.py`.  Only one
is compatible with the current single-output value-constraint API
(`InequalityConstraint` in `dfs/field_constrained.py`).  The other five
require architectural extensions that are deferred to Phase-2.

---

## Wired (Phase-1b B)

### `k_sign` — ΔK⁺ >= 0 wherever MR occupancy > 0

**Status:** Implemented in `dfs/safety.py::fit_safety_field`.

**API fit:** Single-output GP on the `safety["delta_k"]` surface.
Constraint is a point-wise inequality on a Latin-hypercube grid of 50
virtual observation points sampled from the normalised covariate space
with `mr_occupancy` clamped to the active region [0.1, 1.0].

**Real-data result (MRA-HFpEF, 6 trials):**
- All 6 trials included; 0 skipped.
- All `satisfies_k_sign = true` (observed delta-K+: 0.02 to 0.23 mmol/L).
- `any_binding = false` — the constraint is non-binding everywhere on
  the 50-point virtual grid.
- Minimum posterior mean on the virtual grid: +0.047 mmol/L.
- Conclusion: ΔK⁺ >= 0 holds mechanistically across all MRA-HFpEF
  trials, as expected from aldosterone blockade.  TOPCAT-Russia/Georgia
  has the lowest value (0.02 mmol/L), consistent with the near-zero MR
  occupancy implied by the non-adherence signal.

**Artefacts produced by `scripts/run_dfs.py`:**
- `outputs/safety_delta_k_lvef_egfr.png` — posterior mean + SD heatmap
  of the ΔK⁺ surface over the LVEF × eGFR slice.
- `outputs/k_sign_constraint_report.json` — per-trial ΔK⁺ values,
  binding status, and hyperparameters.

---

## Deferred (Phase-2) — architectural blockers

### `mortality_decomposition` — HR(ACM) = p·HR(CV) + q·HR(non-CV)

**Status:** Deferred.

**Blocker:** This constraint relates outputs of *three separate GPs*
(one per outcome: ACM, CV death, non-CV death).  The current
`fit_constrained_gp` API operates on a single GP output vector;
encoding a cross-GP linear equality requires a **multi-output GP**
framework where all outcome surfaces share a joint posterior (e.g.
intrinsic co-regionalisation model, or a joint QP over stacked
output vectors).

### `cv_death_subdecomposition` — HR(CV) = Σᵢ propᵢ · HR_i

**Status:** Deferred.

**Blocker:** Same as `mortality_decomposition` — the constraint spans
multiple output GPs (sudden death, pump failure, MI, stroke).  Requires
a multi-output GP or a joint QP over the stacked outcome vectors.

### `sbp_sign` — ΔSBP ≤ 0 monotone in MR occupancy

**Status:** Deferred.

**Blocker:** Classified as a *soft* constraint (see `conservation.py`
rationale — some normotensive subgroups show SBP unchanged).  Soft
constraints require an **additive penalty term** in the QP objective
function rather than a hard inequality.  The current
`fit_constrained_gp` API only supports hard inequality constraints via
CVXPY; a weighted penalty formulation (quadratic slack or barrier) must
be added to the QP objective before this law can be wired.

### `dose_monotonicity` — ∂log-HR/∂dose ≤ 0 within tested dose range

**Status:** Deferred.

**Blocker:** A gradient constraint (∂f/∂x ≤ 0) cannot be enforced
purely at observation points.  It requires **joint GP modelling of f
and ∂f**: either (a) placing the derivative process in the GP via an
analytically derived cross-covariance (Solak et al. 2003), or (b)
using finite-difference virtual observations on a sufficiently fine
dose grid.  Neither is available in the current kernel/QP stack.

### `egfr_dip_plateau` — acute ΔeGFR < 0 at 4 months, recovering by 24 months

**Status:** Deferred.

**Blocker:** Two compounding extensions needed: (1) same soft-penalty
extension as `sbp_sign` (constraint is soft per `conservation.py`), and
(2) a **time-dependent constraint** — the sign of ΔeGFR flips between
the 4-month and 24-month timepoints.  The current GP operates on a
static covariate space with no time dimension; incorporating a temporal
axis requires either a separate GP per timepoint or a spatio-temporal
kernel.

---

## Summary table

| Law | Phase | API requirement |
|---|---|---|
| `k_sign` | **1b — done** | Single-output GP + hard inequality |
| `mortality_decomposition` | Phase-2 | Multi-output GP |
| `cv_death_subdecomposition` | Phase-2 | Multi-output GP |
| `sbp_sign` | Phase-2 | Soft QP penalty term |
| `dose_monotonicity` | Phase-2 | Joint f + df/dx GP |
| `egfr_dip_plateau` | Phase-2 | Soft penalty + time-dependent constraint |
