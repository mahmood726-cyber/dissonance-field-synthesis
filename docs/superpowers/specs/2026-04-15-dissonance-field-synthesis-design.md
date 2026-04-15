# Dissonance Field Synthesis (DFS) — Design Specification

- **Date:** 2026-04-15
- **Status:** Draft, pre-implementation
- **Author:** Mahmood Ahmad (clinical), Claude (methodological scaffolding)
- **Scope:** Proof-of-concept on MRA in HFpEF, followed by reusable engine

---

## 1. Motivation

Meta-analysis silently imports three assumptions:

1. The unit of synthesis is the study.
2. The output is a pooled point estimate with an uncertainty interval.
3. Evidence lives on a number line.

Dropping any one of these opens a different synthesis universe. Dissonance Field Synthesis (DFS) drops all three simultaneously:

- **Primitive:** the unit is the **pairwise disagreement between trials**, not the trial itself.
- **Substrate:** effects live as a **field over covariate space**, constrained by **conservation laws** drawn from pharmacology and physiology.
- **Output:** instead of a pooled effect, DFS produces a disagreement map, a fitted effect field, a **mind-change price** map, and a **feasibility region** of clinically defensible recommendations.

DFS is not a refinement of meta-analysis. It is the technique you would invent if meta-analysis did not exist and you were told to synthesize clinical trial evidence.

## 2. Clinical target

**Question:** In patients with heart failure with preserved ejection fraction (HFpEF), which patients benefit from mineralocorticoid-receptor antagonist (MRA) therapy?

The HFpEF MRA evidence base is the strongest possible testbed for DFS:

- Trials genuinely disagree (TOPCAT-Americas vs TOPCAT-Russia/Georgia; TOPCAT vs FINEARTS-HF).
- The mechanism is well-characterised: MR blockade raises serum K⁺, lowers SBP, causes an acute eGFR dip, and provides a cardiac-fibrosis-mediated hard-endpoint benefit.
- Outcomes decompose cleanly: primary composite = CV death + HF hospitalisation; ACM = CV death + non-CV death; CV death = sudden + pump-failure + MI + stroke.
- Clinical confusion is persistent: prescribing in HFpEF remains inconsistent, and the steroidal-vs-non-steroidal debate muddies the water.

A second question — **does MR-blockade class matter, or only receptor occupancy?** — is deferred to a follow-up paper once the machinery works.

## 3. Architecture

```
  ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
  │  Trial      │      │  Dissonance  │      │  Covariate      │
  │  extracts   │─────▶│  extractor   │─────▶│  field fitter   │
  │  (6 trials) │      │  (pairwise)  │      │  (constrained)  │
  └─────────────┘      └──────────────┘      └───────┬─────────┘
                                                     │
                                                     ▼
               ┌──────────────┐     ┌──────────────────────────┐
               │ Conservation │     │   DFS outputs (4 + 1)    │
               │ law library  │────▶│  • dissonance map        │
               │ (6 laws)     │     │  • effect field          │
               └──────────────┘     │  • mind-change price map │
                                    │  • feasibility region    │
                                    │  • conservation diagnos. │
                                    └──────────────────────────┘
```

**Core inversion vs meta-analysis:** no pooled estimate is ever computed. Effects exist as a function over covariate space; trials are boundary conditions; conservation laws are hard or soft constraints on the fit; the decision-relevant outputs are read off the fitted field, not summarised from it.

## 4. Input schema

Each trial contributes a **boundary-condition record**. *The values in the example column below are illustrative placeholders for schema design only — not source-verified extracts. Real values are populated during POC data curation from the Finerenone / Cardiology Mortality Atlas fixtures.*

| Field | Role | Example (TOPCAT-Americas) |
|---|---|---|
| `trial_id` | label | `TOPCAT-Americas` |
| `drug` | drug name | spironolactone |
| `mr_occupancy_equivalent` | covariate (1.0 = 25 mg spironolactone, mapped by in vitro potency) | 1.0 |
| `anchor_covariates` | 7-vector: LVEF, eGFR, age, K⁺, DM fraction, MR-occupancy, adherence-proxy | (57, 66, 72, 4.3, 0.32, 1.0, 0.85) |
| `covariate_ranges` | support of each covariate within the trial | LVEF ∈ [45, 75], eGFR ∈ [30, 120], … |
| `primary_composite` | log-HR and SE | log-HR = −0.186, SE = 0.080 |
| `decomposed_outcomes` | for each of (ACM, CV death, non-CV death, HF hosp, sudden, pump-failure, MI, stroke): log-HR and SE, plus baseline-event proportion | {ACM: (−0.08, 0.07, prop=1.0), CV death: (−0.12, 0.09, prop=0.58), …} |
| `safety_outcomes` | ΔK⁺ at 4 months, hyperkalemia incidence, ΔeGFR, ΔSBP with SEs | {ΔK⁺: (0.21 mEq/L, 0.03), …} |
| `design_priors` | placebo event rate, LTFU fraction, adherence proxy | {placebo_rate: 0.18/yr, ltfu: 0.09, adherence_proxy: 0.85} |

**Trials included** (six boundary conditions):

1. **TOPCAT-Americas** (spironolactone, HFpEF, subset of main trial)
2. **TOPCAT-Russia/Georgia** (spironolactone, HFpEF, deliberately split from TOPCAT-Americas; dissonance is the signal)
3. **FINEARTS-HF** (finerenone, HFpEF/HFmrEF)
4. **FIDELIO-DKD HF-subgroup** (finerenone, DKD patients with HF at baseline)
5. **FIGARO-DKD HF-subgroup** (finerenone, DKD patients with HF at baseline)
6. **Aldo-DHF** (spironolactone, HFpEF, smaller older trial)

**Why split TOPCAT:** the regional disagreement is the central dissonance datum. Pooling it as one attenuated estimate discards exactly the information DFS is designed to use. Splitting it — with adherence-proxy as a distinguishing covariate — lets the field explain the gap rather than average it away.

**Why include FIDELIO/FIGARO subgroups:** they extend the covariate support of MR-blockade effect into the DKD population, which is essential for the feasibility region to cover realistic HFpEF patients (most of whom have some degree of CKD).

## 5. The field model

**Formal definition:**

Let covariate vector **x** ∈ ℝ⁷ = (LVEF, eGFR, age, baseline K⁺, DM, MR-occupancy, adherence-proxy). Define *f*(**x**) = log-HR for the primary composite.

*f* is modelled as a Gaussian process:

    f ~ GP(m(x), k(x, x'))

with:

- **Prior mean** *m*(**x**) = β₀ + β_occ · occupancy
  - β₀ = 0 (null prior on baseline effect at zero occupancy)
  - β_occ ∈ [−0.15, −0.05] (mild negative: a 1.0× occupancy shifts log-HR prior by 5–15%, weak enough that trial data dominates)
- **Kernel** *k*(**x**, **x**') = σ² · Matérn₅/₂(‖**x** − **x**'‖_L)
  - L = diag(ℓ₁, …, ℓ₇): per-covariate length-scales (ARD)
  - ARD-Matérn 5/2 is smooth (twice differentiable) but not infinitely so, avoiding RBF over-smoothing; ARD lets adherence-proxy have a short length-scale (encoding "adherence matters a lot") while LVEF can have a longer one.
- **Likelihood:** trial *i* contributes observation *y_i* = log-HR̂_i at anchor **x**_i with heteroscedastic noise variance *v_i* = SE_i².

**Multi-output structure:** the same kernel structure is used for component outcomes (CV death, non-CV death, HF hosp, sudden death, pump-failure, MI, stroke) and for safety surfaces (ΔK⁺, ΔSBP, ΔeGFR). Components share the ARD length-scales but have independent σ² per output.

**Posterior:** standard GP closed form, solved with the added virtual-observation constraints from §6.

**Why GP over alternatives:**

- **BART / boosted trees:** better with large data; we have 6 trials. GP shines at small-n with informative structure.
- **PDEs (diffusion, elasticity):** physically appealing but would require declaring a specific PDE structure for MRA pharmacology that is not independently justified. Overclaim risk.
- **Kernel ridge without Bayesian layer:** loses the calibrated uncertainty that mind-change pricing requires.

**Computational notes:**

- Data observations: 6 trials × ~10 outputs = ~60 heteroscedastic observations. Virtual-observation grid: ~200 points × ~6 constraints (laws 1, 2, 3 value-level; law 5 gradient-level) = ~1200 virtual constraints. QP dimension ≤ ~1300 variables — tractable in seconds via CVXPY (ECOS/OSQP) or SLSQP; falls back to iterative projection if memory becomes an issue.
- ARD hyperparameters fit by type-II maximum likelihood with a mechanism-informed prior (Gamma on ℓ with mode at clinically plausible widths: e.g., ℓ_LVEF ~ Gamma(mode=10%, shape=2)).
- Virtual observation grid: Latin hypercube of 200 points across the support defined by union of trial covariate ranges. Larger grids tested in sensitivity analysis.

## 6. Conservation laws

| # | Law | Type | Encoded as |
|---|---|---|---|
| 1 | **Mortality decomposition**: HR(ACM) is consistent with a convex combination p·HR(CV) + q·HR(non-CV) on the hazard-difference scale, with p, q baseline event proportions | **Hard** | Linear equality on virtual obs: p(z_j)·f_CV(z_j) + q(z_j)·f_nonCV(z_j) − f_ACM(z_j) = 0 at each z_j |
| 2 | **CV-death subdecomposition**: HR(CV death) is a weighted combination of sudden + pump-failure + MI + stroke subtypes | **Hard** | Linear equality on virtual obs (sum of per-subtype log-HRs × subtype proportions = f_CV) |
| 3 | **Mechanism sign (K⁺)**: ΔK⁺(x) ≥ 0 wherever MR occupancy > 0 | **Hard** | Inequality: g_K(z_j) ≥ 0 at all z_j with occupancy > 0 |
| 4 | **Mechanism sign (SBP)**: ΔSBP(x) ≤ 0, monotone in MR occupancy | **Soft** | Penalty λ_SBP · ∑ max(0, g_SBP(z_j))² plus max(0, ∂g_SBP/∂occupancy(z_j))² |
| 5 | **Dose-response monotonicity within drug**: ∂log-HR/∂dose ≤ 0 within each drug's tested dose range | **Hard** | Gradient inequality on virtual obs (using joint GP model of f and ∂f/∂dose) |
| 6 | **eGFR dip-then-plateau**: ΔeGFR negative at 4 months, approaches placebo by 24 months | **Soft** | Two-point penalty: λ_eGFR · [max(0, −g_eGFR_4mo)² + max(0, g_eGFR_24mo − 0.5)²] |

**Encoding — hard constraints via virtual-observation projection:**

Standard technique from constrained-GP literature (Da Veiga & Marrel 2020; Swiler et al. 2020):

1. Sample {z_j} (j = 1..M) on a Latin hypercube across the support of the union of trial covariate ranges.
2. At each z_j, impose the hard-constraint inequality/equality on the GP function values (and their gradients where required; gradients are jointly Gaussian with function values, so this stays linear).
3. Solve the resulting quadratic program for the MAP of the constrained posterior.
4. Sample from the constrained posterior via truncated-Gaussian MCMC (minimax-tilt or Gibbs) for uncertainty propagation.

Soft constraints enter as penalty terms in the negative log-posterior and are jointly minimised with the data fit.

**Clinical judgment in law assignment:**

The hard-vs-soft column is a clinical judgment. The defaults above reflect my read, but three entries are genuinely debatable and the user (cardiologist) will author `dfs/conservation.py` (see §10). Specifically:

- Law 5 (dose-response monotonicity within drug) might be weakened to soft if HFpEF exhibits inverted-U (benefit plateaus, hyperkalemia harm dominates at high dose).
- Law 3 (K⁺ sign) might be softened to allow for measurement-timing artefacts.
- Law 4 (SBP sign) might be hardened, since any MR blockade mechanistically must lower SBP to some extent.

## 7. Outputs

### 7.1 Dissonance map

For each of the 15 trial pairs:

- **Observed dissonance:** *d_ij* = |log-HR_i − log-HR_j| / √(SE_i² + SE_j²)
- **Covariate distance:** Δ**x**_ij = **x**_i − **x**_j, normalised by kernel length-scales
- **Decomposition:** covariate-explained (what a GP would predict given Δ**x**), law-explained (what conservation laws predict given the outcome decomposition), and residual (unexplained)
- Tabular output + heatmap visualisation

Expected finding: TOPCAT-Americas vs TOPCAT-Russia shows highest *d*, almost entirely explained by adherence-proxy difference + proxy LVEF-distribution shift.

### 7.2 Effect field

- GP posterior mean *μ*(**x**) and 95% CrI over covariate space
- Reported as 2-D heatmap slices at clinically meaningful anchor values (e.g., fix age=72, DM=yes, non-steroidal MRA; vary LVEF × eGFR)
- Also reported along 1-D cuts (e.g., effect vs LVEF holding all else at HFpEF-cohort median)

### 7.3 Mind-change price map

At every (**x**, decision threshold *T_cross*, observation target *T_obs*):

    MCP(x; T_cross, T_obs) = smallest pseudo-N for a hypothetical new trial
        centred at T_obs that shifts the posterior mean past T_cross
        with probability ≥ 0.5

**Closed-form derivation.** Let the current posterior at **x** be N(μ, σ²) where μ = μ(**x**), σ² = σ²(**x**). Treat the current posterior as an effective sample of size *n_eff = σ²_trial / σ²* centred at μ. A hypothetical new trial with per-patient outcome variance σ²_trial observed at point estimate *T_obs* with sample size *n* updates the posterior mean to:

    μ_new = (μ · n_eff + T_obs · n) / (n_eff + n)

Setting μ_new = T_cross and solving for n:

    MCP = n_eff · (μ − T_cross) / (T_cross − T_obs)       [for T_obs ≠ T_cross]

**Two use-cases:**

- *Disconfirmation price*: T_obs = null (log-HR = 0). "How many pt-yrs of null-result trial would flip a current recommendation?"
- *Confirmation price*: T_obs = T_cross − δ (margin). "How many pt-yrs of further-benefit trial would firmly settle the recommendation?"

Reported as heatmaps in patient-years. Low price regions = fragile recommendations; high price regions = settled. Full implementation and edge cases (denominator → 0, μ already past threshold, sign conventions) in `dfs/mind_change.py`.

### 7.4 Feasibility region

    F(T) = { x : 95% CrI of f(x) excludes T }

If *F*(T) covers most of the HFpEF target population (say eGFR > 30, K⁺ < 5.2, LVEF ∈ [45, 65]), the recommendation is defensible without additional trials. If *F*(T) is narrow, more evidence is needed — and the mind-change price map says how much.

### 7.5 Conservation diagnostics (bonus)

If the solver cannot satisfy all hard constraints given the trial observations, it fails closed with a structured diagnostic:

    Trial TOPCAT-Americas: reported HR(ACM)=0.89, HR(CV)=0.83, HR(non-CV)=1.10.
    Baseline proportions p=0.58, q=0.42. Expected HR(ACM) ≈ p·0.83 + q·1.10 = 0.94.
    Discrepancy: 3.2σ. Likely causes: (a) transcription error, (b) differential
    follow-up, (c) outcome-specific censoring.

This is a data-integrity gift. The Finerenone/Cardiology Mortality Atlas project caught 4 Peto-method transcription errors by hand; DFS automates that class of catch.

## 8. Validation strategy

Four falsification tests, each a `pytest` unit test in `tests/`:

### 8.1 MA-equivalence limit

Strip all conservation laws and covariate information; feed DFS only (effect, SE) pairs with an intercept-only covariate. The fitted field value at the "average" covariate point must match a REML random-effects estimate (via `metafor` through `rpy2` or via `statsmodels`) to 1e-3 in log-HR space.

- **If matches:** MA is a degenerate limit of DFS. Publishable result in its own right.
- **If mismatches:** our math is wrong; fix before proceeding.

### 8.2 Conservation-violation detection

Synthetic trial set in which CV-death HR + non-CV-death HR (weighted) intentionally ≠ ACM HR by 2σ. DFS must flag the inconsistency with correct σ diagnosis. Standard MA would silently pool.

### 8.3 Dissonance resolution

Synthetic pair of trials with identical anchor covariates except adherence-proxy. DFS must:

- Produce distinct boundary conditions (not pool).
- Reconcile as the adherence-proxy length-scale shrinks (i.e., the field distinguishes them cleanly).
- Decompose their observed disagreement as adherence-covariate-explained with low residual.

Benchmark: standard MA would forest-plot them as heterogeneity.

### 8.4 Leave-one-trial-out on real data

Hold out FINEARTS-HF. Fit DFS on the other 5 trials. Predict FINEARTS-HF's observed HR at its anchor point. Report whether the observed HR is inside the predicted 95% CrI.

- **Inside CrI:** DFS generalises; report as supporting evidence.
- **Outside CrI:** structural miss; investigate and diagnose before publishing.

## 9. Non-goals

Explicit scope boundaries for the POC:

- **No IPD (individual patient data).** Trial-level boundary conditions only. IPD extension is a future paper; doubles engineering.
- **No publication-bias correction.** DFS has no built-in mechanism for missing-trial bias.
- **No NMA-style indirect comparison networks.** All trials share one anchor space; we do not build a network graph.
- **No continuous (streaming) updating.** POC is one-shot. CardioSynth-style living updates are phase-2.
- **No automated trial extraction.** Hand-curated from the Finerenone / Cardiology Mortality Atlas fixtures.
- **No class-vs-occupancy analysis.** That is clinical question 2; deferred to a second paper.
- **No methods-paper-first strategy.** POC is applied (target: *Circulation* or *JAMA Cardiology*). Methods paper follows once the machinery works end-to-end.
- **No reusable engine in POC.** The POC is single-domain; engine-isation (phase 3) happens only if the POC paper is accepted or placed.

## 10. Package structure

```
dissonance-field-synthesis/
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-15-dissonance-field-synthesis-design.md    (this file)
├── dfs/
│   ├── __init__.py
│   ├── schema.py               # TrialBoundaryCondition dataclass
│   ├── dissonance.py           # pairwise disagreement extraction + decomposition
│   ├── field.py                # constrained GP field fitter (hard + soft)
│   ├── conservation.py         # CONSERVATION_LAWS dict         [user-authored]
│   ├── adherence_proxy.py      # adherence_proxy() function     [user-authored]
│   ├── decisions.py            # DECISION_THRESHOLDS dict       [user-authored]
│   ├── mind_change.py          # mind-change price computation
│   ├── feasibility.py          # feasibility region computation
│   └── outputs.py              # plotting + report generation
├── data/
│   └── mra_hfpef/
│       ├── topcat_americas.json
│       ├── topcat_russia_georgia.json
│       ├── fineartshf.json
│       ├── fidelio_hf_subgroup.json
│       ├── figaro_hf_subgroup.json
│       └── aldo_dhf.json
├── tests/
│   ├── test_schema.py
│   ├── test_ma_equivalence.py
│   ├── test_conservation_detection.py
│   ├── test_dissonance_resolution.py
│   └── test_loo_fineartshf.py
├── scripts/
│   └── run_dfs.py              # end-to-end POC pipeline
├── .gitignore
├── pyproject.toml
└── README.md
```

## 11. User contribution files (learning-mode authorship moments)

Three 5–10 line files where clinical domain knowledge beats any code I could generate:

### `dfs/conservation.py`

Encodes the six laws from §6 as a `CONSERVATION_LAWS` dict. Each entry: `{name, type ("hard"|"soft"), penalty_weight, rationale}`. Author owns the hard/soft assignment and penalty weights based on clinical judgment.

### `dfs/adherence_proxy.py`

A single function `adherence_proxy(trial) -> float` in [0, 1]. Maps trial-reported features (placebo event rate vs expected rate, LTFU fraction, protocol adherence %) to a scalar. This is what splits TOPCAT into two boundary conditions and lets DFS explain the regional gap. Pure clinical/epidemiological judgment.

### `dfs/decisions.py`

A `DECISION_THRESHOLDS` dict: for each (endpoint, patient-severity) pair, the log-HR threshold that counts as "recommend," "borderline," "do not recommend." Feeds mind-change price and feasibility region. Prescribing heuristic made explicit.

## 12. Future work (explicit phase-2+ items)

- **Phase 2:** Class-vs-occupancy analysis (clinical question 1 from brainstorm). Uses the same engine.
- **Phase 3:** Promote POC codebase to a reusable `dfs` Python package, with JSON-schema-validated boundary-condition records, pluggable conservation-law library, and a CardioSynth-style living-update API.
- **Phase 4:** IPD-DFS extension.
- **Phase 5:** NMA-style multi-drug network extension.
- **Deferred indefinitely:** publication-bias correction (requires a separate theoretical development around missing-boundary-condition priors).

## 13. Risks and open questions

- **Risk: GP posterior is over-confident in regions far from any trial anchor.** Mitigation: ARD with weakly-informative length-scale priors; explicit CrI width reporting per region; feasibility region only trusted where density of anchor points is sufficient.
- **Risk: hard conservation constraints are infeasible given real trial data (transcription errors, differential follow-up).** Mitigation: conservation diagnostics output (§7.5) fails closed with structured explanation; fall-back to soft-constraint mode for production use.
- **Open question: how to handle time-to-event structure.** Current design treats log-HR as the primary unit. Does not currently use RMST differences or full survival curves. Future work if the POC demands it.
- **Open question: how to specify MR-occupancy-equivalent across steroidal and non-steroidal drugs.** Current plan: use published in vitro MR-binding affinity ratios (spironolactone 1.0×, eplerenone 0.5×, finerenone ~1.2× with different receptor kinetics). Authoritative source table to be assembled before POC run.
- **Open question: what is the target journal-specific statistical reporting standard.** Circulation demands reproducible code; JAMA Cardiology demands conservative presentation. Finalise before write-up.

## 14. Acceptance criteria for POC completion

The POC is considered complete when:

1. Validation tests §8.1 (MA-equivalence), §8.2 (conservation-violation detection), and §8.3 (dissonance resolution) pass in CI. These are pass/fail tests — failure means the math is wrong and must be fixed.
2. Validation test §8.4 (leave-one-trial-out) runs to completion and its result is **reported honestly** — observed FINEARTS-HF HR either inside or outside the predicted 95% CrI. Either outcome is a legitimate scientific finding; neither blocks completion. An "outside CrI" result is diagnosed (structural miss vs. statistical chance) and included in the discussion.
3. The end-to-end pipeline (`scripts/run_dfs.py`) runs cleanly on the six curated MRA-HFpEF trial files and produces all five outputs (§7.1–§7.5) as PNG + JSON + CSV artefacts in `outputs/`.
4. Conservation diagnostics either report zero violations on the real data, or flag specific violations with σ-level detail that the user can investigate against source (Finerenone Atlas).
5. User-authored files (`conservation.py`, `adherence_proxy.py`, `decisions.py`) are filled in, and results match the user's clinical expectation on the HFpEF target population (face-validity gate).
6. Design document (this file) and all code are committed and pushed to GitHub with a CC-licensed README pointing to reproducible artefacts.

Once (1)–(6) are satisfied, the project moves to manuscript drafting (target: *Circulation* or *JAMA Cardiology*), with the methods paper (phase 2) following once the applied paper is submitted.
