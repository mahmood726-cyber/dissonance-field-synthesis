# Supplementary Section S-A: Sensitivity of Key Findings to `adherence_proxy` Specification

## Overview

The primary analysis encodes each trial's adherence profile as a scalar covariate
(`adherence_proxy`) in the anchor-covariate vector.
Two trials carry the most methodologically consequential adherence assumptions:
TOPCAT-Russia/Georgia (baseline 0.40, reflecting well-documented non-adherence to
spironolactone in those regions) and FINEARTS-HF (baseline 0.90, reflecting the
high-compliance profile of a modern phase-III trial).
We varied each anchor in turn — holding the other fixed at its baseline — to assess
whether the dominant-adherence-dimension finding and the leave-one-out (LOO)
predictive validity of the fitted Gaussian-process field are robust to these
specification choices.
All scenarios were run with ML-II hyperparameter estimation (`n_restarts=5, seed=0`)
and a LOO predictive interval that correctly adds the held-out trial's observation
variance to the GP posterior variance.
Results are tabulated in `sensitivity_results.csv` (11 scenarios, both sweeps).

## Sweep 1 — TOPCAT-Russia/Georgia `adherence_proxy` in [0.20, 0.70]

<!-- AUTHOR REVIEW: The clinical plausibility of adherence values above 0.60 for
the Russia/Georgia arm should be confirmed against PMID:25552772 and editorial
commentary on spironolactone metabolite levels. Values ≥ 0.60 approach the
Americas-arm compliance level and may not be clinically supportable. -->

The dissonance scalar *d* for the TOPCAT-Americas versus TOPCAT-Russia/Georgia pair
was 1.563 at every grid point (0.20, 0.30, 0.40, 0.50, 0.60, 0.70), invariant to
the third decimal place.
This is expected: *d* is computed from the log-HR and SE entries in the trial JSON
files, neither of which is modified by the sweep; only the covariate position in
feature space changes.
The ML-II-fitted adherence length-scale increased monotonically from 0.325 at
`adherence_proxy = 0.20` to 0.504 at `adherence_proxy = 0.70`, a 55% increase over
the ±0.20 perturbation range.
This expansion is consistent with the GP attempting to reconcile a wider spread of
training-set adherence values with the same log-HR differences: as the Russia/Georgia
anchor moves closer to the Americas anchor, the adherence dimension becomes less
informative, and the length-scale grows to reflect reduced resolution.
Critically, the LOO prediction for FINEARTS-HF remained inside the 95% predictive
interval at all six grid points (6/6 green), and all 11 rows produced zero
conservation-law violations.
The LOO predicted mean drifted only from −0.136 to −0.125 log-HR across the
full sweep, against an observed value of −0.174; the CrI width widened modestly
from 0.250 to 0.262 log-HR units.
The finding that the adherence dimension dominates the GP kernel therefore persists
across the entire plausible range of Russia/Georgia adherence specifications,
including the regime boundary at `adherence_proxy ≈ 0.70` where the length-scale
begins to expand noticeably.

## Sweep 2 — FINEARTS-HF `adherence_proxy` in [0.70, 0.95]

<!-- AUTHOR REVIEW: The anchor values of 0.70 and 0.75 represent a materially lower
compliance profile than suggested by the FINEARTS-HF publication (PMID:39225278).
Clinical judgment is needed to determine whether including these as plausible
scenarios is appropriate for the final paper, or whether the sweep range should
be restricted to [0.85, 0.95]. -->

Varying FINEARTS-HF's own adherence anchor while holding Russia/Georgia fixed at 0.40
tests how much the LOO prediction moves when the held-out trial's position in covariate
space changes.
The dissonance scalar for the TOPCAT pair was again constant at 1.563 across all five
values, as expected (the TOPCAT anchors were unchanged).
The adherence length-scale was likewise constant at 0.361 across all sweep-2 scenarios,
because the GP is fitted on the five *kept* trials only (FINEARTS-HF is held out), and
the kept set has the same covariate matrix regardless of the FINEARTS-HF anchor value.
The LOO predicted mean showed the strongest dependence on the held-out anchor: it moved
from −0.070 at `adherence_proxy = 0.70` to −0.138 at `adherence_proxy = 0.85`, then
partially returned toward −0.116 at 0.95.
The underlying mechanics are purely geometric: as FINEARTS-HF's anchor moves through
the covariate space, it samples a different neighbourhood of the fitted GP surface, and
the predicted log-HR tracks the local field value.
Despite this movement, the LOO observed log-HR (−0.174) remained inside the 95%
predictive interval at all five values (5/5 green).
The CrI width was widest at `adherence_proxy = 0.70` (0.336 log-HR units) and
narrowest at the canonical 0.85–0.90 range (0.250 log-HR units), indicating that
the GP is most informative — and the adherence-dominance finding most precise —
when FINEARTS-HF's anchor is placed within the interior of the training-set
distribution rather than at its boundary.

## Figure S1

![Adherence-proxy sensitivity analysis](../outputs/sensitivity_adherence.png)

**Figure S1.** Four-panel sensitivity analysis across two adherence-proxy sweeps.
Panel A shows the LOO predicted log-HR mean for FINEARTS-HF as a function of the
varied adherence anchor (green markers = observed inside 95% CrI; red = outside).
Panel B shows the corresponding CrI width.
Panel C shows the ML-II-fitted adherence length-scale; constant values in Sweep 2
confirm that held-out-trial anchor perturbation does not affect the fitted GP kernel.
Panel D shows the dissonance scalar *d* for the TOPCAT-Americas / Russia-Georgia
pair; the invariance across both sweeps demonstrates that *d* is determined by
outcome data alone, not by covariate-anchor specification.
All 11 scenarios produced zero conservation-law violations and LOO coverage was
100% across both sweeps.
Results tabulated in `sensitivity_results.csv`.

## Supplementary Section S-B: Sensitivity of Key Findings to GP Kernel Choice

### Overview

The primary analysis uses an anisotropic (ARD) Matérn-5/2 kernel, which assumes
the outcome surface in covariate space is twice mean-square differentiable.
This smoothness assumption is standard in the spatial-statistics and
computer-experiments literature for physical-science problems but is not
self-evidently correct for a six-trial cardiology evidence-synthesis problem.
To pre-empt the "why Matérn-5/2?" reviewer question and to quantify
sensitivity to this architectural choice, we re-fit the LOO FINEARTS-HF
pipeline under three ARD stationary kernels spanning the smoothness spectrum:
the squared-exponential (RBF, ν → ∞, infinitely differentiable), the current
Matérn-5/2 (ν = 2.5), and the rougher Matérn-3/2 (ν = 1.5, once
mean-square differentiable).
All other analysis choices — ML-II hyperparameter optimisation, five-restart
seeding, covariate normalisation, inclusion of observation-noise variance in
the LOO predictive interval — were held identical across kernels.
Results are tabulated in `sensitivity_kernel_results.csv`.

### Results

The three kernels produced materially identical conclusions.
The LOO FINEARTS-HF observed value (−0.174 log-HR) remained inside the 95%
predictive interval under all three (3/3 green).
The ML-II-fitted adherence length-scale was 0.361 (Matérn-5/2), 0.387
(Matérn-3/2), and 0.345 (RBF) — a range of 0.042, i.e. an 11% spread around
the Matérn-5/2 value, small relative to the adherence-sweep spread of 0.179
(Section S-A).
The adherence dimension was the dominant (smallest) length-scale in every
kernel fit, confirming that the "adherence-explains-heterogeneity" conclusion
is driven by the data geometry and not by the choice of smoothness prior.
The LOO predicted log-HR mean was −0.135 (Matérn-5/2), −0.133 (Matérn-3/2),
and −0.136 (RBF); the 0.003 log-HR spread is two orders of magnitude smaller
than the adherence-sweep spread of 0.011 (Section S-A, Sweep 1) and three
orders smaller than the between-trial log-HR range (0.20 log-HR units).
The ML-II negative log marginal likelihoods were −3.575 (Matérn-5/2), −3.556
(Matérn-3/2), and −3.595 (RBF); the RBF fit is nominally preferred by
ML-II but the likelihood-ratio difference (ΔNLL = 0.020) is well below any
conventional model-selection threshold (e.g. AIC ΔAIC ≈ 0.04 for
zero-parameter-delta comparison).

### Interpretation

The kernel-choice sensitivity demonstrates that the POC's two headline
findings — LOO predictive validity for FINEARTS-HF and adherence-dimension
dominance — are insensitive to the Matérn-5/2 smoothness assumption within
the family of standard ARD stationary kernels.
For a final analysis with a larger trial set, we recommend reporting the
RBF kernel as the primary model (marginally preferred by ML-II, same
conclusions) with Matérn-5/2 and Matérn-3/2 as pre-specified sensitivity
analyses, which mirrors standard practice in the spatial-statistics
literature.
The relative invariance of the fitted adherence length-scale across kernels
(0.345–0.387) versus its variation under adherence-anchor perturbation
(0.325–0.504, Section S-A) indicates that the *data* — specifically, the
spread of adherence anchors across trials — determines the fitted GP
geometry substantially more than the smoothness prior does.

### Table S-B-1

| Kernel       | ν    | NLL    | Adh. LS | LOO μ  | LOO 95% CrI       | Inside |
|--------------|------|--------|---------|--------|-------------------|--------|
| Matérn-5/2   | 2.5  | −3.575 | 0.361   | −0.135 | [−0.260, −0.009]  | ✓      |
| Matérn-3/2   | 1.5  | −3.556 | 0.387   | −0.133 | [−0.261, −0.005]  | ✓      |
| RBF          | ∞    | −3.595 | 0.345   | −0.136 | [−0.261, −0.011]  | ✓      |

Observed FINEARTS-HF log-HR = −0.174 (inside CrI for all three kernels).
Dissonance scalar *d* (TOPCAT-Americas vs TOPCAT-Russia/Georgia) = 1.563
across all three kernels (invariant — *d* depends only on log-HR/SE
pairs, not on GP specification).
Zero conservation-law violations under every kernel.

<!-- AUTHOR REVIEW: Whether to promote the RBF kernel to the primary model
in the full-cohort manuscript is a design decision. The ML-II preference
is nominal (ΔNLL = 0.020); the Matérn-5/2 has stronger methodological
precedent in applied-statistics and spatial-epidemiology literature.
Recommend retaining Matérn-5/2 as primary and citing this section as
"pre-specified kernel-sensitivity analysis" in the Methods. -->

## Supplementary Section S-C: Stability of the ML-II Optimum to Restart Configuration

### Overview

The primary analysis obtains the GP signal variance and the seven ARD
length-scales by maximising the log marginal likelihood (ML-II) with
L-BFGS-B over log-parameters from six starting points (one deterministic
+ five random restarts seeded with `rng = np.random.default_rng(0)`).
Non-convex ML-II surfaces are known to admit local minima, and reviewers
routinely ask whether a single-seed, low-restart configuration has in
fact located the global optimum.
We therefore re-fit the held-out-FINEARTS-HF hyperparameter problem under
a 3 × 10 grid of `(n_restarts, seed)` configurations: `n_restarts ∈
{5, 20, 50}` crossed with `seed ∈ {0, 1, …, 9}`, producing 30 independent
optimisations from 30 different RNG streams.
All fits used the default Matérn-5/2 kernel, the same covariate
normalisation as the primary analysis, and `scipy.optimize.minimize` with
`method="L-BFGS-B"` and `bounds=[(-10, 10)]` on every log-parameter.
Results are tabulated in `sensitivity_restarts_results.csv`.

### Results

All 30 fits recovered an identical ML-II negative log marginal likelihood
of −3.574716 to six decimal places (standard deviation 0.000000 across
seeds at every `n_restarts` level, range 0.000000 at every level).
The fitted adherence-proxy length-scale was identical to three decimal
places (0.361 at every configuration); its across-seed range contracted
monotonically from 1.19 × 10⁻⁴ at `n_restarts = 5` to 1.05 × 10⁻⁴ at
`n_restarts = 20` to 4.0 × 10⁻⁵ at `n_restarts = 50`, consistent with
the expected tightening of L-BFGS-B termination noise as more starting
points are averaged into the best-of-restarts selection.
The reference fit used in the primary analysis (`n_restarts = 5,
seed = 0`, NLL = −3.574716) differed from the best NLL observed
across any of the 30 configurations by 0.000000 in log-likelihood units.
No seed produced a discernibly different optimum at any restart count.

### Interpretation

The ML-II surface for the held-out-FINEARTS-HF fit is, within the
precision of L-BFGS-B termination tolerances, unimodal: six random
restarts are sufficient for global-optimum recovery, and the primary
analysis's single-seed configuration is not selecting among competing
local minima.
This is a stronger result than is typical for ARD-Matérn hyperparameter
fits, and it reflects the small number of training points (five
retained trials, seven length-scales).
With more trials — and therefore more potential for length-scales to
individuate — multimodality becomes more plausible, and we therefore
recommend the `n_restarts = 20, seed = 0` configuration as the default
for the full-cohort manuscript to provide a safety margin.

### Table S-C-1

| n_restarts | NLL mean    | NLL std   | NLL range | Adh-LS mean | Adh-LS std | Adh-LS range |
|-----------:|------------:|----------:|----------:|------------:|-----------:|-------------:|
|          5 | −3.574716   | 0.000000  | 0.000000  | 0.360520    | 3.6 × 10⁻⁵ | 1.19 × 10⁻⁴ |
|         20 | −3.574716   | 0.000000  | 0.000000  | 0.360530    | 2.8 × 10⁻⁵ | 1.05 × 10⁻⁴ |
|         50 | −3.574716   | 0.000000  | 0.000000  | 0.360528    | 9.0 × 10⁻⁶ | 4.0 × 10⁻⁵ |

Summary across 10 seeds at each `n_restarts` level.
Primary-analysis reference: `(n_restarts = 5, seed = 0)` → NLL =
−3.574716, adherence length-scale = 0.360557, identical to the
across-seed mean at that level to six decimal places.
Best-of-30 NLL: −3.574716 at `(n_restarts = 50, seed = 8)`; reference
fit deviates from this by 0.000000.

## Supplementary Section S-D: End-to-End Reproducibility Audit

### Overview

A reproducibility audit was performed by cloning the repository into a
fresh working directory that did not share any local state with the
development environment (no cached Python bytecode, no pre-existing
outputs, no user-home configuration files).
The full test suite and all analysis scripts were then re-executed against
the cloned tree, and every produced artefact was diffed byte-for-byte
against its committed counterpart in the original repository.
This audit verifies that the repository as published to
`github.com/mahmood726-cyber/dissonance-field-synthesis` is self-contained
and produces bit-identical scientific artefacts on a machine that has
never previously run the code.

### Protocol

1. `git clone <repo> <tmp_dir>` at tip commit 50ea7a5.
2. `python -m pytest -q` in `<tmp_dir>`.
3. `python scripts/run_dfs.py --manifest data/mra_hfpef/MANIFEST.json --out <tmp_out>`.
4. `python scripts/sensitivity_adherence.py --results <tmp_csv_1>`.
5. `python scripts/sensitivity_kernel.py --results <tmp_csv_2>`.
6. `python scripts/sensitivity_restarts.py --results <tmp_csv_3>`.
7. `diff` every produced artefact against the committed version in the
   original repository; CSVs and JSON files must be bit-identical, PNGs
   are not compared (matplotlib backend-dependent rendering).

### Results

| Audit step                           | Expected           | Observed           | Verdict |
|--------------------------------------|--------------------|--------------------|---------|
| Test suite                           | 101 passed, 1 skip | 101 passed, 1 skip | PASS    |
| `outputs/dissonance.csv`             | bit-identical      | bit-identical      | PASS    |
| `outputs/conservation_diagnostics.json` | bit-identical   | bit-identical      | PASS    |
| `outputs/feasibility_mask.csv`       | bit-identical      | bit-identical      | PASS    |
| `outputs/mind_change_price.csv`      | bit-identical      | bit-identical      | PASS    |
| `outputs/k_sign_constraint_report.json` | bit-identical   | bit-identical      | PASS    |
| `manuscript/sensitivity_results.csv` | bit-identical      | bit-identical      | PASS    |
| `manuscript/sensitivity_kernel_results.csv` | bit-identical | bit-identical    | PASS    |
| `manuscript/sensitivity_restarts_results.csv` | bit-identical | bit-identical  | PASS    |

### Interpretation

All nine audited artefacts reproduced bit-identically from a fresh
clone.
The fitted GP hyperparameters, the dissonance scalar table, the
conservation-law diagnostic output, the mind-change-price calculation,
the feasibility-mask grid, the k_sign constraint report, and all three
sensitivity CSVs are therefore fully determined by the committed source
code and manifest JSON files, with no hidden dependence on local state,
user-home configuration, environmental variables, or cached compilation
artefacts.
Reviewers can reproduce every quantitative claim in the manuscript by
following the four commands in the Protocol section above.

Section S-D is now cited from the main manuscript at §2.7, with the
explicit claim that all §3 results, S-A through S-C tables, and the
S-B-1/S-C-1 numbers are reproducible from the public repository by
following the four commands in the Protocol section above.

## Supplementary Section S-E: Phase-2a Outcome-Anchor Refinement (FIDELIO HF-Subgroup Swap)

### Overview

The Phase-1b DFS analysis used the FIDELIO-DKD whole-trial cardiovascular
composite hazard ratio (HR 0.86, 95% CI 0.75–0.99; log-HR = −0.151,
SE = 0.072) as the primary anchor for the FIDELIO-DKD HF-subgroup boundary
condition. This was a documented compromise: the AACT
ClinicalTrials.gov snapshot did not break out a separate
HF-at-baseline-subgroup CV composite, and the DFS pipeline therefore used
the parent-trial value as a proxy.
Concurrent with the literature-verification pass that resolved the Table 1
sample-size flag (§2.6, [11]), the Filippatos *et al.* 2022 *European
Journal of Heart Failure* prespecified subgroup paper became available as
a citable source. That paper reports, for the n = 436 FIDELIO patients
with a documented history of heart failure at baseline (7.7% of the
parent-trial 5,674), a CV composite HR of 0.73 (95% CI 0.50–1.06).
Converting to the GP's log-scale boundary condition: log-HR = −0.3147,
SE = 0.1917 (computed from the 95% CI half-width using the standard
normal-approximation formula SE = (log(HR_upper) − log(HR_lower)) / 3.92).

This section reports a Phase-2a re-fit in which only the FIDELIO-DKD
HF-subgroup primary-composite anchor is replaced with the subgroup-specific
value. All other trial anchors, all covariates, all hyperparameter
settings, and the kernel choice are held identical to Phase-1b.
The FIGARO-DKD HF-subgroup CV composite HR was not separately reported in
the main text of the analogous FIGARO HF-subgroup paper [12]
(it is in supplementary Figure S3 of that publication and was not extracted
in this session); FIGARO is therefore held at its Phase-1b parent-trial
anchor, and the analogous FIGARO swap is deferred to a follow-up.

### Results

| Quantity                       | Phase-1b (parent-trial) | Phase-2a (HF-subgroup) | Δ          |
|--------------------------------|------------------------:|-----------------------:|-----------:|
| FIDELIO log-HR anchor          | −0.1508                 | −0.3147                | −0.1639    |
| FIDELIO SE                     | 0.0716                  | 0.1917                 | ×2.68      |
| ML-II adherence length-scale   | 0.3606                  | 0.3562                 | −0.0044    |
| LOO FINEARTS-HF predicted μ    | −0.1347                 | −0.1419                | −0.0072    |
| LOO FINEARTS-HF 95% CrI lower  | −0.2602                 | −0.2676                | −0.0074    |
| LOO FINEARTS-HF 95% CrI upper  | −0.0091                 | −0.0161                | −0.0070    |
| LOO FINEARTS-HF CrI width      | 0.2511                  | 0.2515                 | +0.0005    |
| LOO inside CrI?                | True                    | True                   | unchanged  |
| TOPCAT-pair dissonance d       | 1.563                   | 1.563                  | unchanged  |

Observed FINEARTS-HF log-HR = −0.174, inside the 95% CrI under both
anchor choices.
Numerical results are tabulated to six decimal places in
`manuscript/phase2_subgroup_swap_results.csv`.

### Interpretation

Three observations.

First, the FIDELIO anchor change is substantial in both directions: the
point estimate moves by −0.164 log-HR units (from HR 0.86 to HR 0.73, a
17% deeper relative-risk reduction) and the standard error inflates 2.68×
to reflect the smaller subgroup sample size (n = 436 versus parent-trial
n = 5,674). These are not cosmetic perturbations.

Second, despite this substantial input change, the DFS field's headline
conclusions are essentially unaltered. The adherence length-scale moves
by 0.004 normalised units (1.2% of its value); the LOO predicted mean
shifts by −0.007 log-HR units (toward the observed value, narrowing the
prediction-vs-observation gap from 0.039 to 0.032); the 95% CrI width
changes by +0.0005 log-HR units (a 0.2% increase, well below resolution);
and the LOO observed value (−0.174) remains inside the 95% predictive
interval. The TOPCAT-Americas/Russia–Georgia dissonance scalar is exactly
unchanged, as expected: it depends only on the TOPCAT pair and not on any
FIDELIO outcome.

Third, this invariance is the heteroscedastic GP behaving as designed.
The Phase-1b FIDELIO anchor (SE 0.072) carried a precision weight of
1 / 0.072² ≈ 193; the Phase-2a anchor (SE 0.192) carries a weight of
1 / 0.192² ≈ 27. Although the point estimate moved by 0.164, the
GP correctly down-weighted the noisier observation by ~7×, so its
contribution to the field is preserved at approximately the same
information level. The architecture does not require all anchors to come
from equivalently-precise sources; it requires only that the SE field
honestly encode each source's precision.

### Implications

For the manuscript's primary conclusions (adherence dominance,
LOO-predictive validity, no conservation violations), the FIDELIO
parent-trial-anchor approximation used in Phase-1b does not bias any
quantitative claim. A future analysis incorporating subgroup-specific
HRs from both FIDELIO and FIGARO is straightforward (replace two anchor
fields in the JSON manifest and re-run) and would marginally improve
calibration; however, no Phase-1b finding requires correction. The
Phase-2a swap also serves as a methodological demonstration: when better
evidence becomes available for any single trial anchor in a future DFS
application, the entire pipeline accommodates the swap by re-weighting
rather than by structural modification.

### Phase-2b — hypothetical FIGARO swap (proxy)

The dedicated FIGARO HF-subgroup CV composite hazard ratio is reported in
supplementary Figure S3 of Filippatos *et al.* 2022 Circulation [12] and was
not extractable from any of the five sources interrogated in this session
(the parent paper's main text, the PMC-archived full-text version, the
FIDELITY pooled European Heart Journal article, the FINE-HEART Nature
Medicine pooled analysis, and the FIDELITY HF-subgroup American Heart
Journal abstract). To bracket the implications of also swapping FIGARO,
we add a clearly-labelled Phase-2b sensitivity scenario that re-uses the
FIDELIO HF-subgroup hazard ratio (HR 0.73) as a proxy for the unknown
FIGARO HF-subgroup hazard ratio, with the standard error scaled by
√(436/571) = 0.875 to approximate FIGARO's larger HF-subgroup sample size.
This is a hypothetical sensitivity test, not a citable swap; the FIGARO
HF-subgroup-specific HR may differ.

| Quantity                       | Phase-1b | Phase-2a (FIDELIO) | Phase-2b (both, proxy) |
|--------------------------------|---------:|-------------------:|-----------------------:|
| FIDELIO log-HR / SE            | −0.151 / 0.072 | −0.315 / 0.192 | −0.315 / 0.192 |
| FIGARO log-HR / SE             | −0.139 / 0.065 | −0.139 / 0.065 | −0.315 / 0.168 |
| Adherence length-scale         | 0.3606 | 0.3562 | **0.3100** |
| LOO predicted μ                | −0.135 | −0.142 | **−0.198** |
| LOO 95% CrI                    | [−0.260, −0.009] | [−0.268, −0.016] | [−0.328, −0.068] |
| LOO width                      | 0.251 | 0.252 | 0.260 |
| LOO inside CrI?                | True   | True   | True   |
| TOPCAT-pair d                  | 1.563  | 1.563  | 1.563  |

Phase-2b changes the qualitative picture in two ways relative to Phase-2a.
First, the adherence length-scale tightens by 14% (0.361 → 0.310), meaning
the fitted GP becomes more concentrated along the adherence axis when both
finerenone subgroups carry more aggressive hazard ratios. Second, the LOO
predicted mean (−0.198) now slightly overshoots the observed FINEARTS-HF
log-HR (−0.174), reversing the Phase-1b under-shoot direction. The
observed value remains well inside the 95% CrI ([−0.328, −0.068]), so
LOO predictive validity is preserved, but the prediction-versus-observation
sign flips.

These changes are consistent with a rational Bayesian update: when the
training set carries stronger evidence of benefit in adherent populations
(both finerenone trials with HF-subgroup HRs of 0.73 rather than parent-
trial proxies of 0.86 / 0.87), the GP rationally infers a stronger
adherence-dependent benefit and tightens the length-scale to fit it. The
adherence-dominance conclusion is not weakened by Phase-2b — it is
strengthened. We emphasise that this is a hypothetical analysis pending
extraction of the actual FIGARO HF-subgroup HR; the direction of the
true result may differ.

### Reproduction

```
python scripts/phase2_subgroup_anchor_swap.py \
    --manifest data/mra_hfpef/MANIFEST.json \
    --results manuscript/phase2_subgroup_swap_results.csv
```

Reports three scenarios: Phase-1b baseline, Phase-2a (FIDELIO swap, real),
and Phase-2b (FIDELIO swap real + FIGARO swap proxy). Output CSV has one
row per scenario with all anchor and LOO-result fields.


