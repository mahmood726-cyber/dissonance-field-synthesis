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
