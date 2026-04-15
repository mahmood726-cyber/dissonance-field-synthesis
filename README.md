# Dissonance Field Synthesis (DFS) — POC

> A novel evidence-synthesis technique that replaces meta-analysis. This proof-of-concept applies DFS to mineralocorticoid-receptor antagonist (MRA) therapy in heart failure with preserved ejection fraction (HFpEF) across six trials.

## What DFS is

Meta-analysis silently imports three assumptions:
1. The unit of synthesis is the study.
2. The output is a pooled point estimate with an uncertainty interval.
3. Evidence lives on a number line.

DFS drops all three:

- **Primitive:** the unit is the *pairwise disagreement between trials*, not the trial itself.
- **Substrate:** effects live as a *field over covariate space*, constrained by *conservation laws* drawn from pharmacology and physiology.
- **Output:** instead of a pooled effect, DFS produces a dissonance map, a fitted effect field, a **mind-change price map**, a **feasibility region**, and **conservation diagnostics** — no pooled estimate is ever computed.

Design document: [`docs/superpowers/specs/2026-04-15-dissonance-field-synthesis-design.md`](docs/superpowers/specs/2026-04-15-dissonance-field-synthesis-design.md)
Implementation plan: [`docs/superpowers/plans/2026-04-15-dissonance-field-synthesis-poc.md`](docs/superpowers/plans/2026-04-15-dissonance-field-synthesis-poc.md)

## POC clinical question

**In HFpEF, which patients benefit from MRA therapy?**

Six trials form the boundary-condition set:

| Trial | Drug | Population | Source |
|---|---|---|---|
| TOPCAT-Americas | spironolactone | HFpEF | Pfeffer *Circulation* 2015 (PMID:25552772) + AACT NCT00094302 |
| TOPCAT-Russia/Georgia | spironolactone | HFpEF (regional split) | Pfeffer *Circulation* 2015 (PMID:25552772) |
| FINEARTS-HF | finerenone | HFpEF/HFmrEF | AACT NCT04435626 |
| FIDELIO-DKD HF-subgroup | finerenone | DKD with HF at baseline | AACT NCT02540993 |
| FIGARO-DKD HF-subgroup | finerenone | DKD with HF at baseline | AACT NCT02545049 |
| Aldo-DHF | spironolactone | HFpEF (small) | Edelmann *JAMA* 2013 (PMID:23440502) |

## Install and run

```
python -m pip install -e ".[dev]"
python -m pytest -v
python scripts/run_dfs.py --manifest data/mra_hfpef/MANIFEST.json --out outputs/
```

Outputs appear in `outputs/`:

- `dissonance.csv` — pairwise disagreement table (15 pairs) with covariate-delta columns
- `field_lvef_egfr.png` — posterior effect field over LVEF × eGFR with posterior SD
- `mind_change_price.csv` — pseudo-N of evidence needed to flip recommendation, per grid point
- `feasibility_mask.csv` — boolean map of defensible-without-more-trials regions
- `conservation_diagnostics.json` — flagged violations of mortality-decomposition law

## Current findings on real data

- **Conservation diagnostics:** zero violations across all 6 trials — the reported HR(ACM) values are consistent with p·HR(CV) + q·HR(non-CV) to within 2σ. Data integrity confirmed.
- **Highest dissonance pair:** TOPCAT-Americas vs TOPCAT-Russia/Georgia, d = 1.56. DFS correctly identifies the regional-adherence signal as the dominant disagreement in the evidence base — a signal standard meta-analysis silently pools away.
- **Leave-one-trial-out (FINEARTS-HF):** observed log-HR (−0.174) lies inside the predicted 95% CrI [−0.990, +0.967] from a GP trained on the other five trials.

## Scope boundaries

POC, not a reusable engine. Explicit non-goals:
- No individual-patient data (trial-level boundary conditions only)
- No publication-bias correction
- No NMA-style indirect comparison network
- No streaming/continuous updating
- No class-vs-occupancy analysis (deferred to follow-up paper)

The `dfs/` package is internally organised for POC clarity, not external reuse. Engine-isation is phase-3.

## Three files a clinician owns

Three small modules encode clinical judgment and are intended for revision:

- [`dfs/conservation.py`](dfs/conservation.py) — hard/soft assignment and penalty weights for the six pharmacology conservation laws (spec §6)
- [`dfs/adherence_proxy.py`](dfs/adherence_proxy.py) — maps trial design features to a scalar adherence score (load-bearing for splitting TOPCAT into Americas/Russia)
- [`dfs/decisions.py`](dfs/decisions.py) — log-HR prescribing thresholds per endpoint and per-patient variance estimates for mind-change pricing

Defaults are committed; revisit them against your clinical judgment.

## Reproducibility

- Trial-level data curated from AACT 2026-04-12 snapshot + primary publications. Every outcome record carries a `source` field with PMID or AACT `outcome_id`.
- Test suite: 78 passing, 1 intentional skip (combinatorial counting test deferred to integration contract). Four falsification tests: MA-equivalence limit, conservation-violation detection, dissonance resolution, leave-one-out.
- Deterministic across seeds 0, 1, 42 for the MA-equivalence limit test.

## License

Code: MIT. Documentation and figures: CC-BY-4.0.

## Status

v0.1.0-poc. Applied target: *Circulation* or *JAMA Cardiology*. Methods paper (*Statistics in Medicine*) follows once the POC is accepted or placed.
