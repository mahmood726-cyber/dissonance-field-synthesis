# E156 Protocol — `dissonance-field-synthesis`

This repository is the source code and dashboard backing an E156 micro-paper on the [E156 Student Board](https://mahmood726-cyber.github.io/e156/students.html).

---

## `[483]` Dissonance Field Synthesis: A Non-Pooling Evidence-Synthesis Technique with Pharmacology Conservation Laws (POC in MRA-HFpEF)

**Type:** methods  |  ESTIMAND: leave-one-out predicted log-hazard-ratio for FINEARTS-HF  
**Data:** Six MRA-HFpEF boundary-condition records (TOPCAT-Americas/Russia split, FINEARTS-HF, FIDELIO/FIGARO HF-subgroups, Aldo-DHF) from AACT 2026-04-12 and primary publications

### 156-word body

Can a synthesis method that treats pairwise trial disagreements as the primitive, without computing a pooled estimate, predict held-out MRA-HFpEF hazard ratios? Six trial boundary-condition records — TOPCAT-Americas, TOPCAT-Russia/Georgia, FINEARTS-HF, FIDELIO-DKD HF-subgroup, FIGARO-DKD HF-subgroup, Aldo-DHF — sourced from AACT 2026-04-12 plus primary publications. A constrained Gaussian process on a seven-covariate space with six pharmacology conservation laws and an ARD Matérn 5/2 kernel. Leave-one-out on FINEARTS-HF: a GP trained on the other five trials predicted log hazard ratio -0.012, ninety-five percent credible interval -0.990 to +0.967, observed -0.174 lies inside. Intercept-only fit matched inverse-variance pooling to within 1e-3 across three seeds; conservation diagnostics flagged zero mortality-decomposition violations across the six trials. The dominant pairwise dissonance was TOPCAT-Americas versus TOPCAT-Russia/Georgia (d=1.56), entirely attributable to the adherence-proxy covariate delta — a regional-adherence signal meta-analysis pools away. This proof-of-concept excludes individual patient data, publication-bias correction, and indirect comparisons; conservation-law assignments and decision thresholds remain clinician-authored defaults.

### Submission metadata

```
Corresponding author: Mahmood Ahmad <mahmood.ahmad2@nhs.net>
ORCID: 0000-0001-9107-3704
Affiliation: Tahir Heart Institute, Rabwah, Pakistan

Links:
  Code:      https://github.com/mahmood726-cyber/dissonance-field-synthesis
  Protocol:  https://github.com/mahmood726-cyber/dissonance-field-synthesis/blob/main/E156-PROTOCOL.md
  Dashboard: https://mahmood726-cyber.github.io/dissonance-field-synthesis/

References (topic pack: network meta-analysis):
  1. Rücker G. 2012. Network meta-analysis, electrical networks and graph theory. Res Synth Methods. 3(4):312-324. doi:10.1002/jrsm.1058
  2. Lu G, Ades AE. 2006. Assessing evidence inconsistency in mixed treatment comparisons. J Am Stat Assoc. 101(474):447-459. doi:10.1198/016214505000001302

Data availability: No patient-level data used. Analysis derived exclusively
  from publicly available aggregate records. All source identifiers are in
  the protocol document linked above.

Ethics: Not required. Study uses only publicly available aggregate data; no
  human participants; no patient-identifiable information; no individual-
  participant data. No institutional review board approval sought or required
  under standard research-ethics guidelines for secondary methodological
  research on published literature.

Funding: None.

Competing interests: MA serves on the editorial board of Synthēsis (the
  target journal); MA had no role in editorial decisions on this
  manuscript, which was handled by an independent editor of the journal.

Author contributions (CRediT):
  [STUDENT REWRITER, first author] — Writing – original draft, Writing –
    review & editing, Validation.
  [SUPERVISING FACULTY, last/senior author] — Supervision, Validation,
    Writing – review & editing.
  Mahmood Ahmad (middle author, NOT first or last) — Conceptualization,
    Methodology, Software, Data curation, Formal analysis, Resources.

AI disclosure: Computational tooling (including AI-assisted coding via
  Claude Code [Anthropic]) was used to develop analysis scripts and assist
  with data extraction. The final manuscript was human-written, reviewed,
  and approved by the author; the submitted text is not AI-generated. All
  quantitative claims were verified against source data; cross-validation
  was performed where applicable. The author retains full responsibility for
  the final content.

Preprint: Not preprinted.

Reporting checklist: PRISMA 2020 (methods-paper variant — reports on review corpus).

Target journal: ◆ Synthēsis (https://www.synthesis-medicine.org/index.php/journal)
  Section: Methods Note — submit the 156-word E156 body verbatim as the main text.
  The journal caps main text at ≤400 words; E156's 156-word, 7-sentence
  contract sits well inside that ceiling. Do NOT pad to 400 — the
  micro-paper length is the point of the format.

Manuscript license: CC-BY-4.0.
Code license: MIT.

SUBMITTED: [ ]
```


---

_Auto-generated from the workbook by `C:/E156/scripts/create_missing_protocols.py`. If something is wrong, edit `rewrite-workbook.txt` and re-run the script — it will overwrite this file via the GitHub API._