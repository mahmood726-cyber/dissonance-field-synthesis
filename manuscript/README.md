# Manuscript Directory

Applied paper targeting *Circulation* or *JAMA Cardiology*.

**Status:** First-pass draft. All quantitative claims are traced to
`data/mra_hfpef/`, `outputs/`, or test-suite output. Author-review flags
are HTML comments (`<!-- AUTHOR REVIEW -->`).

---

## Contents

| File | Purpose |
|---|---|
| `main.md` | ~3,000-word primary draft (abstract + full text + tables) |
| `methods_supplement.md` | ~2,000-word supplementary extended methods |
| `figures.md` | Figure list with captions and artefact mapping |
| `references.bib` | BibTeX bibliography with 9 primary entries |
| `README.md` | This file |

---

## Word Counts

Current word counts (run `wc -w main.md methods_supplement.md` from this
directory to verify):

| File | Target | Current (approx) |
|---|---|---|
| `main.md` | 2,500–3,500 | ~3,100 |
| `methods_supplement.md` | 1,500–2,500 | ~2,000 |

---

## Journal Word Limits

| Journal | Max main text | Max abstract | Max references |
|---|---|---|---|
| *Circulation* (Original Research) | 4,500 | 250 | 50 |
| *JAMA Cardiology* (Original Investigation) | 3,000 | 300 | 50 |

The current draft targets *Circulation* at approximately 3,100 main-text words
(excluding abstract, tables, figure captions, and references). This leaves
~1,400 words of headroom within the *Circulation* limit and is approximately
at the *JAMA Cardiology* limit. Mahmood should decide on target journal before
final editing; JAMA Cardiology may require trimming the Discussion by ~100–200
words.

---

## How to Compile to PDF (Pandoc)

```bash
# Install Pandoc and a LaTeX engine if not present
# From the manuscript/ directory:

pandoc main.md methods_supplement.md \
    --bibliography=references.bib \
    --csl=circulation.csl \       # download from Zotero style repo
    -o draft_submission.pdf \
    --pdf-engine=xelatex \
    --variable geometry:margin=2.5cm
```

For a Word file (journal submission portals often prefer .docx):

```bash
pandoc main.md \
    --bibliography=references.bib \
    --csl=circulation.csl \
    -o draft_submission.docx
```

CSL files are available at https://github.com/citation-style-language/styles.
The Circulation CSL identifier is `circulation`. JAMA Cardiology uses the
`jama` CSL.

---

## Figure Generation

Figures 2 and 5 are produced directly by `scripts/run_dfs.py`:

```bash
python scripts/run_dfs.py \
    --manifest data/mra_hfpef/MANIFEST.json \
    --out outputs/
```

Figures 1, 3, 4, and 6 require additional rendering code in a planned script
`scripts/render_figures.py` (TODO — not yet implemented). That script should:

1. Load `outputs/dissonance.csv` and render Figure 1 (6×6 heatmap).
2. Load `outputs/mind_change_price.csv` and render Figure 3 (LVEF × eGFR
   heatmap of MCP values).
3. Load `outputs/feasibility_mask.csv` and render Figure 4 (Boolean overlay).
4. Optionally, render Figure 6 (forest plot with DFS LOO prediction).

---

## Next Steps for Mahmood

**Immediate (before revision):**

1. Review all `<!-- AUTHOR REVIEW -->` comments in `main.md` and
   `methods_supplement.md`. There are 10 flagged passages — see the full list
   in the commissioning report.

2. Confirm the TOPCAT split pre-specification rationale is acceptable to the
   target journal.

3. Verify the Aldo-DHF placeholder (log-HR 0.0, wide CI) is disclosed
   appropriately; consider a sensitivity analysis excluding Aldo-DHF.

4. Confirm prescribing thresholds in `dfs/decisions.py` (HR < 0.90 for
   recommendation) reflect current clinical practice.

**Before submission:**

5. Fill in `Acknowledgements`, `Funding`, and `Competing Interests` sections.

6. Supply missing citations marked `[CITE]` in `main.md`:
   - HFpEF guideline citation (ESC 2023 or ACC/AHA 2022)
   - Spironolactone metabolite / non-adherence citation
   - Published MRA-HFpEF systematic review for I² comparison

7. Verify Table 1 patient numbers for FIDELIO and FIGARO HF subgroups
   (flagged with `<!-- AUTHOR REVIEW -->`).

8. Verify FIGARO log-HR for the HF subgroup (Table 1 row is incomplete).

9. Implement `scripts/render_figures.py` to generate Figures 1, 3, 4, and
   optionally 6.

10. Run final word count: `wc -w main.md` should be ≤ 4,500 for *Circulation*
    or ≤ 3,000 for *JAMA Cardiology*.

11. Check reference list: confirm Da Veiga & Marrel 2020 journal citation;
    confirm CLARABEL arXiv has a published venue.

---

## Relationship to Code

The manuscript draws on the following project outputs. Do not edit these files
as part of manuscript preparation:

| Manuscript claim | Source |
|---|---|
| d = 1.56 (TOPCAT Americas vs Russia/Georgia) | `outputs/dissonance.csv` row 1 |
| Adherence-proxy length-scale 0.44 | `outputs/k_sign_constraint_report.json` `.hyperparameters.length_scales[6]` |
| All other length-scales > 2,900 | same file, indices 0–5 |
| Zero conservation violations | `outputs/conservation_diagnostics.json` (empty array) |
| LOO prediction [−0.260, −0.009] | `tests/test_loo_fineartshf.py` output |
| Observed FINEARTS-HF log-HR −0.174 | `data/mra_hfpef/fineartshf.json` |
| k_sign non-binding, min +0.047 | `outputs/k_sign_constraint_report.json` |
| 83 tests passed, 1 skipped | `python -m pytest -v` |
| ΔK⁺ range +0.02 to +0.23 | `outputs/k_sign_constraint_report.json` per-trial |
