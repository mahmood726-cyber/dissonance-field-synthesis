"""
extract_aact_mra.py  (UTF-8 stdout shim applied at top)
-------------------
Extracts outcome_analyses, outcomes, and baseline_measurements from the AACT
2026-04-12 snapshot for the 5 MRA-HFpEF trials that have AACT coverage:
  NCT00094302  TOPCAT (full trial)
  NCT04435626  FINEARTS-HF
  NCT02540993  FIDELIO-DKD
  NCT02545049  FIGARO-DKD
  NCT00290433  Aldo-DHF (expected: no results, included for completeness)

Usage:
  python scripts/extract_aact_mra.py [--aact-dir <path>]

Outputs a structured summary to stdout suitable for JSON fixture upgrades.
"""

import csv
import io
import math
import sys
import argparse
from collections import defaultdict

# Windows cp1252 console fix — wrap stdout to UTF-8 before any print
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_AACT = r"C:\Users\user\AACT\2026-04-12"

NCT_IDS = {
    "NCT00094302": "TOPCAT",
    "NCT04435626": "FINEARTS-HF",
    "NCT02540993": "FIDELIO-DKD",
    "NCT02545049": "FIGARO-DKD",
    "NCT00290433": "Aldo-DHF",
}

# Outcome title keywords we care about (case-insensitive substrings)
OUTCOME_KEYWORDS = [
    "cardiovascular death",
    "cardiovascular mortality",
    "all-cause mortality",
    "all cause mortality",
    "heart failure",
    "hospitalization",
    "composite",
    "sudden death",
    "myocardial infarction",
    "aborted cardiac arrest",
    "total worsening heart failure",
    "total heart failure",
    "worsening heart failure",
    "first and recurrent",
    "recurrent",
    "kidney",          # FIDELIO/FIGARO primary
]

# Baseline measurement title keywords
BASELINE_KEYWORDS = [
    "age",
    "lvef",
    "left ventricular ejection fraction",
    "egfr",
    "glomerular filtration rate",
    "creatinine clearance",
    "potassium",
    "serum potassium",
    "diabetes",
    "diabetes mellitus",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def log_hr_se(hr, ci_lo, ci_hi):
    """Return (log_hr, se) from HR and 95% CI bounds."""
    if hr is None or ci_lo is None or ci_hi is None:
        return None, None
    if hr <= 0 or ci_lo <= 0 or ci_hi <= 0:
        return None, None
    log_hr = math.log(hr)
    se = (math.log(ci_hi) - math.log(ci_lo)) / (2 * 1.96)
    return round(log_hr, 6), round(se, 6)


def matches_any(text, keywords):
    t = (text or "").lower()
    return any(kw in t for kw in keywords)


def robust_reader(path, encoding="utf-8"):
    """Open a pipe-delimited file with csv.reader, skipping bad lines."""
    with open(path, encoding=encoding, errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            yield row


# ---------------------------------------------------------------------------
# Step 1: Read outcomes.txt — build {outcome_id -> (nct_id, title)}
# ---------------------------------------------------------------------------
def load_outcomes(aact_dir, nct_ids):
    """Returns dict outcome_id -> (nct_id, title, outcome_type) for our trials."""
    path = f"{aact_dir}/outcomes.txt"
    print(f"[INFO] Reading {path} ...", flush=True)
    outcome_map = {}   # id -> (nct_id, title, outcome_type)
    nct_set = set(nct_ids)
    header = None
    row_count = 0
    match_count = 0
    for row in robust_reader(path):
        if header is None:
            header = row
            col = {c: i for i, c in enumerate(header)}
            # Verify expected columns
            for needed in ("id", "nct_id", "outcome_type", "title"):
                if needed not in col:
                    print(f"[WARN] Missing column '{needed}' in outcomes.txt header")
            continue
        row_count += 1
        if row_count % 200000 == 0:
            print(f"  ... {row_count} rows scanned, {match_count} matched", flush=True)
        try:
            nct = row[col["nct_id"]]
        except IndexError:
            continue
        if nct not in nct_set:
            continue
        try:
            oid = row[col["id"]]
            otype = row[col["outcome_type"]]
            title = row[col["title"]]
        except IndexError:
            continue
        outcome_map[oid] = (nct, title, otype)
        match_count += 1
    print(f"[INFO] outcomes.txt: {row_count} rows, {match_count} matched for target NCTs\n")
    return outcome_map


# ---------------------------------------------------------------------------
# Step 2: Read outcome_analyses.txt
# ---------------------------------------------------------------------------
def load_outcome_analyses(aact_dir, outcome_map):
    """Returns list of dicts for outcome analyses matching our outcome IDs."""
    path = f"{aact_dir}/outcome_analyses.txt"
    print(f"[INFO] Reading {path} ...", flush=True)
    target_oids = set(outcome_map.keys())
    results = []
    header = None
    row_count = 0
    match_count = 0
    for row in robust_reader(path):
        if header is None:
            header = row
            col = {c: i for i, c in enumerate(header)}
            for needed in ("id", "nct_id", "outcome_id", "param_type", "param_value",
                           "ci_lower_limit", "ci_upper_limit", "p_value"):
                if needed not in col:
                    print(f"[WARN] Missing column '{needed}' in outcome_analyses.txt header")
            continue
        row_count += 1
        if row_count % 500000 == 0:
            print(f"  ... {row_count} rows scanned, {match_count} matched", flush=True)
        try:
            oid = row[col["outcome_id"]]
        except IndexError:
            continue
        if oid not in target_oids:
            continue
        match_count += 1
        try:
            rec = {
                "analysis_id": row[col["id"]],
                "nct_id": row[col["nct_id"]],
                "outcome_id": oid,
                "param_type": row[col.get("param_type", -1)] if col.get("param_type") is not None else "",
                "param_value": safe_float(row[col["param_value"]]),
                "ci_lower_limit": safe_float(row[col["ci_lower_limit"]]),
                "ci_upper_limit": safe_float(row[col["ci_upper_limit"]]),
                "p_value": row[col["p_value"]],
                "dispersion_type": row[col.get("dispersion_type", -1)] if "dispersion_type" in col else "",
                "dispersion_value": row[col.get("dispersion_value", -1)] if "dispersion_value" in col else "",
                "method": row[col["method"]] if "method" in col else "",
                "ci_percent": row[col["ci_percent"]] if "ci_percent" in col else "",
                "non_inferiority_type": row[col["non_inferiority_type"]] if "non_inferiority_type" in col else "",
            }
            # Attach outcome metadata
            nct_id, title, otype = outcome_map.get(oid, ("?", "?", "?"))
            rec["outcome_title"] = title
            rec["outcome_type"] = otype
            results.append(rec)
        except (IndexError, Exception) as e:
            pass
    print(f"[INFO] outcome_analyses.txt: {row_count} rows, {match_count} matched\n")
    return results


# ---------------------------------------------------------------------------
# Step 3: Read baseline_measurements.txt
# ---------------------------------------------------------------------------
def load_baselines(aact_dir, nct_ids):
    path = f"{aact_dir}/baseline_measurements.txt"
    print(f"[INFO] Reading {path} ...", flush=True)
    nct_set = set(nct_ids)
    results = []
    header = None
    row_count = 0
    match_count = 0
    for row in robust_reader(path):
        if header is None:
            header = row
            col = {c: i for i, c in enumerate(header)}
            for needed in ("id", "nct_id", "title", "param_type", "param_value",
                           "ctgov_group_code"):
                if needed not in col:
                    print(f"[WARN] Missing baseline column '{needed}'")
            continue
        row_count += 1
        if row_count % 500000 == 0:
            print(f"  ... {row_count} rows scanned, {match_count} matched", flush=True)
        try:
            nct = row[col["nct_id"]]
        except IndexError:
            continue
        if nct not in nct_set:
            continue
        try:
            title = row[col["title"]]
        except IndexError:
            title = ""
        if not matches_any(title, BASELINE_KEYWORDS):
            continue
        match_count += 1
        try:
            rec = {
                "id": row[col["id"]],
                "nct_id": nct,
                "ctgov_group_code": row[col["ctgov_group_code"]],
                "classification": row[col.get("classification", -1)] if "classification" in col else "",
                "category": row[col.get("category", -1)] if "category" in col else "",
                "title": title,
                "param_type": row[col["param_type"]],
                "param_value": row[col["param_value"]],
                "param_value_num": safe_float(row[col["param_value_num"]]) if "param_value_num" in col else None,
                "dispersion_type": row[col.get("dispersion_type", -1)] if "dispersion_type" in col else "",
                "dispersion_value": row[col.get("dispersion_value", -1)] if "dispersion_value" in col else "",
                "dispersion_lower_limit": row[col.get("dispersion_lower_limit", -1)] if "dispersion_lower_limit" in col else "",
                "dispersion_upper_limit": row[col.get("dispersion_upper_limit", -1)] if "dispersion_upper_limit" in col else "",
                "number_analyzed": row[col.get("number_analyzed", -1)] if "number_analyzed" in col else "",
            }
            results.append(rec)
        except (IndexError, Exception):
            pass
    print(f"[INFO] baseline_measurements.txt: {row_count} rows, {match_count} matched\n")
    return results


# ---------------------------------------------------------------------------
# Step 4: Print summaries
# ---------------------------------------------------------------------------
def print_outcome_summary(analyses, outcome_map):
    # Group by trial
    by_trial = defaultdict(list)
    for rec in analyses:
        by_trial[rec["nct_id"]].append(rec)

    print("=" * 80)
    print("OUTCOME ANALYSES SUMMARY")
    print("=" * 80)

    for nct_id in sorted(by_trial.keys()):
        trial_name = NCT_IDS.get(nct_id, nct_id)
        recs = by_trial[nct_id]
        print(f"\n{'─'*70}")
        print(f"Trial: {nct_id} ({trial_name}) — {len(recs)} analysis rows")
        print(f"{'-'*70}")

        # Group by outcome_id
        by_oid = defaultdict(list)
        for r in recs:
            by_oid[r["outcome_id"]].append(r)

        for oid, orecs in sorted(by_oid.items()):
            title = orecs[0]["outcome_title"]
            otype = orecs[0]["outcome_type"]
            print(f"\n  Outcome: {title[:80]}")
            print(f"  outcome_id={oid}  type={otype}")
            for r in orecs:
                pv = r["param_value"]
                lo = r["ci_lower_limit"]
                hi = r["ci_upper_limit"]
                pt = r["param_type"]
                m  = r["method"]
                p  = r["p_value"]
                ci_pct = r["ci_percent"]
                log_hr, se = log_hr_se(pv, lo, hi)
                print(f"    analysis_id={r['analysis_id']}")
                print(f"    param_type={pt!r}  value={pv}  CI-{ci_pct}%=[{lo}, {hi}]  p={p}")
                print(f"    method={m!r}")
                if log_hr is not None:
                    print(f"    => log_hr={log_hr}  se={se}")
                else:
                    print(f"    => log_hr/se: cannot compute (missing HR or CI bounds)")


def print_baseline_summary(baselines):
    by_trial = defaultdict(list)
    for rec in baselines:
        by_trial[rec["nct_id"]].append(rec)

    print("\n" + "=" * 80)
    print("BASELINE MEASUREMENTS SUMMARY")
    print("=" * 80)

    for nct_id in sorted(by_trial.keys()):
        trial_name = NCT_IDS.get(nct_id, nct_id)
        recs = by_trial[nct_id]
        print(f"\n{'─'*70}")
        print(f"Trial: {nct_id} ({trial_name}) — {len(recs)} baseline rows")
        print(f"{'-'*70}")

        # Group by title+group_code for display
        by_title = defaultdict(list)
        for r in recs:
            by_title[r["title"]].append(r)

        for title, trecs in sorted(by_title.items()):
            print(f"\n  Measure: {title[:80]}")
            for r in trecs:
                grp = r["ctgov_group_code"]
                pt = r["param_type"]
                pv = r["param_value"]
                pv_num = r["param_value_num"]
                dt = r["dispersion_type"]
                dv = r["dispersion_value"]
                dl = r["dispersion_lower_limit"]
                du = r["dispersion_upper_limit"]
                n  = r["number_analyzed"]
                cat = r["category"]
                cls = r["classification"]
                print(f"    group={grp}  n={n}  param_type={pt}  value={pv} ({pv_num})")
                print(f"      dispersion: type={dt}  value={dv}  range=[{dl},{du}]")
                if cat or cls:
                    print(f"      category={cat!r}  classification={cls!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract AACT data for MRA-HFpEF trials")
    parser.add_argument("--aact-dir", default=DEFAULT_AACT,
                        help="Path to AACT 2026-04-12 snapshot directory")
    args = parser.parse_args()

    aact_dir = args.aact_dir
    print(f"[INFO] AACT directory: {aact_dir}")
    print(f"[INFO] Target NCTs: {list(NCT_IDS.keys())}\n")

    # Load outcomes for our trials
    outcome_map = load_outcomes(aact_dir, NCT_IDS)
    print(f"[INFO] Total outcomes found for target NCTs: {len(outcome_map)}")

    # Filter to only outcomes matching our keywords
    filtered_oid_map = {
        oid: v for oid, v in outcome_map.items()
        if matches_any(v[1], OUTCOME_KEYWORDS)
    }
    print(f"[INFO] After keyword filter: {len(filtered_oid_map)} outcomes\n")

    # Load analyses for filtered outcomes
    analyses = load_outcome_analyses(aact_dir, filtered_oid_map)

    # Load baselines
    baselines = load_baselines(aact_dir, NCT_IDS)

    # Print summaries
    print_outcome_summary(analyses, filtered_oid_map)
    print_baseline_summary(baselines)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
