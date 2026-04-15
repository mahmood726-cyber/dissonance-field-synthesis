"""Decision thresholds for mind-change pricing and feasibility regions.

USER-AUTHORED: the cardiologist sets the prescribing heuristic as
explicit log-HR boundaries. These feed the feasibility-region mask
(§7.4) and mind-change price map (§7.3).

Authoring rules:
  - All thresholds are log-HR (use math.log on an HR if needed).
  - recommend_if_log_hr_below < do_not_recommend_if_log_hr_above.
    The band between is 'borderline'.
  - PER_PATIENT_VAR is the per-patient log-HR variance for a
    hypothetical new trial at that endpoint (used in MCP denominator).
    ~1.0 is a reasonable default for mortality; 0.3-0.5 for composites.
"""
from __future__ import annotations

import math


DECISION_THRESHOLDS: dict[str, dict[str, float]] = {
    "primary_composite": {
        "recommend_if_log_hr_below": math.log(0.90),
        "do_not_recommend_if_log_hr_above": math.log(1.00),
    },
    "acm": {
        "recommend_if_log_hr_below": math.log(0.90),
        "do_not_recommend_if_log_hr_above": math.log(1.00),
    },
    "cv_death": {
        "recommend_if_log_hr_below": math.log(0.90),
        "do_not_recommend_if_log_hr_above": math.log(1.00),
    },
    "hf_hosp": {
        "recommend_if_log_hr_below": math.log(0.85),
        "do_not_recommend_if_log_hr_above": math.log(1.00),
    },
}

PER_PATIENT_VAR: dict[str, float] = {
    "primary_composite": 0.5,
    "acm": 1.0,
    "cv_death": 1.0,
    "hf_hosp": 0.4,
}
