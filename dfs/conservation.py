"""Conservation laws for MRA pharmacology.

USER-AUTHORED: the cardiologist (Mahmood) authors the body of this
module. The schema of each entry is fixed. The clinical judgment —
hard vs soft, penalty weight, and the rationale string — is yours.

Do NOT change field names; downstream code in field_constrained.py
depends on them. Do change: the 'type' assignment, the
'penalty_weight' value for soft laws, and the rationale string.

See spec §6 for the six laws and §11 for authorship instructions.
"""
from __future__ import annotations

from typing import Any


CONSERVATION_LAWS: list[dict[str, Any]] = [
    {
        "name": "mortality_decomposition",
        "type": "hard",        # USER: confirm hard or soften
        "penalty_weight": None,
        "rationale": (
            "HR(ACM) = p*HR(CV_death) + q*HR(non_CV_death) on hazard-diff "
            "scale; baseline-prop weights p, q. Confirmed hard because "
            "this is arithmetic identity, not pharmacology."
        ),
    },
    {
        "name": "cv_death_subdecomposition",
        "type": "hard",        # USER: confirm
        "penalty_weight": None,
        "rationale": (
            "HR(CV_death) = sum over subtype of prop_i * HR_i. Same "
            "arithmetic argument as mortality_decomposition."
        ),
    },
    {
        "name": "k_sign",
        "type": "hard",        # USER: confirm (spec §6 flagged this as debatable)
        "penalty_weight": None,
        "rationale": (
            "ΔK⁺ >= 0 wherever MR occupancy > 0 — mechanism sign forced "
            "by aldosterone blockade. Hard because violations must "
            "indicate measurement-timing artefact, not real effect."
        ),
    },
    {
        "name": "sbp_sign",
        "type": "soft",        # USER: confirm (might be hard)
        "penalty_weight": 1.0, # USER: tune
        "rationale": (
            "ΔSBP <= 0 monotone in MR occupancy. Soft because magnitude "
            "is small and some trials show SBP unchanged in normotensive "
            "subgroups. Weight reflects weak confidence in uniform effect."
        ),
    },
    {
        "name": "dose_monotonicity",
        "type": "hard",        # USER: confirm (spec §6 flagged — inverted-U?)
        "penalty_weight": None,
        "rationale": (
            "Within a single drug, ∂log-HR/∂dose <= 0 within tested "
            "dose range. Hard because every titration protocol was "
            "designed under this assumption."
        ),
    },
    {
        "name": "egfr_dip_plateau",
        "type": "soft",        # USER: confirm
        "penalty_weight": 0.5, # USER: tune
        "rationale": (
            "ΔeGFR acute dip (<0) at 4 months, then recovers toward "
            "placebo by 24 months. Soft because follow-up timing varies "
            "across trials; relative weight moderate."
        ),
    },
]
