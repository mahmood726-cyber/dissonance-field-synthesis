import pytest


def test_all_six_laws_registered() -> None:
    from dfs.conservation import CONSERVATION_LAWS
    names = {law["name"] for law in CONSERVATION_LAWS}
    expected = {
        "mortality_decomposition",
        "cv_death_subdecomposition",
        "k_sign",
        "sbp_sign",
        "dose_monotonicity",
        "egfr_dip_plateau",
    }
    assert names == expected


def test_each_law_has_required_keys() -> None:
    from dfs.conservation import CONSERVATION_LAWS
    for law in CONSERVATION_LAWS:
        assert "name" in law
        assert law["type"] in {"hard", "soft"}
        if law["type"] == "soft":
            assert isinstance(law["penalty_weight"], (int, float))
            assert law["penalty_weight"] > 0
        assert "rationale" in law
        assert len(law["rationale"]) > 10


def test_hard_laws_have_no_penalty_weight() -> None:
    from dfs.conservation import CONSERVATION_LAWS
    for law in CONSERVATION_LAWS:
        if law["type"] == "hard":
            assert law.get("penalty_weight") is None
