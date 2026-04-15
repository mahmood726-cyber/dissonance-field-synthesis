import math

import numpy as np
import pytest


def test_price_is_positive_when_crossing_needed() -> None:
    from dfs.mind_change import mind_change_price

    # μ = -0.1, σ² = 0.01; threshold T = 0.0 (recommend if log-HR < 0).
    # Disconfirmation: new evidence at T_obs = 0.1 (harm direction).
    n = mind_change_price(
        posterior_mean=-0.1, posterior_var=0.01,
        t_cross=0.0, t_obs=0.1,
        per_patient_var=1.0,
    )
    assert n > 0


def test_price_infinite_when_observation_at_threshold() -> None:
    from dfs.mind_change import mind_change_price

    n = mind_change_price(
        posterior_mean=-0.1, posterior_var=0.01,
        t_cross=0.0, t_obs=0.0,
        per_patient_var=1.0,
    )
    assert math.isinf(n)


def test_price_zero_when_already_past_threshold_in_hostile_direction() -> None:
    from dfs.mind_change import mind_change_price

    # μ already > T_cross, so no new evidence needed to "stay past" — define as 0.
    n = mind_change_price(
        posterior_mean=0.1, posterior_var=0.01,
        t_cross=0.0, t_obs=-0.1,
        per_patient_var=1.0,
    )
    assert n == 0.0


def test_disconfirmation_vs_confirmation_prices() -> None:
    from dfs.mind_change import mind_change_price

    disc = mind_change_price(
        posterior_mean=-0.3, posterior_var=0.005,
        t_cross=-0.1, t_obs=0.0, per_patient_var=1.0,
    )
    conf = mind_change_price(
        posterior_mean=-0.3, posterior_var=0.005,
        t_cross=-0.4, t_obs=-0.5, per_patient_var=1.0,
    )
    assert disc > 0
    assert conf >= 0


def test_negative_variance_rejected() -> None:
    from dfs.mind_change import mind_change_price

    with pytest.raises(ValueError, match="positive"):
        mind_change_price(
            posterior_mean=0.0, posterior_var=-1.0,
            t_cross=0.0, t_obs=0.1,
            per_patient_var=1.0,
        )
