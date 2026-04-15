import numpy as np


def test_feasibility_region_excludes_threshold() -> None:
    from dfs.feasibility import feasibility_region

    mu = -0.3 * np.ones(50)
    var = 0.01 * np.ones(50)
    mask = feasibility_region(mu, var, threshold=0.0, ci_level=0.95, direction="below")
    assert mask.all()


def test_threshold_inside_ci_returns_false() -> None:
    from dfs.feasibility import feasibility_region

    mu = np.array([-0.05])
    var = np.array([1.0])
    mask = feasibility_region(mu, var, threshold=0.0, ci_level=0.95, direction="below")
    assert not mask[0]


def test_direction_above() -> None:
    from dfs.feasibility import feasibility_region

    mu = np.array([0.3])
    var = np.array([0.01])
    mask = feasibility_region(mu, var, threshold=0.0, ci_level=0.95, direction="above")
    assert mask[0]


def test_invalid_ci_level_raises() -> None:
    import pytest
    from dfs.feasibility import feasibility_region

    mu = np.array([0.0])
    var = np.array([1.0])
    with pytest.raises(ValueError, match="ci_level"):
        feasibility_region(mu, var, threshold=0.0, ci_level=1.0, direction="below")
    with pytest.raises(ValueError, match="ci_level"):
        feasibility_region(mu, var, threshold=0.0, ci_level=0.0, direction="below")
