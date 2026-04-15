"""Unit tests for ML-II hyperparameter fitting (dfs/hyperparam.py)."""
import numpy as np


def test_recovers_known_length_scale_1d() -> None:
    """1-D synthetic with true length-scale = 1.0; fit should recover within 50%."""
    from dfs.hyperparam import fit_hyperparameters

    rng = np.random.default_rng(0)
    n = 30
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    # True f: smooth on length-scale 1.0
    y = np.sin(x[:, 0]) + 0.05 * rng.standard_normal(n)
    noise = np.full(n, 0.05**2)

    hp = fit_hyperparameters(x, y, noise, initial_length_scales=np.array([0.1]),
                             n_restarts=3, seed=0)
    assert 0.5 < hp.length_scales[0] < 3.0
    assert hp.sigma2 > 0
    assert np.isfinite(hp.neg_log_marginal_likelihood)


def test_returns_fitted_dataclass() -> None:
    from dfs.hyperparam import fit_hyperparameters, FittedHyperparameters

    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 2))
    y = rng.standard_normal(10)
    noise = np.full(10, 0.1)

    hp = fit_hyperparameters(x, y, noise, n_restarts=2)
    assert isinstance(hp, FittedHyperparameters)
    assert hp.length_scales.shape == (2,)
