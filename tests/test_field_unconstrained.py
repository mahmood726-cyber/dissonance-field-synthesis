import numpy as np
import pytest


def test_gp_interpolates_observations() -> None:
    """Posterior mean at obs points matches obs values (low noise)."""
    from dfs.field_unconstrained import fit_unconstrained_gp

    x_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 0.5])
    noise = np.array([1e-6, 1e-6, 1e-6])

    gp = fit_unconstrained_gp(
        x_train, y_train, noise,
        sigma2=1.0, length_scales=np.array([0.5]),
    )
    mu, var = gp.predict(x_train)
    np.testing.assert_allclose(mu, y_train, atol=1e-4)


def test_gp_variance_grows_far_from_obs() -> None:
    from dfs.field_unconstrained import fit_unconstrained_gp

    x_train = np.array([[0.0]])
    y_train = np.array([1.0])
    gp = fit_unconstrained_gp(
        x_train, y_train, np.array([1e-6]),
        sigma2=1.0, length_scales=np.array([1.0]),
    )
    _, var_near = gp.predict(np.array([[0.1]]))
    _, var_far = gp.predict(np.array([[10.0]]))
    assert var_far[0] > var_near[0]
    assert var_far[0] == pytest.approx(1.0, abs=1e-3)


def test_heteroscedastic_noise_honored() -> None:
    """High-noise observation should not pin the posterior."""
    from dfs.field_unconstrained import fit_unconstrained_gp

    x_train = np.array([[0.0], [1.0]])
    y_train = np.array([0.0, 1.0])
    gp_low = fit_unconstrained_gp(
        x_train, y_train, np.array([1e-6, 1e-6]),
        sigma2=1.0, length_scales=np.array([1.0]),
    )
    gp_high = fit_unconstrained_gp(
        x_train, y_train, np.array([1e-6, 10.0]),
        sigma2=1.0, length_scales=np.array([1.0]),
    )
    mu_low, _ = gp_low.predict(np.array([[1.0]]))
    mu_high, _ = gp_high.predict(np.array([[1.0]]))
    assert abs(mu_low[0] - 1.0) < 1e-3
    assert abs(mu_high[0] - 1.0) > 0.1
