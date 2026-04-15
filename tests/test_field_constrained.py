import numpy as np
import pytest


def test_nonnegativity_constraint_respected() -> None:
    """If we require f(x) >= 0 at sample points, posterior mean respects it."""
    from dfs.field_constrained import fit_constrained_gp

    x_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([-0.5, 0.1, 0.3])
    noise = np.array([0.01, 0.01, 0.01])
    virtual_grid = np.linspace(-0.5, 2.5, 50).reshape(-1, 1)

    gp = fit_constrained_gp(
        x_train, y_train, noise,
        sigma2=1.0, length_scales=np.array([0.7]),
        inequality_constraints=[{
            "matrix": np.eye(50),
            "bound": np.zeros(50),
            "direction": "geq",
            "grid": virtual_grid,
        }],
    )
    mu, _ = gp.predict(virtual_grid)
    assert np.all(mu >= -1e-4)


def test_constrained_gp_reduces_to_unconstrained_without_constraints() -> None:
    """No constraints -> same answer as Task 5's unconstrained fitter."""
    from dfs.field_unconstrained import fit_unconstrained_gp
    from dfs.field_constrained import fit_constrained_gp

    x_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.2, -0.1, 0.3])
    noise = np.array([0.01, 0.02, 0.01])
    kw = dict(sigma2=1.0, length_scales=np.array([0.5]))

    gp_u = fit_unconstrained_gp(x_train, y_train, noise, **kw)
    gp_c = fit_constrained_gp(
        x_train, y_train, noise, **kw, inequality_constraints=[],
    )
    x_test = np.linspace(-1, 3, 20).reshape(-1, 1)
    mu_u, _ = gp_u.predict(x_test)
    mu_c, _ = gp_c.predict(x_test)
    np.testing.assert_allclose(mu_u, mu_c, atol=1e-4)
