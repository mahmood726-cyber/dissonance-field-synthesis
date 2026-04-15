import numpy as np
import pytest


def test_kernel_at_zero_distance_equals_sigma2() -> None:
    from dfs.kernel import ard_matern_52

    x = np.array([[1.0, 2.0, 3.0]])
    k = ard_matern_52(x, x, sigma2=2.5, length_scales=np.array([1.0, 1.0, 1.0]))
    assert k.shape == (1, 1)
    assert k[0, 0] == pytest.approx(2.5, abs=1e-12)


def test_kernel_decays_with_distance() -> None:
    from dfs.kernel import ard_matern_52

    x = np.array([[0.0]])
    y = np.array([[0.5], [2.0], [10.0]])
    k = ard_matern_52(x, y, sigma2=1.0, length_scales=np.array([1.0]))
    assert k.shape == (1, 3)
    assert k[0, 0] > k[0, 1] > k[0, 2]
    assert k[0, 2] < 0.01


def test_ard_length_scales_asymmetric() -> None:
    """Short length-scale dimension decays fast; long one decays slow."""
    from dfs.kernel import ard_matern_52

    x = np.array([[0.0, 0.0]])
    y_dim0 = np.array([[1.0, 0.0]])
    y_dim1 = np.array([[0.0, 1.0]])
    ls = np.array([0.1, 10.0])
    k0 = ard_matern_52(x, y_dim0, sigma2=1.0, length_scales=ls)
    k1 = ard_matern_52(x, y_dim1, sigma2=1.0, length_scales=ls)
    assert k0[0, 0] < k1[0, 0]


def test_symmetric_positive_definite() -> None:
    from dfs.kernel import ard_matern_52

    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3))
    K = ard_matern_52(x, x, sigma2=1.0, length_scales=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(K, K.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(K + 1e-8 * np.eye(5))
    assert np.all(eigs > 0)
