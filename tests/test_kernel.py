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


# --- Additional kernels (Matérn-3/2, RBF) for sensitivity sweep ---------

@pytest.mark.parametrize("name", ["matern52", "matern32", "rbf"])
def test_all_kernels_at_zero_equal_sigma2(name: str) -> None:
    from dfs.kernel import KERNELS

    x = np.array([[0.3, -0.5, 1.2]])
    k = KERNELS[name](x, x, sigma2=2.5, length_scales=np.array([1.0, 1.0, 1.0]))
    assert k.shape == (1, 1)
    assert k[0, 0] == pytest.approx(2.5, abs=1e-12)


@pytest.mark.parametrize("name", ["matern52", "matern32", "rbf"])
def test_all_kernels_decay_and_are_psd(name: str) -> None:
    from dfs.kernel import KERNELS

    rng = np.random.default_rng(42)
    x = rng.standard_normal((6, 3))
    ls = np.array([1.0, 1.0, 1.0])
    K = KERNELS[name](x, x, sigma2=1.0, length_scales=ls)
    assert np.allclose(K, K.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(K + 1e-8 * np.eye(6))
    assert np.all(eigs > 0)


def test_kernel_smoothness_ordering() -> None:
    """At any r>0, RBF (ν→∞) > Matérn-5/2 (ν=2.5) > Matérn-3/2 (ν=1.5).

    The smoother kernel assigns higher covariance at the same distance.
    """
    from dfs.kernel import ard_matern_32, ard_matern_52, ard_rbf

    x = np.array([[0.0]])
    y = np.array([[0.5]])
    ls = np.array([1.0])
    k_rbf = ard_rbf(x, y, 1.0, ls)[0, 0]
    k_52 = ard_matern_52(x, y, 1.0, ls)[0, 0]
    k_32 = ard_matern_32(x, y, 1.0, ls)[0, 0]
    assert k_rbf > k_52 > k_32
