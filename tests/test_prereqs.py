"""Preflight: all downstream tasks assume these imports succeed."""
import importlib

import pytest


REQUIRED = [
    "numpy", "scipy", "scipy.optimize", "scipy.linalg",
    "cvxpy", "statsmodels.api", "scipy.stats.qmc", "matplotlib",
    "dfs", "dfs.config",
]


@pytest.mark.parametrize("mod", REQUIRED)
def test_importable(mod: str) -> None:
    importlib.import_module(mod)


def test_covariate_names_length() -> None:
    from dfs.config import COVARIATE_NAMES
    assert len(COVARIATE_NAMES) == 7
