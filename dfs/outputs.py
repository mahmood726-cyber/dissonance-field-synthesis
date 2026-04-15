"""Render DFS outputs as plots + CSV/JSON."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from dfs.dissonance import DissonancePair


def write_dissonance_table(pairs: list[DissonancePair], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["trial_a", "trial_b", "outcome", "d", "log_hr_delta"]
        cov_keys: list[str] = []
        if pairs:
            cov_keys = list(pairs[0].covariate_delta.keys())
            header += [f"delta_{k}" for k in cov_keys]
        writer.writerow(header)
        for p in pairs:
            row = [p.trial_ids[0], p.trial_ids[1], p.outcome,
                   f"{p.d:.4f}", f"{p.log_hr_delta:+.4f}"]
            row += [f"{p.covariate_delta[k]:+.4f}" for k in cov_keys]
            writer.writerow(row)


def plot_field_slice(
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    mu_grid: NDArray[np.float64],
    var_grid: NDArray[np.float64],
    out_path: Path,
    x_label: str,
    y_label: str,
) -> None:
    fig, (ax_mu, ax_var) = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax_mu.pcolormesh(x_grid, y_grid, mu_grid, shading="auto", cmap="RdBu_r")
    ax_mu.set_title("Posterior mean log-HR")
    ax_mu.set_xlabel(x_label)
    ax_mu.set_ylabel(y_label)
    fig.colorbar(im1, ax=ax_mu)
    im2 = ax_var.pcolormesh(x_grid, y_grid, np.sqrt(np.clip(var_grid, 0.0, None)),
                            shading="auto", cmap="viridis")
    ax_var.set_title("Posterior SD log-HR")
    ax_var.set_xlabel(x_label)
    fig.colorbar(im2, ax=ax_var)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
