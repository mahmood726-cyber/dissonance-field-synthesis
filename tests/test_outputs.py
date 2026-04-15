import json
from pathlib import Path

import numpy as np


def test_dissonance_table_written(tmp_path: Path, synth_trial_a, synth_trial_b) -> None:
    from dfs.dissonance import pairwise_dissonance
    from dfs.outputs import write_dissonance_table

    pairs = pairwise_dissonance([synth_trial_a, synth_trial_b], outcome="primary_composite")
    out_csv = tmp_path / "dissonance.csv"
    write_dissonance_table(pairs, out_csv)
    content = out_csv.read_text(encoding="utf-8").splitlines()
    assert content[0].startswith("trial_a,trial_b,outcome,d,log_hr_delta")
    assert len(content) == 2


def test_field_slice_png_written(tmp_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from dfs.outputs import plot_field_slice

    grid = np.linspace(0, 1, 20)
    mu_grid = -0.2 * np.ones((20, 20))
    var_grid = 0.01 * np.ones((20, 20))
    out = tmp_path / "slice.png"
    plot_field_slice(grid, grid, mu_grid, var_grid, out,
                     x_label="LVEF (norm)", y_label="eGFR (norm)")
    assert out.exists()
    assert out.stat().st_size > 1000
