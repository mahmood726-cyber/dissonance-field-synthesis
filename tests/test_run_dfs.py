import subprocess
import sys
from pathlib import Path


def test_run_dfs_completes_and_produces_artefacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    result = subprocess.run(
        [sys.executable, "scripts/run_dfs.py",
         "--manifest", "data/mra_hfpef/MANIFEST.json",
         "--out", str(out_dir)],
        capture_output=True, text=True,
        timeout=600,  # ML-II + CVXPY import on this machine takes ~100-150s
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    assert (out_dir / "dissonance.csv").exists()
    assert (out_dir / "field_lvef_egfr.png").exists()
    assert (out_dir / "mind_change_price.csv").exists()
    assert (out_dir / "feasibility_mask.csv").exists()
    assert (out_dir / "conservation_diagnostics.json").exists()
    assert (out_dir / "safety_delta_k_lvef_egfr.png").exists()
    assert (out_dir / "k_sign_constraint_report.json").exists()
