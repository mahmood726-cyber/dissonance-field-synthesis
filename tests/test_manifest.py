import json
from pathlib import Path

import pytest

from dfs.schema import TrialBoundaryCondition


def test_manifest_loads_all_listed_trials(tmp_path: Path, synth_trial_a, synth_trial_b) -> None:
    from dfs.manifest import load_trials

    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    synth_trial_a.to_json(a_path)
    synth_trial_b.to_json(b_path)

    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "trials": [{"id": "SYNTH-A", "file": "a.json"},
                   {"id": "SYNTH-B", "file": "b.json"}]
    }))

    trials = load_trials(manifest)
    assert len(trials) == 2
    assert all(isinstance(t, TrialBoundaryCondition) for t in trials)
    assert {t.trial_id for t in trials} == {"SYNTH-A", "SYNTH-B"}


def test_manifest_missing_file_raises(tmp_path: Path) -> None:
    from dfs.manifest import load_trials

    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "trials": [{"id": "GHOST", "file": "nonexistent.json"}]
    }))
    with pytest.raises(FileNotFoundError):
        load_trials(manifest)


def test_manifest_id_mismatch_fails_closed(tmp_path: Path, synth_trial_a) -> None:
    from dfs.manifest import load_trials

    a_path = tmp_path / "a.json"
    synth_trial_a.to_json(a_path)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "trials": [{"id": "WRONG-ID", "file": "a.json"}]
    }))
    with pytest.raises(ValueError, match="trial_id mismatch"):
        load_trials(manifest)
