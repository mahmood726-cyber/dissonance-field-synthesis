"""Loads trial boundary-condition records from a manifest JSON."""
from __future__ import annotations

import json
from pathlib import Path

from dfs.schema import TrialBoundaryCondition


def load_trials(manifest_path: Path) -> list[TrialBoundaryCondition]:
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "trials" not in raw:
        raise KeyError(
            f"{manifest_path}: expected top-level 'trials' key, got keys: {list(raw)}"
        )
    base = manifest_path.parent
    trials: list[TrialBoundaryCondition] = []
    for entry in raw["trials"]:
        trial_path = base / entry["file"]
        if not trial_path.exists():
            raise FileNotFoundError(f"Manifest lists {trial_path}, not found")
        record = TrialBoundaryCondition.from_json(trial_path)
        if record.trial_id != entry["id"]:
            raise ValueError(
                f"trial_id mismatch in {trial_path}: "
                f"manifest says {entry['id']!r}, file says {record.trial_id!r}"
            )
        trials.append(record)
    ids = [t.trial_id for t in trials]
    if len(ids) != len(set(ids)):
        dupes = sorted({tid for tid in ids if ids.count(tid) > 1})
        raise ValueError(
            f"Duplicate trial_id(s) in manifest {manifest_path}: {dupes}"
        )
    return trials
