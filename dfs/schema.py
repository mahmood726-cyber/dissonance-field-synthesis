"""Boundary-condition record for a single trial in DFS."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dfs.config import COVARIATE_NAMES


@dataclass(frozen=True)
class TrialBoundaryCondition:
    trial_id: str
    drug: str
    mr_occupancy_equivalent: float
    anchor_covariates: dict[str, float]
    covariate_ranges: dict[str, tuple[float, float]]
    outcomes: dict[str, dict[str, float]]
    safety: dict[str, dict[str, float]]
    design_priors: dict[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.mr_occupancy_equivalent, bool) or not isinstance(
            self.mr_occupancy_equivalent, (int, float)
        ):
            raise TypeError(
                f"mr_occupancy_equivalent must be numeric (not bool), got "
                f"{type(self.mr_occupancy_equivalent).__name__}"
            )
        missing = [c for c in COVARIATE_NAMES if c not in self.anchor_covariates]
        if missing:
            raise KeyError(
                f"anchor_covariates missing required keys: {missing}. "
                f"Expected all of {list(COVARIATE_NAMES)}"
            )

    def to_json(self, path: Path) -> None:
        payload: dict[str, Any] = asdict(self)
        payload["covariate_ranges"] = {
            k: list(v) for k, v in self.covariate_ranges.items()
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> TrialBoundaryCondition:
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["covariate_ranges"] = {
            k: tuple(v) for k, v in raw.get("covariate_ranges", {}).items()
        }
        try:
            return cls(**raw)
        except (TypeError, KeyError) as exc:
            raise type(exc)(f"While loading {path}: {exc}") from exc
