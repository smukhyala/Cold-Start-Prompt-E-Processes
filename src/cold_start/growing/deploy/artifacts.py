"""Saved-model contract shared by the trainer (writer) and `ModelPolicy` (reader).

A model artifact is one joblib file holding a dict:

    {
        "pipeline": <sklearn estimator with predict_proba(X) -> (n, 2)>,
        "features": [str, ...],      # column order the pipeline expects, all deployable
        "k": int,                    # commitment horizon the label was defined at (1/4/16)
        "tau": float | None,         # selected SEARCH threshold on P(SEARCH); None = unset
        "meta": {...},               # variant name, training rows, offline metrics, seeds
    }

The pipeline is stored whole (scaler included) so deployment never folds coefficients
by hand -- that is where the previous `phi.json` path would have gone wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .feature_groups import assert_deployable

REQUIRED_KEYS = ("pipeline", "features", "k", "tau", "meta")


def save_model(path: str | Path, artifact: dict[str, Any]) -> Path:
    missing = [k for k in REQUIRED_KEYS if k not in artifact]
    if missing:
        raise ValueError(f"model artifact missing keys: {missing}")
    assert_deployable(list(artifact["features"]))
    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise TypeError("pipeline must implement predict_proba")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_model(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(Path(path))
    missing = [k for k in REQUIRED_KEYS if k not in artifact]
    if missing:
        raise ValueError(f"model artifact at {path} missing keys: {missing}")
    assert_deployable(list(artifact["features"]))
    return artifact
