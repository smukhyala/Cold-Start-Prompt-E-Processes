"""Load a JSONL run log into a pandas DataFrame."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_run(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)
    return df
