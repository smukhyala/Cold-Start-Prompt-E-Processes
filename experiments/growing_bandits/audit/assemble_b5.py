#!/usr/bin/env python3
"""EXPLORATORY: assemble b5_sim.csv / b5_validity.csv / b5_headline.{csv,md} from the two partial runs."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.growing_bandits.audit import run_b5 as rb  # noqa: E402

OUT = rb.OUT
sim = pd.concat([pd.read_csv(OUT / "b5_sim_efficacy.csv"), pd.read_csv(OUT / "b5_sim_futility.csv")],
                ignore_index=True)
sim.to_csv(OUT / "b5_sim.csv", index=False)
is_val = sim["scenario"].isin(["allnull", "reallike_below", "boundary"])
val = sim[is_val & sim["design"].isin(["combined_a", "combined_b", "known_only"])]
val.to_csv(OUT / "b5_validity.csv", index=False)
head = rb.build_headline(sim)
head.to_csv(OUT / "b5_headline.csv", index=False)
rb.write_md(head, val, OUT / "b5_headline.md")
