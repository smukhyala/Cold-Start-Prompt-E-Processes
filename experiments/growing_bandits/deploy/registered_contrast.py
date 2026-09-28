"""The registered contrast H1b' (DEPLOYMENT_PLAN.md, "Pre-registration 2"), and nothing else.

One primary row: Δ = regret(policy) − regret(reference) per episode, pooled over the
registered horizons of one test with the cell-stratified paired bootstrap, and the
environment-mean t interval over the panel's environments with its two-sided `cluster_p`.
Three secondary rows (one per registered horizon) Holm-corrected among themselves. The
remaining horizons are reported as `structural`, uncorrected, because the registration
says the two policies cannot differ there.

The verdict is computed from the rule written down in the registration, so the script
cannot be talked into a different reading afterwards:

* ``supported``   iff ``cluster_p < 0.05`` and ``delta < 0`` and ``|delta| >= mei``;
* ``refuted``     iff ``delta >= 0`` or the t interval lies entirely above ``-mei``
                  (the effect is significantly smaller than the minimum of interest);
* ``inconclusive`` otherwise.

Usage::

    .venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py

writes ``tables/h1b_prime.csv``. The defaults ARE the registration; passing anything
else produces a different contrast and the output says which.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_deployment as ad  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.deploy import stats  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402

log = logging.getLogger("deploy.registered")

#: The registrations, verbatim (DEPLOYMENT_PLAN.md, Pre-registrations 2 and 3). Each is one
#: primary contrast; the secondary of registration 3 is listed under its own name.
REGISTRATIONS: dict[str, dict] = {
    "h1b_prime": {"policy": "phi_k4", "reference": "p3_star", "test": "robust",
                  "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_prime.csv"},
    "h1b_null": {"policy": "phi_k4", "reference": "fixed_K_star", "test": "robust",
                 "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_null.csv"},
    "h1b_null_secondary": {"policy": "fixed_K_star", "reference": "p3_star", "test": "robust",
                           "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_null_secondary.csv"},
    "h1b_adaptive": {"policy": "phi_k4", "reference": "adaptive_K_star", "test": "robust",
                     "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_adaptive.csv"},
    "h1b_adaptive_secondary": {"policy": "adaptive_K_star", "reference": "fixed_K_star", "test": "robust",
                               "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_adaptive_secondary.csv"},
    "h1b_bestmean": {"policy": "phi_k4", "reference": "bestmean_star", "test": "robust",
                     "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_bestmean.csv"},
    "h1b_bestmean_secondary": {"policy": "bestmean_star", "reference": "fixed_K_star", "test": "robust",
                               "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_bestmean_secondary.csv"},
    "h1b_level": {"policy": "phi_k4", "reference": "level_star", "test": "robust",
                  "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_level.csv"},
    "h1b_level_secondary": {"policy": "level_star", "reference": "fixed_K_star", "test": "robust",
                            "horizons": (50, 100, 200), "mei": 0.002, "out": "h1b_level_secondary.csv"},
    # Pre-registration 8: the held-out family, uncapped, n_envs = 3 -- rules on the PAIRED CI.
    "capc_primary": {"policy": "p3_star", "reference": "fixed_K_star", "test": "capc",
                     "horizons": (200, 1000), "mei": 0.002, "out": "capc_primary.csv", "rule": "noninferiority"},
    "capc_level": {"policy": "level_star", "reference": "p3_star", "test": "capc",
                   "horizons": (200, 1000), "mei": 0.002, "out": "capc_level.csv", "rule": "not_better"},
    "capc_phi": {"policy": "phi_k4", "reference": "p3_star", "test": "capc",
                 "horizons": (200, 1000), "mei": 0.002, "out": "capc_phi.csv", "rule": "not_better"},
    # Pre-registration 9: the real prompt pools, uncapped, informative primary cells only, on the
    # PROMPT-BOOTSTRAP interval (B = 200 NPMLE re-estimates; `prompt_bootstrap_contrast`).
    "emp_primary": {"policy": "p3_star", "reference": "fixed_K_star", "test": "emp",
                    "horizons": (50, 100, 200), "mei": 0.002, "out": "emp_primary.csv", "rule": "noninferiority",
                    "interval": "prompt_bootstrap", "n_boot": 200},
    "emp_level": {"policy": "level_star", "reference": "p3_star", "test": "emp",
                  "horizons": (50, 100, 200), "mei": 0.002, "out": "emp_level.csv", "rule": "not_better",
                  "interval": "prompt_bootstrap", "n_boot": 200},
    "emp_phi": {"policy": "phi_k4", "reference": "p3_star", "test": "emp",
                "horizons": (50, 100, 200), "mei": 0.002, "out": "emp_phi.csv", "rule": "not_better",
                "interval": "prompt_bootstrap", "n_boot": 200},
}
#: Decision rules. ``superiority`` (the default) reads the environment-mean t interval and
#: is the rule of Pre-registrations 2-6; the two paired-CI rules are Pre-registration 8's,
#: for a panel below `CLUSTER_MIN_ENVS` where no environment-level interval exists.
RULES: tuple[str, ...] = ("superiority", "noninferiority", "not_better")
REGISTERED = REGISTRATIONS["h1b_prime"]
ALL_HORIZONS: tuple[int, ...] = (50, 100, 200, 500, 1000)

COLUMNS: tuple[str, ...] = (
    "row", "policy", "reference", "test", "horizon", "delta", "lo", "hi", "se", "win",
    "n_cells", "n_episodes", "n_envs", *ad.CLUSTER_KEYS, "p_holm", "mei", "rule", "verdict",
    "as_registered",
)


def _holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [np.nan] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def _diffs(out_dir: Path, test: str, policy: str, reference: str, horizons: tuple[int, ...]):
    """Per-episode paired differences by cell, with each cell's environment."""
    root = out_dir / "episodes" / test
    if not root.exists():
        raise FileNotFoundError(f"no episodes for test {test!r} under {out_dir}")
    col = f"regret_{PRIMARY_RECOMMENDER}"
    diffs: dict[str, np.ndarray] = {}
    env_of: dict[str, str] = {}
    horizon_of: dict[str, int] = {}
    for cell_dir in sorted(root.iterdir()):
        ref_path = cell_dir / f"{reference}.parquet"
        pol_path = cell_dir / f"{policy}.parquet"
        if not ref_path.exists():
            continue
        b = pd.read_parquet(ref_path)
        T = int(b["horizon"].iloc[0])
        if T not in horizons:
            continue
        if not pol_path.exists():
            raise FileNotFoundError(f"{policy} has no episodes in {test}/{cell_dir.name}")
        a = pd.read_parquet(pol_path)
        if not np.array_equal(a["episode"].to_numpy(), b["episode"].to_numpy()):
            raise ValueError(f"{cell_dir.name}: episode indices differ -- not paired")
        if int(a["base_seed"].iloc[0]) != int(b["base_seed"].iloc[0]):
            raise ValueError(f"{cell_dir.name}: base_seed differs -- not paired")
        diffs[cell_dir.name] = a[col].to_numpy(dtype=np.float64) - b[col].to_numpy(dtype=np.float64)
        env_of[cell_dir.name] = str(a["env_id"].iloc[0])
        horizon_of[cell_dir.name] = T
    if not diffs:
        raise FileNotFoundError(f"no cells of {test} at horizons {horizons} hold both {policy} and {reference}")
    return diffs, env_of, horizon_of


def _stratum(diffs: dict[str, np.ndarray], env_of: dict[str, str], *, n_boot: int, seed: int) -> dict:
    sp = stats.stratified_pooled(diffs, n_boot=n_boot, seed=seed)
    cm = pd.DataFrame({"env_id": [env_of[c] for c in diffs], "value": [sp["cell_means"][c] for c in diffs]})
    bounds = ad.cluster_bounds(cm, n_boot=n_boot, seed=seed + 1)
    return {
        "delta": sp["mean"], "lo": sp["lo"], "hi": sp["hi"], "se": sp["se"],
        "win": float(np.mean([ad.win_rate(d) for d in diffs.values()])),
        "n_cells": len(diffs), "n_episodes": int(sum(d.size for d in diffs.values())),
        "n_envs": bounds["n_envs"], **{k: bounds[k] for k in ad.CLUSTER_KEYS},
    }


def verdict(delta: float, cluster_lo: float, cluster_hi: float, cluster_p: float, mei: float) -> str:
    """The superiority rule of Pre-registrations 2-6, on the environment-mean t interval."""
    if not np.isfinite(cluster_p):
        return "inconclusive"
    if delta >= 0.0 or cluster_lo > -mei:
        return "refuted"
    if cluster_p < 0.05 and abs(delta) >= mei:
        return "supported"
    return "inconclusive"


def verdict_by_rule(rule: str, *, delta: float, lo: float, hi: float, mei: float, cluster_p: float = np.nan,
                    cluster_lo: float = np.nan, cluster_hi: float = np.nan) -> str:
    """`verdict` for ``superiority``; Pre-registration 8's paired-CI rules otherwise.

    ``noninferiority``: the policy is not worse than the reference by more than `mei` --
    supported iff the paired upper bound is below +mei, refuted iff the lower bound is at or
    above +mei. ``not_better``: the policy is not better than the reference by more than
    `mei` -- supported iff the paired lower bound is above -mei, refuted iff the upper
    bound is at or below -mei. Anything else is inconclusive.
    """
    if rule == "superiority":
        return verdict(delta, cluster_lo, cluster_hi, cluster_p, mei)
    if rule == "noninferiority":
        if hi < mei:
            return "supported"
        if lo >= mei:
            return "refuted"
        return "inconclusive"
    if rule == "not_better":
        if lo > -mei:
            return "supported"
        if hi <= -mei:
            return "refuted"
        return "inconclusive"
    raise KeyError(f"unknown rule {rule!r}; rules={RULES}")


def registered_contrast(
    policy: str, reference: str, *, test: str, horizons: tuple[int, ...], mei: float,
    out_dir: Path, n_boot: int = 10_000, rule: str = "superiority",
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    horizons = tuple(int(h) for h in horizons)
    if rule not in RULES:
        raise KeyError(f"unknown rule {rule!r}; rules={RULES}")
    as_registered = any(
        (policy, reference, test, horizons, float(mei), rule) == (
            r["policy"], r["reference"], r["test"], tuple(r["horizons"]), float(r["mei"]),
            r.get("rule", "superiority"))
        for r in REGISTRATIONS.values()
    )
    if not as_registered:
        log.warning("NOT the registered contrast: %s vs %s on %s at %s, mei %s",
                    policy, reference, test, horizons, mei)
    diffs, env_of, horizon_of = _diffs(out_dir, test, policy, reference, ALL_HORIZONS)
    base = {"policy": policy, "reference": reference, "test": test, "mei": float(mei), "rule": rule,
            "as_registered": as_registered}
    rows: list[dict] = []

    primary_cells = {c: d for c, d in diffs.items() if horizon_of[c] in horizons}
    if not primary_cells:
        raise FileNotFoundError(f"no cells of {test} at the registered horizons {horizons}")
    s = _stratum(primary_cells, env_of, n_boot=n_boot, seed=ad._seed("registered", policy, reference, test))
    rows.append({**base, "row": "primary", "horizon": "all", **s, "p_holm": np.nan,
                 "verdict": verdict_by_rule(rule, delta=s["delta"], lo=s["lo"], hi=s["hi"], mei=mei,
                                            cluster_p=s["cluster_p"], cluster_lo=s["cluster_lo"],
                                            cluster_hi=s["cluster_hi"])})

    secondary = []
    for T in horizons:
        cells_T = {c: d for c, d in diffs.items() if horizon_of[c] == T}
        if not cells_T:
            continue
        s = _stratum(cells_T, env_of, n_boot=n_boot, seed=ad._seed("registered", policy, reference, test, T))
        secondary.append({**base, "row": "secondary", "horizon": T, **s, "verdict": ""})
    ps = [r["cluster_p"] if np.isfinite(r["cluster_p"]) else 1.0 for r in secondary]
    for r, adj in zip(secondary, _holm(ps), strict=True):
        r["p_holm"] = adj
    rows.extend(secondary)

    for T in ALL_HORIZONS:
        if T in horizons:
            continue
        cells_T = {c: d for c, d in diffs.items() if horizon_of[c] == T}
        if not cells_T:
            continue
        s = _stratum(cells_T, env_of, n_boot=n_boot, seed=ad._seed("registered", policy, reference, test, T))
        rows.append({**base, "row": "structural", "horizon": T, **s, "p_holm": np.nan, "verdict": ""})
    return pd.DataFrame(rows)[list(COLUMNS)]


#: Pre-registration 9's flatness guard: below this many informative primary cells, every
#: contrast's verdict is "uninformative" -- K barely matters on the real pools.
FLATNESS_MIN_INFORMATIVE = 2
BOOT_TEST = "emp_boot"
_BOOT_ENV = re.compile(r"^(?P<base>emp_[A-Za-z]+_npmle)_b(?P<b>\d{3})$")
BOOT_COLUMNS: tuple[str, ...] = (
    "row", "policy", "reference", "test", "horizon", "delta", "lo", "hi", "paired_lo", "paired_hi",
    "n_informative", "informative_cells", "n_boot", "mei", "rule", "verdict", "as_registered",
)
#: The reservoir manifest `replay.py estimate` writes beside the frozen outcomes snapshot.
EMP_RESERVOIR_MANIFEST = ROOT / "data" / "empirical_pool" / "reservoirs" / "manifest.json"
#: `replay.py point`/`boot` stamp the manifest they ran against, per test, under ``tables/``.
RESERVOIR_STAMP = "{test}_reservoir_stamp.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_reservoir_stamp(out_dir: Path, test: str, manifest_path: Path) -> Path:
    """Record which reservoir manifest the `test` episodes were generated from."""
    path = Path(out_dir) / "tables" / RESERVOIR_STAMP.format(test=test)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"test": test, "manifest_sha256": file_sha256(manifest_path)}, indent=2))
    return path


def check_reservoir_snapshot(out_dir: Path, flatness: pd.DataFrame, manifest_path: Path,
                             horizons: tuple[int, ...]) -> None:
    """Raise unless every input to the verdict comes from the reservoirs in `manifest_path`.

    The flatness rows used (npmle, primary horizons) must carry the sha256 of the reservoir
    their K-grid ran on, equal to the manifest's; the ``emp`` and ``emp_boot`` episodes must be
    stamped with this manifest's sha256. A re-run of ``replay.py estimate`` after any of them
    therefore fails loudly instead of mixing two data snapshots in one verdict.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    shas = manifest["reservoirs"]
    rows = flatness[(flatness["variant"] == "npmle") & flatness["horizon"].isin(horizons)]
    if "reservoir_sha256" not in rows.columns:
        raise ValueError("emp_flatness.csv has no reservoir_sha256 column: re-run replay.py kgrid and describe.py")
    stale = sorted({str(r.env_id) for r in rows.itertuples()
                    if shas.get(str(r.env_id).removeprefix("emp_")) != str(r.reservoir_sha256)})
    if stale:
        raise ValueError(f"emp_flatness.csv rows {stale} were computed on reservoirs other than {manifest_path}; "
                         "re-run replay.py kgrid and describe.py")
    want = file_sha256(manifest_path)
    for test in ("emp", BOOT_TEST):
        stamp_path = Path(out_dir) / "tables" / RESERVOIR_STAMP.format(test=test)
        if not stamp_path.exists():
            raise ValueError(f"no {stamp_path.name}: the {test} episodes are not tied to a reservoir snapshot")
        if json.loads(stamp_path.read_text()).get("manifest_sha256") != want:
            raise ValueError(f"the {test} episodes were generated from a different reservoir manifest than "
                             f"{manifest_path}; re-run replay.py {'point' if test == 'emp' else 'boot'}")


def _informative_cells(flatness_path: Path, horizons: tuple[int, ...]) -> set[tuple[str, int]]:
    flat = pd.read_csv(flatness_path)
    flat = flat[(flat["variant"] == "npmle") & flat["horizon"].isin(horizons)]
    return {(str(r.env_id), int(r.horizon)) for r in flat.itertuples() if bool(r.informative)}


def prompt_bootstrap_contrast(
    policy: str, reference: str, *, horizons: tuple[int, ...], mei: float, rule: str, out_dir: Path,
    expected_n_boot: int, flatness_path: Path | None = None, reservoir_manifest: Path | None = None,
    paired_n_boot: int = 10_000,
) -> pd.DataFrame:
    """Pre-registration 9: the full-sample Δ on informative primary cells, with the 95% percentile
    interval of the same statistic over the prompt-bootstrap replicates (test ``emp_boot``).

    Pre-registration 9 fixes B = `expected_n_boot` (200) replicates. The replicate indices
    present must be exactly ``{0, ..., expected_n_boot - 1}`` -- trailing replicates lost to a
    boot run that died early, or an `emp_boot` tree with no data at all, raise rather than
    silently inferring a smaller B or falling through to an "inconclusive" verdict. Cell
    coverage (both `emp` and `emp_boot`) is checked by key, not count: the informative cells
    used must be exactly the informative set, and two cells claiming the same (env_id, horizon)
    raise instead of one silently overwriting the other.

    The episode-paired CI (the cell-stratified paired bootstrap over the informative cells'
    episode differences, as `_stratum` computes it) is reported beside the interval of record
    as ``paired_lo``/``paired_hi``; it does not enter the verdict. With `reservoir_manifest`
    (the CLI always passes it) every input is first checked against that one data snapshot
    (`check_reservoir_snapshot`).
    """
    out_dir = Path(out_dir)
    horizons = tuple(int(h) for h in horizons)
    flatness_path = flatness_path or out_dir / "tables" / "emp_flatness.csv"
    if reservoir_manifest is not None:
        check_reservoir_snapshot(out_dir, pd.read_csv(flatness_path), reservoir_manifest, horizons)
    informative = _informative_cells(flatness_path, horizons)
    as_registered = any(
        (policy, reference, horizons, float(mei), rule) == (r["policy"], r["reference"], tuple(r["horizons"]),
                                                             float(r["mei"]), r.get("rule"))
        for r in REGISTRATIONS.values() if r.get("interval") == "prompt_bootstrap"
    )
    row = {"row": "primary", "policy": policy, "reference": reference, "test": "emp", "horizon": "all",
           "n_informative": len(informative), "informative_cells": ";".join(f"{e}@{t}" for e, t in sorted(informative)),
           "mei": float(mei), "rule": rule, "as_registered": as_registered,
           "delta": np.nan, "lo": np.nan, "hi": np.nan, "paired_lo": np.nan, "paired_hi": np.nan, "n_boot": 0}
    if len(informative) < FLATNESS_MIN_INFORMATIVE:
        row["verdict"] = "uninformative"
        return pd.DataFrame([row])[list(BOOT_COLUMNS)]

    diffs, env_of, horizon_of = _diffs(out_dir, "emp", policy, reference, horizons)
    point_by_key: dict[tuple[str, int], float] = {}
    paired_cells: dict[str, np.ndarray] = {}
    for c, d in diffs.items():
        key = (env_of[c], horizon_of[c])
        if key not in informative:
            continue
        if key in point_by_key:
            raise ValueError(f"emp holds two cells for the informative key {key} (last: {c})")
        point_by_key[key] = float(d.mean())
        paired_cells[c] = d
    missing_point = sorted(informative - set(point_by_key))
    if missing_point:
        raise ValueError(f"emp is missing informative cells {missing_point}")
    point = [point_by_key[k] for k in sorted(informative)]

    bdiffs, benv, bhor = _diffs(out_dir, BOOT_TEST, policy, reference, horizons)
    by_boot: dict[int, dict[tuple[str, int], float]] = {}
    for c, d in bdiffs.items():
        m = _BOOT_ENV.match(benv[c])
        if m is None:
            raise ValueError(f"{c}: env id {benv[c]!r} is not a bootstrap replicate")
        key = (m.group("base"), bhor[c])
        if key not in informative:
            continue
        b = int(m.group("b"))
        cells_b = by_boot.setdefault(b, {})
        if key in cells_b:
            raise ValueError(f"emp_boot holds two cells for b{b:03d}'s informative key {key} (last: {c})")
        cells_b[key] = float(d.mean())

    present, expected = set(by_boot), set(range(expected_n_boot))
    if present != expected:
        parts = []
        missing_b = sorted(expected - present)
        extra_b = sorted(present - expected)
        if missing_b:
            parts.append(f"missing {[f'b{b:03d}' for b in missing_b]}")
        if extra_b:
            parts.append(f"unexpected {[f'b{b:03d}' for b in extra_b]}")
        raise ValueError(f"emp_boot does not hold exactly replicates b000..b{expected_n_boot - 1:03d}: "
                         + "; ".join(parts))

    stats_b = []
    for b in range(expected_n_boot):
        cells_b = by_boot[b]
        missing_cells = sorted(informative - set(cells_b))
        if missing_cells:
            raise ValueError(f"bootstrap replicate b{b:03d} lacks informative cells {missing_cells}")
        stats_b.append(float(np.mean([cells_b[k] for k in sorted(informative)])))

    delta = float(np.mean(point))
    lo, hi = float(np.percentile(stats_b, 2.5)), float(np.percentile(stats_b, 97.5))
    paired = stats.stratified_pooled(paired_cells, n_boot=paired_n_boot,
                                     seed=ad._seed("registered", policy, reference, "emp"))
    row.update({"delta": delta, "lo": lo, "hi": hi, "paired_lo": float(paired["lo"]),
                "paired_hi": float(paired["hi"]), "n_boot": expected_n_boot,
                "verdict": verdict_by_rule(rule, delta=delta, lo=lo, hi=hi, mei=mei)})
    return pd.DataFrame([row])[list(BOOT_COLUMNS)]


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registration", default="h1b_prime", choices=sorted(REGISTRATIONS),
                    help="which pre-registered contrast to compute (its defaults fill the rest)")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--mei", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=None, help="default: <out-dir>/tables/<registration>.csv")
    ap.add_argument("--reservoir-manifest", type=Path, default=EMP_RESERVOIR_MANIFEST,
                    help="prompt-bootstrap registrations: the snapshot every input must match")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    reg = REGISTRATIONS[args.registration]
    for key in ("policy", "reference", "test", "mei"):
        if getattr(args, key) is None:
            setattr(args, key, reg[key])
    horizons = (tuple(int(x) for x in args.horizons.split(",") if x.strip())
                if args.horizons else tuple(reg["horizons"]))
    if reg.get("interval") == "prompt_bootstrap":
        out = prompt_bootstrap_contrast(args.policy, args.reference, horizons=horizons, mei=args.mei,
                                        rule=reg["rule"], out_dir=args.out_dir, expected_n_boot=reg["n_boot"],
                                        reservoir_manifest=args.reservoir_manifest)
        path = args.out or (args.out_dir / "tables" / reg["out"])
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        p = out.iloc[0]
        log.info("%s: %s vs %s on emp (informative %d): delta=%+.6f prompt-bootstrap [%+.6f, %+.6f] B=%d "
                 "(episode-paired [%+.6f, %+.6f]) -> %s", args.registration, args.policy, args.reference,
                 p["n_informative"], p["delta"], p["lo"], p["hi"], p["n_boot"], p["paired_lo"], p["paired_hi"],
                 str(p["verdict"]).upper())
        log.info("wrote %s", path)
        return out
    out = registered_contrast(args.policy, args.reference, test=args.test, horizons=horizons,
                              mei=args.mei, out_dir=args.out_dir, n_boot=args.n_boot,
                              rule=reg.get("rule", "superiority"))
    path = args.out or (args.out_dir / "tables" / reg["out"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    p = out[out["row"] == "primary"].iloc[0]
    log.info("%s: %s vs %s on %s at T in %s: delta=%+.6f paired [%+.6f, %+.6f] cluster t [%+.6f, %+.6f] "
             "p=%.3g sign p=%s n_envs=%d n_cells=%d  ->  %s",
             args.registration, args.policy, args.reference, args.test, horizons, p["delta"], p["lo"], p["hi"],
             p["cluster_lo"], p["cluster_hi"], p["cluster_p"], p["cluster_p_sign"], p["n_envs"], p["n_cells"],
             p["verdict"].upper())
    for _, r in out[out["row"] != "primary"].iterrows():
        log.info("  %-10s T=%-5s delta=%+.6f cluster [%+.6f, %+.6f] p=%.3g holm=%s",
                 r["row"], r["horizon"], r["delta"], r["cluster_lo"], r["cluster_hi"], r["cluster_p"],
                 "-" if not np.isfinite(r["p_holm"]) else f"{r['p_holm']:.3g}")
    log.info("wrote %s", path)
    return out


if __name__ == "__main__":
    main()
