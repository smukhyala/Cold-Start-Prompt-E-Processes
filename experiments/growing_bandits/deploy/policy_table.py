"""The deployment study's policy table as data (DEPLOYMENT_PLAN.md "Policies").

`rules.POLICY_SPECS` knows how to *build* every kind of SEARCH policy; this module says
which concrete configurations the study deploys, under which name, and where each
tuned constant comes from. Keeping that as one table means the runner, the summary
tables and the M7 analysis all agree on what "phi_k16" is, and that no tuned value is
typed twice.

Every entry is ``{"kind", "params", "group", "requires"}``:

* ``kind`` is a `rules.KINDS` member.
* ``params`` are the constructor overrides. A value of ``None`` is a slot that is
  filled *per horizon* at resolve time from the study's tuning outputs (M5):
  ``baseline_params.json`` for the schedule constants (``c`` per ``(alpha, T)``,
  ``K0`` per ``T``, the validation-selected ``p3_star`` per ``T``) and
  ``thresholds.json`` for every learned threshold (``tau_val``). Because the smoke run
  happens before tuning, both files may be absent: the slot then takes the registered
  placeholder (`rules.POLICY_SPECS`) or the artifact's own offline ``tau``, and the
  substitution is logged once per (policy, reason) *and* stamped into the resolved
  params as ``params_tuned: False`` (`PARAMS_TUNED`) -- never silently. That stamp
  reaches the run manifest, so the analysis can refuse to pool a cell whose baseline
  was an untuned placeholder rather than a tuned comparator.
* ``group`` is ``baseline`` (P0-P3), ``reference`` (``cp0``, ``p3_star``: the two
  pre-registered comparison targets), ``learned`` (a saved model) or ``rule`` (the
  P11 hand rule).
* ``requires`` names the external inputs the entry depends on, so a runner can check
  them up front and decide, for instance, whether a pairwise log-e table must be built
  before workers are forked.

`resolve_params` turns an entry into concrete constructor parameters for one horizon;
`build_policy` constructs the `SearchPolicy`; `resolve_policy` does both and returns
the study's per-policy seed component ``crc32(name)`` (the plan's convention, keyed on
the *study* name so ``phi_k16`` and ``phi_k16_perstep`` never share a stream).
"""

from __future__ import annotations

import json
import logging
import zlib
from pathlib import Path
from typing import Any

from cold_start.growing.deploy import rules
from cold_start.growing.deploy.artifacts import load_model
from cold_start.growing.deploy.feature_groups import EVIDENCE_LOGE
from cold_start.growing.search_policies import SearchPolicy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS_DIR = ROOT / "results" / "growing_bandits" / "deploy"
DEFAULT_MODELS_DIR = RESULTS_DIR / "models"
DEFAULT_BASELINE_PARAMS_PATH = RESULTS_DIR / "baseline_params.json"
DEFAULT_THRESHOLDS_PATH = RESULTS_DIR / "thresholds.json"

log = logging.getLogger("deploy.policy_table")

GROUPS: tuple[str, ...] = ("baseline", "reference", "learned", "rule")

# The four P3 exponents, keyed by the short label used in the policy name. The value
# is the exact float the schedule is built with; `baseline_params.json` keys are
# matched to it numerically (JSON stringifies floats).
POWER_ALPHAS: dict[str, float] = {
    "0.25": 0.25,
    "0.33": 1.0 / 3.0,
    "0.5": 0.5,
    "0.67": 2.0 / 3.0,
}

# Registered placeholder for each exponent when `baseline_params.json` is absent.
_PLACEHOLDER_POWER_SPEC: dict[str, str] = {
    "0.25": "power_quarter",
    "0.33": "power_cbrt",
    "0.5": "power_sqrt",
    "0.67": "power_t23",
}
_PLACEHOLDER_K0 = int(rules.POLICY_SPECS["refine_after_init_K4"]["K0"])
_PLACEHOLDER_P3_STAR = {
    "alpha": float(rules.POLICY_SPECS["power_sqrt"]["alpha"]),
    "c": float(rules.POLICY_SPECS["power_sqrt"]["c"]),
}
_PLACEHOLDER_RULE_TAU = float(rules.POLICY_SPECS["reservoir_rule"]["tau"])
_PLACEHOLDER_FIXED_K = int(rules.POLICY_SPECS["fixed_K16"]["K"])

# Which `rules.POLICY_SPECS` entry each kind is built through; `params` override the
# rest, so only the kind of the registered entry matters here.
_SPEC_FOR_KIND: dict[str, str] = {
    "always_search": "always_search",
    "refine_after_init": "refine_after_init_K2",
    "uniform": "uniform",
    "fixed_K": "fixed_K16",
    "power": "power_sqrt",
    "cp0": "cp0",
    "reservoir_rule": "reservoir_rule",
    "model": "model",
}

# The M4 variant behind every learned policy (`train_policies.VARIANTS` names).
CQE = "clock_quality_evidence"


def _learned(
    variant: str,
    *,
    tau: float | None = None,
    per_step: bool = False,
    affordability_guard: bool = False,
) -> dict[str, Any]:
    requires = [f"artifact:{variant}"]
    if tau is None:
        requires.append("thresholds")
    return {
        "kind": "model",
        "params": {
            "artifact": variant,
            "tau": tau,
            "k": None,
            "per_step": per_step,
            "affordability_guard": affordability_guard,
        },
        "group": "learned",
        "requires": requires,
    }


def _power(alpha_label: str) -> dict[str, Any]:
    return {
        "kind": "power",
        "params": {"alpha": POWER_ALPHAS[alpha_label], "c": None},
        "group": "baseline",
        "requires": ["baseline_params"],
    }


POLICIES: dict[str, dict[str, Any]] = {
    # ---- P0-P3: schedules ---------------------------------------------------------
    "always_search": {"kind": "always_search", "params": {}, "group": "baseline", "requires": []},
    "refine_after_init": {
        "kind": "refine_after_init",
        "params": {"K0": None},
        "group": "baseline",
        "requires": ["baseline_params"],
    },
    "uniform": {"kind": "uniform", "params": {}, "group": "baseline", "requires": []},
    "fixed_K16": {"kind": "fixed_K", "params": {"K": 16}, "group": "baseline", "requires": []},
    # Pre-registration 3's null model: recruit to K(T) immediately, then refine; K(T) the
    # validation argmin per horizon (`select_fixed_k.py`, `baseline_params.json["fixed_K_star"]`).
    "fixed_K_star": {
        "kind": "fixed_K",
        "params": {"K": None},
        "group": "baseline",
        "requires": ["baseline_params"],
    },
    "power_a0.25": _power("0.25"),
    "power_a0.33": _power("0.33"),
    "power_a0.5": _power("0.5"),
    "power_a0.67": _power("0.67"),
    # ---- the two pre-registered references -------------------------------------
    # P3*: the (alpha, c) pair selected per horizon on validation seeds (H1b).
    "p3_star": {
        "kind": "power",
        "params": {"alpha": None, "c": None},
        "group": "reference",
        "requires": ["baseline_params"],
    },
    # The label continuation policy (H1a); its constants are the corpus's, untuned.
    "cp0": {
        "kind": "cp0",
        "params": {"alpha": 0.5, "c": 1.0, "min_pulls_per_arm": 2},
        "group": "reference",
        "requires": [],
    },
    # ---- learned: commitment ladder (P4, P5, P6 = P9) ----------------------------
    "phi_k1": _learned(f"{CQE}_k1"),
    "phi_k4": _learned(f"{CQE}_k4"),
    "phi_k16": _learned(f"{CQE}_k16"),
    "phi_k16_tau05": _learned(f"{CQE}_k16", tau=0.5),
    # P6' and P6g: same model, different deployment mechanics.
    "phi_k16_perstep": _learned(f"{CQE}_k16", per_step=True),
    "phi_k16_guard": _learned(f"{CQE}_k16", affordability_guard=True),
    # ---- learned: feature-set ladder (P8, P7, P9a, P10) ---------------------------
    "phi_k16_clock": _learned("clock_k16"),
    "phi_k16_quality": _learned("clock_quality_k16"),
    "phi_k16_cs": _learned("clock_quality_cs_k16"),
    "phi_k16_all71": _learned("all71_k16"),
    # ---- learned: estimator / weighting / row-filter sensitivities (P12, audit C) ---
    "phi_k16_hgb": _learned(f"{CQE}_k16_hgb"),
    "phi_k16_weighted": _learned(f"{CQE}_k16_weighted"),
    "phi_k16_noambig": _learned(f"{CQE}_k16_noambig"),
    "phi_k16_notrunc": _learned(f"{CQE}_k16_notrunc"),
    "phi_k16_lucb": _learned(f"{CQE}_k16_lucb"),
    # ---- P11: reservoir-aware rule, hand form and logistic form ---------------------
    "rule_reservoir": {
        "kind": "reservoir_rule",
        "params": {"tau": None},
        "group": "rule",
        "requires": ["thresholds"],
    },
    "phi_reservoir_rule": _learned("reservoir_rule_k16"),
    # ---- generalization (Tests B, C, D) ----------------------------------------------
    "phi_k16_nopolicy": _learned(f"{CQE}_k16_nopolicy"),
    "phi_k16_famA_only": _learned(f"{CQE}_k16_famA_only"),
    "phi_k16_famB_only": _learned(f"{CQE}_k16_famB_only"),
    "phi_k16_noT1000": _learned(f"{CQE}_k16_noT1000"),
    "phi_k16_noT200": _learned(f"{CQE}_k16_noT200"),
    "phi_sf_k16": _learned("clock_sf_quality_cs_k16"),
    "phi_sf_k16_noT1000": _learned("clock_sf_quality_cs_k16_noT1000"),
    # ---- M9: one policy-iteration step (relabel_onpolicy.py) -------------------------
    # phi_k16 refitted on the corpus plus its own on-policy labels, and on those alone;
    # same commitment (k from the artifact, 16) and mechanics as phi_k16.
    "phi_k16_onpolicy_union": _learned("phi_k16_onpolicy_union"),
    "phi_k16_onpolicy_only": _learned("phi_k16_onpolicy_only"),
}

ALL_POLICIES: tuple[str, ...] = tuple(POLICIES)
#: The M9 models exist only after `relabel_onpolicy.py train`; they deploy on Test A alone.
ONPOLICY_POLICIES: tuple[str, ...] = ("phi_k16_onpolicy_union", "phi_k16_onpolicy_only")
#: The table every other test deploys from: everything trained on the corpus alone.
CORPUS_POLICIES: tuple[str, ...] = tuple(p for p in ALL_POLICIES if p not in ONPOLICY_POLICIES)

_ROBUST: tuple[str, ...] = (
    "always_search",
    "refine_after_init",
    "p3_star",
    "cp0",
    "phi_k16_quality",
    "phi_k16",
)

#: Which policies each test deploys. A runs the whole table (M9's on-policy models
#: included); C (held-out family) runs the corpus-trained table; B and D run the models
#: trained for them plus the references; the robustness and cap sweeps run the six the
#: plan names.
TEST_POLICIES: dict[str, tuple[str, ...]] = {
    "A": ALL_POLICIES,
    "C": CORPUS_POLICIES,
    "B": ("phi_k16", "phi_k16_nopolicy", "phi_k16_all71", "cp0", "p3_star"),
    "D": (
        "phi_k16",
        "phi_k16_clock",
        "phi_k16_noT1000",
        "phi_k16_noT200",
        "phi_sf_k16",
        "phi_sf_k16_noT1000",
        "always_search",
        "refine_after_init",
        "cp0",
        "p3_star",
    ),
    "robust": _ROBUST,
    "cap": _ROBUST,
    "smoke": CORPUS_POLICIES,
    # The K-matched control deploys these plus the ``--match`` policy (run_deployment.py).
    "capmatch": ("always_search", "uniform"),
}


# ---- tuning outputs ----------------------------------------------------------------


def load_json_or_none(path: str | Path, what: str) -> dict | None:
    """Load a tuning output, or return ``None`` with a logged warning if it is absent.

    The runner must not crash before tuning has happened (the smoke run precedes
    M5), but it must also never *silently* deploy placeholders, hence the warning.
    """
    p = Path(path)
    if not p.exists():
        log.warning("%s not found at %s; placeholders will be used", what, p)
        return None
    with open(p) as fh:
        return json.load(fh)


def load_baseline_params(path: str | Path = DEFAULT_BASELINE_PARAMS_PATH) -> dict | None:
    return load_json_or_none(path, "baseline_params.json")


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS_PATH) -> dict | None:
    return load_json_or_none(path, "thresholds.json")


def tuning_cap(baseline_params: dict | None) -> int | None:
    """The live-arm cap `baseline_params` was tuned under (``meta.cap``), or ``None``.

    ``None`` means the file does not say, in which case nothing is claimed about the cap
    and no item is marked untuned on that ground -- silence is not evidence of a match.
    """
    meta = (baseline_params or {}).get("meta") or {}
    cap = meta.get(TUNING_CAP_KEY)
    return None if cap is None else int(cap)


def baseline_params_for_cap(
    baseline_params: dict | None, cap: int | None
) -> tuple[dict | None, int | None]:
    """``(the tuning block that applies at `cap`, the cap it was tuned under)``.

    Today's file has one block, tuned at ``meta.cap``. A per-cap retune adds a `BY_CAP_KEY`
    mapping ``"<cap>" -> {power, p3_star, refine_after_init}``; when it holds `cap`, that
    block is the tuned one and there is no mismatch. The file on disk is unchanged by
    this: `migrate_baseline_params` is the (reversible) writer side, and nothing here
    requires it to have been run.
    """
    if baseline_params is None:
        return None, None
    by_cap = baseline_params.get(BY_CAP_KEY) or {}
    if cap is not None:
        block = _lookup_by_number(by_cap, int(cap))
        if block:
            return {**baseline_params, **block}, int(cap)
    return baseline_params, tuning_cap(baseline_params)


def migrate_baseline_params(baseline_params: dict) -> dict:
    """Move the top-level tuned blocks under ``by_cap["<meta.cap>"]``, losing nothing.

    The forward half of the per-cap schema. It is pure and in memory: the shipped
    ``baseline_params.json`` is NOT rewritten, so the M8fix audit anchor -- the file is
    byte-identical to its pre-T=2000 state once the six T=2000 keys are stripped -- stays
    checkable against the real file (`test_baseline_params_migration_is_reversible`).
    """
    cap = tuning_cap(baseline_params)
    if cap is None:
        raise ValueError("baseline_params has no meta.cap; it cannot be keyed by cap")
    if BY_CAP_KEY in baseline_params:
        raise ValueError("baseline_params is already keyed by cap")
    block = {k: v for k, v in baseline_params.items() if k in TUNED_BLOCKS}
    out: dict[str, Any] = {}
    for key, value in baseline_params.items():
        if key in TUNED_BLOCKS:
            out.setdefault(BY_CAP_KEY, {str(cap): block})
            continue
        out[key] = value
    out.setdefault(BY_CAP_KEY, {str(cap): block})
    return out


def unmigrate_baseline_params(baseline_params: dict) -> dict:
    """The exact inverse of `migrate_baseline_params`, key order included."""
    cap = tuning_cap(baseline_params)
    by_cap = baseline_params.get(BY_CAP_KEY)
    if cap is None or by_cap is None:
        raise ValueError("baseline_params is not keyed by cap")
    block = _lookup_by_number(by_cap, cap)
    if block is None:
        raise ValueError(f"by_cap has no block for the tuning cap {cap}")
    if set(by_cap) != {str(cap)}:
        raise ValueError(f"by_cap holds more than the tuning cap: {sorted(by_cap)}")
    out: dict[str, Any] = {}
    for key, value in baseline_params.items():
        if key == BY_CAP_KEY:
            out.update(block)
        else:
            out[key] = value
    return out


def _lookup_by_number(table: dict, key: float, tol: float = 1e-6):
    """Fetch ``table[key]`` where the JSON keys are stringified numbers."""
    for k, v in table.items():
        try:
            if abs(float(k) - float(key)) <= tol:
                return v
        except (TypeError, ValueError):
            continue
    return None


_warned: set[tuple[str, str]] = set()


def _warn_once(name: str, reason: str) -> None:
    key = (name, reason)
    if key in _warned:
        return
    _warned.add(key)
    log.warning("policy %s: %s", name, reason)


def _alpha_label(alpha: float) -> str:
    for label, value in POWER_ALPHAS.items():
        if abs(value - alpha) <= 1e-9:
            return label
    raise KeyError(f"alpha {alpha} is not one of the study's exponents {POWER_ALPHAS}")


#: Stamped into the resolved params when a schedule constant fell back to its registered
#: placeholder because `baseline_params.json` has no tuned value for that horizon, plus
#: the values that were substituted. A warning line scrolls away; these keys travel into
#: the run manifest and from there into the analysis, so a placeholder can never be
#: mistaken for a tuned constant (register #7, finding B1).
#:
#: The keys are stamped *only* on a fallback. "Tuned" stays the unmarked case so a tuned
#: item's params -- and therefore its manifest record and `run_deployment.stale_reason` --
#: are byte-identical to what every earlier run wrote.
PARAMS_TUNED = "params_tuned"
PARAMS_FALLBACK = "params_fallback"
#: The live-arm cap the schedule constants were actually tuned under, stamped ONLY when
#: it differs from the cap the item deploys at. `baseline_params.json` tunes at one cap
#: (`meta.cap`, 64 for the shipped file): every constant in it was selected under that
#: many live arms, and a cell that deploys the same constant at cap 32, 128 or T is not
#: running a tuned comparator. `main_cap_primary.csv` asserted ``params_tuned = True`` on
#: cells tuned at a different cap, which is a false assertion however the retune goes
#: (finding 4.5 / ruling 20).
#:
#: Stamped only on a mismatch, exactly like `PARAMS_TUNED`: the unmarked case stays
#: "tuned at this cap", so a matching item's params -- and therefore its manifest record
#: and `run_deployment.stale_reason` -- are byte-identical to what every earlier run
#: wrote.
PARAMS_CAP = "params_cap"
#: Provenance recorded in the params that is not a policy constructor argument.
PROVENANCE_KEYS: tuple[str, ...] = ("tau_source", PARAMS_TUNED, PARAMS_FALLBACK, PARAMS_CAP)

#: The cap `baseline_params.json` records its tuning under, and the optional per-cap
#: block a future retune would add (`baseline_params_for_cap`).
TUNING_CAP_KEY = "cap"
BY_CAP_KEY = "by_cap"
#: The blocks a per-cap retune would have to duplicate; everything else is metadata.
TUNED_BLOCKS: tuple[str, ...] = ("power", "p3_star", "refine_after_init", "fixed_K_star")


def _tuned_c(
    name: str, alpha: float, horizon: int, baseline_params: dict | None
) -> tuple[float, dict | None]:
    """``(c, fallback)`` for ``(alpha, T)`` from ``baseline_params["power"]``.

    `fallback` is ``None`` when the value is the tuned one, else the placeholder values
    that were substituted (see `PARAMS_FALLBACK`).
    """
    label = _alpha_label(alpha)
    placeholder = float(rules.POLICY_SPECS[_PLACEHOLDER_POWER_SPEC[label]]["c"])
    if baseline_params is None:
        _warn_once(name, f"no baseline_params; using placeholder c={placeholder}")
        return placeholder, {"c": placeholder}
    by_t = _lookup_by_number(baseline_params.get("power", {}), alpha)
    c = _lookup_by_number(by_t, horizon) if by_t else None
    if c is None:
        _warn_once(name, f"no tuned c for (alpha={alpha:.4f}, T={horizon}); placeholder {placeholder}")
        return placeholder, {"c": placeholder}
    return float(c), None


def _tuned_k0(name: str, horizon: int, baseline_params: dict | None) -> tuple[int, dict | None]:
    if baseline_params is None:
        _warn_once(name, f"no baseline_params; using placeholder K0={_PLACEHOLDER_K0}")
        return _PLACEHOLDER_K0, {"K0": _PLACEHOLDER_K0}
    k0 = _lookup_by_number(baseline_params.get("refine_after_init", {}), horizon)
    if k0 is None:
        _warn_once(name, f"no tuned K0 for T={horizon}; placeholder K0={_PLACEHOLDER_K0}")
        return _PLACEHOLDER_K0, {"K0": _PLACEHOLDER_K0}
    return int(k0), None


def _tuned_p3_star(
    name: str, horizon: int, baseline_params: dict | None
) -> tuple[tuple[float, float], dict | None]:
    ph = _PLACEHOLDER_P3_STAR
    if baseline_params is None:
        _warn_once(name, f"no baseline_params; using placeholder P3*={ph}")
        return (ph["alpha"], ph["c"]), dict(ph)
    sel = _lookup_by_number(baseline_params.get("p3_star", {}), horizon)
    if not sel or "alpha" not in sel or "c" not in sel:
        _warn_once(name, f"no validation-selected P3* for T={horizon}; placeholder {ph}")
        return (ph["alpha"], ph["c"]), dict(ph)
    return (float(sel["alpha"]), float(sel["c"])), None


def _tuned_fixed_k(name: str, horizon: int, baseline_params: dict | None) -> tuple[int, dict | None]:
    ph = {"K": _PLACEHOLDER_FIXED_K}
    if baseline_params is None:
        _warn_once(name, f"no baseline_params; using placeholder K={_PLACEHOLDER_FIXED_K}")
        return _PLACEHOLDER_FIXED_K, ph
    sel = _lookup_by_number(baseline_params.get("fixed_K_star", {}), horizon)
    if not sel or "K" not in sel:
        _warn_once(name, f"no validation-selected K for T={horizon}; placeholder {_PLACEHOLDER_FIXED_K}")
        return _PLACEHOLDER_FIXED_K, ph
    return int(sel["K"]), None


def _stamp_untuned(params: dict[str, Any], fallback: dict | None) -> None:
    """Record a placeholder substitution in the params it was substituted into."""
    if fallback is None:
        return
    params[PARAMS_TUNED] = False
    params.setdefault(PARAMS_FALLBACK, {}).update(fallback)


def _stamp_cap(name: str, params: dict[str, Any], cap: int | None, tuned_cap: int | None) -> None:
    """Record that the constants in `params` were tuned at a different cap than `cap`.

    Only a slot that came out of `baseline_params.json` can be mis-capped, so this is
    applied to `BASELINE_PARAM_KINDS` alone; a learned policy's tau is tracked by
    `TAU_SOURCES`. Nothing is stamped when either cap is unknown or the two agree, which
    keeps every cap-64 item byte-identical to what earlier runs recorded.
    """
    if cap is None or tuned_cap is None or int(cap) == int(tuned_cap):
        return
    _warn_once(name, f"constants tuned at cap {tuned_cap} but deploying at cap {cap}")
    params[PARAMS_CAP] = int(tuned_cap)
    params[PARAMS_TUNED] = False


def params_are_tuned(params: dict[str, Any] | None) -> bool:
    """Whether `params` (from `resolve_params` or a manifest record) is fully tuned."""
    return bool((params or {}).get(PARAMS_TUNED, True))


#: The kinds whose constructor constants come from `baseline_params.json` (M5). Only
#: these can be "untuned" in the sense of `PARAMS_TUNED`; a learned policy's threshold
#: comes from `thresholds.json` and is tracked by `TAU_SOURCES` instead.
BASELINE_PARAM_KINDS: frozenset[str] = frozenset({"power", "refine_after_init"})


def _reads_baseline_params(entry: dict[str, Any]) -> bool:
    """Whether a table entry fills a slot from `baseline_params.json` (and so can be
    untuned or mis-capped): every `BASELINE_PARAM_KINDS` entry, plus `fixed_K_star`, the
    one `fixed_K` whose K is a ``None`` slot (a fixed integer like `fixed_K16` is not)."""
    return entry["kind"] in BASELINE_PARAM_KINDS or (
        entry["kind"] == "fixed_K" and entry["params"].get("K") is None
    )


def constructor_params(params: dict | None) -> dict:
    """`params` without the provenance keys: what actually reaches the policy constructor.

    Provenance (`PROVENANCE_KEYS`) describes how a constant was chosen, not what it is,
    so two params dicts that differ only there produce bit-identical episodes.
    """
    return {k: v for k, v in (params or {}).items() if k not in PROVENANCE_KEYS}


def deployed_params_are_current(
    name: str,
    horizon: int,
    recorded: dict | None,
    baseline_params: dict | None,
    tol: float = 1e-12,
    *,
    cap: int | None = None,
) -> bool:
    """Whether `recorded` (a manifest's params) is what this table resolves today.

    A constant that was a placeholder when the cell ran, and has since been tuned, leaves
    the tuning outputs looking complete while the episodes on disk were produced by the
    placeholder. `run_deployment.stale_reason` catches that on the next ``--resume``; the
    analysis needs to catch it too, because it reads the episodes, not the runner
    (finding B1). Kinds outside `BASELINE_PARAM_KINDS` are not judged here and return True.
    """
    entry = POLICIES.get(name)
    if entry is None or not _reads_baseline_params(entry) or recorded is None:
        return True
    fresh = constructor_params(resolve_params(name, horizon, cap=cap, baseline_params=baseline_params))
    have = constructor_params(recorded)
    if set(have) != set(fresh):
        return False
    return all(abs(float(have[k]) - float(fresh[k])) <= tol for k in fresh)


def baseline_is_tuned(
    name: str, horizon: int, baseline_params: dict | None, cap: int | None = None
) -> bool:
    """Whether every `baseline_params.json` slot of `name` at `horizon` holds a tuned value.

    `resolve_params` stamps `PARAMS_TUNED` on the item it resolves, but a manifest written
    before that key existed carries no such mark. The analysis re-derives the answer from
    the same table and the same JSON -- which is also what a rerun today would deploy --
    so an old manifest cannot hide an untuned baseline.

    `cap` extends that to the live-arm cap: every shipped manifest line predates
    `PARAMS_CAP`, so the cap sweep's mis-capped cells can only be found by re-deriving
    them here (`_stamp_cap`).
    """
    entry = POLICIES.get(name)
    if entry is None:
        return True
    kind, slots = entry["kind"], entry["params"]
    if _reads_baseline_params(entry) and cap is not None:
        _, tuned_cap = baseline_params_for_cap(baseline_params, cap)
        if tuned_cap is not None and int(tuned_cap) != int(cap):
            return False
    baseline_params = baseline_params_for_cap(baseline_params, cap)[0]
    if kind == "power":
        if slots["alpha"] is None:  # p3_star
            return _tuned_p3_star(name, int(horizon), baseline_params)[1] is None
        if slots["c"] is None:
            return _tuned_c(name, float(slots["alpha"]), int(horizon), baseline_params)[1] is None
    elif kind == "refine_after_init" and slots["K0"] is None:
        return _tuned_k0(name, int(horizon), baseline_params)[1] is None
    elif kind == "fixed_K" and slots["K"] is None:
        return _tuned_fixed_k(name, int(horizon), baseline_params)[1] is None
    return True


#: Where a deployed threshold came from (recorded in ``params["tau_source"]``).
TAU_SOURCES: tuple[str, ...] = (
    "tau_val_excl_heldout",  # validation, excluding the variant's held-out horizons
    "tau_val",  # validation over every horizon
    "artifact_tau",  # the artifact's offline tau_off (no thresholds.json entry)
    "registered",  # the hand rule's registered placeholder
    "fixed",  # set by the policy table itself (the tau05 twin)
)


def heldout_horizons(artifact: dict | None) -> tuple[int, ...]:
    """Horizons the trainer excluded from this variant (``meta.subset.exclude_horizons``)."""
    meta = (artifact or {}).get("meta") or {}
    subset = meta.get("subset") or {}
    return tuple(int(h) for h in subset.get("exclude_horizons", ()))


def _tuned_tau(
    name: str,
    variant: str,
    thresholds: dict | None,
    fallback: float | None,
    fallback_source: str,
    *,
    horizon_holdout: bool = False,
) -> tuple[float | None, str]:
    """``(tau, source)`` for `variant`.

    A horizon-holdout model (trained without T=1000, say) must deploy the threshold
    selected *without* that horizon's validation cells -- ``tau_val_excl_heldout`` --
    wherever it runs, or its transfer test would have peeked at the held-out horizon
    through tau. Every other model deploys ``tau_val``. Missing keys fall back in
    that order, then to `fallback` (the artifact's offline tau or the registered
    hand-rule value), each step logged once. ``None`` is returned only when there is
    no fallback at all, in which case `ModelPolicy` applies its own default.
    """
    entry = thresholds.get(variant) if thresholds else None
    if entry:
        if horizon_holdout:
            tau = entry.get("tau_val_excl_heldout")
            if tau is not None:
                return float(tau), "tau_val_excl_heldout"
            _warn_once(name, f"no tau_val_excl_heldout for horizon-holdout {variant}; using tau_val")
        tau = entry.get("tau_val")
        if tau is not None:
            return float(tau), "tau_val"
    _warn_once(name, f"no validation tau for {variant}; falling back to {fallback}")
    return fallback, fallback_source


# ---- resolution -----------------------------------------------------------------------


def artifact_path(variant: str, models_dir: str | Path = DEFAULT_MODELS_DIR) -> Path:
    return Path(models_dir) / f"{variant}.joblib"


def resolve_params(
    name: str,
    horizon: int,
    *,
    cap: int | None = None,
    baseline_params: dict | None = None,
    thresholds: dict | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    artifact: dict | None = None,
) -> dict[str, Any]:
    """Concrete constructor parameters for `name` at horizon `horizon`.

    Every ``None`` slot of the table entry is filled from the tuning outputs (or its
    placeholder, with a warning). For a learned policy the artifact's own ``tau`` is
    the fallback, so the artifact is loaded here unless `artifact` is passed; the
    result carries ``artifact`` as the resolved *path* (a plain string, so the dict
    is JSON-serialisable and can go into the manifest as provenance) and, for any
    thresholded policy, ``tau_source`` (one of `TAU_SOURCES`) saying where its tau
    came from. A schedule constant that fell back to its placeholder additionally
    carries ``params_tuned: False`` and ``params_fallback`` (`PARAMS_TUNED`).

    `cap` is the live-arm cap the item will deploy at. Every constant in
    `baseline_params.json` was selected under one cap (``meta.cap``), so deploying it at
    another is not a tuned comparison: such an item carries ``params_cap`` (the cap it
    WAS tuned at) and ``params_tuned: False`` (`PARAMS_CAP`). ``None`` means the caller
    is not deploying at a particular cap and nothing is claimed either way.
    """
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; known={sorted(POLICIES)}")
    entry = POLICIES[name]
    kind = entry["kind"]
    params = dict(entry["params"])
    horizon = int(horizon)
    baseline_params, tuned_cap = baseline_params_for_cap(baseline_params, cap)
    from_baseline = False

    if kind == "power":
        if params["alpha"] is None:  # p3_star
            (alpha, c), fallback = _tuned_p3_star(name, horizon, baseline_params)
            params["alpha"], params["c"] = alpha, c
            _stamp_untuned(params, fallback)
            from_baseline = True
        elif params["c"] is None:
            params["c"], fallback = _tuned_c(name, float(params["alpha"]), horizon, baseline_params)
            _stamp_untuned(params, fallback)
            from_baseline = True
    elif kind == "refine_after_init":
        if params["K0"] is None:
            params["K0"], fallback = _tuned_k0(name, horizon, baseline_params)
            _stamp_untuned(params, fallback)
            from_baseline = True
    elif kind == "fixed_K":
        if params["K"] is None:  # fixed_K_star
            params["K"], fallback = _tuned_fixed_k(name, horizon, baseline_params)
            _stamp_untuned(params, fallback)
            from_baseline = True
    if from_baseline:
        _stamp_cap(name, params, cap, tuned_cap)
    elif kind == "reservoir_rule":
        if params["tau"] is None:
            params["tau"], params["tau_source"] = _tuned_tau(
                name, "reservoir_rule", thresholds, _PLACEHOLDER_RULE_TAU, "registered"
            )
        else:
            params["tau_source"] = "fixed"
    elif kind == "model":
        variant = params["artifact"]
        path = artifact_path(variant, models_dir)
        if params["tau"] is None:
            if artifact is None:
                artifact = load_model(path)
            params["tau"], params["tau_source"] = _tuned_tau(
                name,
                variant,
                thresholds,
                artifact.get("tau"),
                "artifact_tau",
                horizon_holdout=bool(heldout_horizons(artifact)),
            )
        else:
            params["tau_source"] = "fixed"
        params["artifact"] = str(path)
    return params


def policy_seed(name: str) -> int:
    """The plan's per-policy seed component: ``crc32`` of the *study* name."""
    return int(zlib.crc32(name.encode("utf-8")))


def build_policy(
    name: str,
    params: dict[str, Any],
    *,
    horizon: int,
    n_replicates: int,
    table,
    pairwise=None,
    artifact: dict | None = None,
) -> SearchPolicy:
    """Construct the policy from resolved `params` (see `resolve_params`).

    Always a fresh object: the harness isolates and resets a policy per cell, but a
    fresh instance per (cell, policy) costs nothing and leaves no way for state to
    leak between work items. `artifact` may supply the already-loaded model dict so a
    worker can cache the joblib load across items; the instance's ``name`` is set to
    the study name so snapshots and the harness's default seed carry it.
    """
    entry = POLICIES[name]
    kind = entry["kind"]
    build_params = dict(params)
    for key in PROVENANCE_KEYS:  # provenance for the manifest, not constructor args
        build_params.pop(key, None)
    if kind == "model" and artifact is not None:
        build_params["artifact"] = artifact
    policy = rules.make_policy(
        _SPEC_FOR_KIND[kind],
        horizon=int(horizon),
        n_replicates=int(n_replicates),
        table=table,
        params=build_params,
        pairwise=pairwise,
    )
    policy.name = name
    return policy


def resolve_policy(
    name: str,
    horizon: int,
    *,
    n_replicates: int,
    table,
    baseline_params: dict | None = None,
    thresholds: dict | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    pairwise=None,
) -> tuple[SearchPolicy, int]:
    """`resolve_params` + `build_policy`; returns ``(policy, policy_seed)``."""
    artifact = None
    if POLICIES[name]["kind"] == "model":
        artifact = load_model(artifact_path(POLICIES[name]["params"]["artifact"], models_dir))
    params = resolve_params(
        name,
        horizon,
        baseline_params=baseline_params,
        thresholds=thresholds,
        models_dir=models_dir,
        artifact=artifact,
    )
    policy = build_policy(
        name,
        params,
        horizon=horizon,
        n_replicates=n_replicates,
        table=table,
        pairwise=pairwise,
        artifact=artifact,
    )
    return policy, policy_seed(name)


# ---- queries the runner needs up front -------------------------------------------------


def variant_of(name: str) -> str | None:
    """The M4 variant a learned policy deploys, or ``None`` for non-model kinds."""
    entry = POLICIES[name]
    return entry["params"]["artifact"] if entry["kind"] == "model" else None


def needs_pairwise_table(name: str, models_dir: str | Path = DEFAULT_MODELS_DIR) -> bool:
    """Whether the policy's model reads ``f_log_e_pair`` (explicit column membership).

    Read from the artifact itself rather than from `variants.json`, so the answer is
    true of the model that will actually be deployed.
    """
    variant = variant_of(name)
    if variant is None:
        return False
    features = load_model(artifact_path(variant, models_dir))["features"]
    return any(c in EVIDENCE_LOGE for c in features)


def check_policies(names: list[str] | tuple[str, ...]) -> None:
    """Raise on an unknown policy name, listing the known ones."""
    unknown = [n for n in names if n not in POLICIES]
    if unknown:
        raise KeyError(f"unknown policies {unknown}; known={sorted(POLICIES)}")
