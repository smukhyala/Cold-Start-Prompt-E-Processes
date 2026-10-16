"""One explicitly authorized continuation without altering frozen run artifacts.

The original request ledger remains immutable. Its known charges and unresolved
maximum charge are deducted from the original USD limit before constructing the
unchanged, fail-closed transport for this new execution segment.

On macOS, wrap execution in ``caffeinate -i`` to prevent idle system sleep for
the process lifetime while allowing the display to sleep. This does not promise
protection against lid closure or loss of power.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import live_stopping_runner as frozen  # noqa: E402


def opening_balance(manifest: dict, original_config: dict) -> dict:
    """Reconcile all original charges, including the invalid half-pair.

    Acknowledging the original ambiguity authorizes a new physical attempt;
    it does not erase or settle the original request's reserved charge.
    """
    source = Path(manifest["source_directory"]).resolve()
    ledger = source / "api_calls.jsonl"
    expected = []
    for name, digest in manifest["source_files"].items():
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = source / candidate
        if candidate.resolve() == ledger:
            expected.append(digest)
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    if expected != [digest]:
        raise ValueError("original API ledger is not uniquely pinned by the amendment")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    pending, seen = {}, set()
    known = Decimal("0")
    for row in rows:
        request_id = row["request_id"]
        event = row["event"]
        if event == "request_started":
            if request_id in seen:
                raise ValueError("duplicate original physical request id")
            seen.add(request_id)
            pending[request_id] = row
        elif event in ("request_completed", "request_rejected"):
            if request_id not in pending:
                raise ValueError("original usage receipt has no pending request")
            amount = Decimal(str(row["cost_usd"]))
            if not amount.is_finite() or amount < 0:
                raise ValueError("invalid original request cost")
            if event == "request_completed":
                computed, *_ = frozen.usage_cost(row["raw_usage"])
                if computed != amount:
                    raise ValueError("original usage and cost disagree")
            known += amount
            del pending[request_id]
        elif event not in ("request_unknown", "invocation_failed"):
            raise ValueError("unknown original ledger event")
    acknowledged = manifest["acknowledged_unknown_request_ids"]
    if sorted(pending) != sorted(acknowledged) or len(set(acknowledged)) != len(acknowledged):
        raise ValueError("amendment must explicitly identify every unresolved original request")
    reserved = sum((Decimal(str(row["reserved_usd"])) for row in pending.values()), Decimal("0"))
    if not reserved.is_finite() or reserved < 0:
        raise ValueError("invalid original reservation")
    maximum = Decimal(str(original_config["max_usd"]))
    remaining = maximum - known - reserved
    if not Decimal("0") < remaining <= maximum <= Decimal("100"):
        raise ValueError("no admissible continuation allocation remains")
    return {
        "schema": "live_stopping/authorized_continuation_balance/1",
        "source_directory": str(source), "source_api_ledger_sha256": digest,
        "global_maximum_usd": float(maximum), "original_known_cost_usd": float(known),
        "original_unknown_reserved_usd": float(reserved),
        "original_unknown_request_ids": sorted(pending),
        "original_started_requests": len(seen),
        "new_segment_maximum_usd": float(remaining),
        "logical_episode_id": manifest["logical_episode_id"],
        "physical_episode_id": manifest["physical_episode_id"],
        "accounting_rule": "global charge upper bound = old known + old reserved + new known + new reserved",
        "request_id_namespace": "request IDs are local to the source/continuation directory segment",
    }


class AuthorizedContinuationEvaluator(frozen.LiveEvaluator):
    """Use the frozen evaluator with a reduced, independently enforced allocation.

    The continuation controller supplies physical episode IDs and enforces the
    single authorized retry. This additional guard rejects alternate retry IDs.
    """

    def __init__(self, config: dict, directory: Path, manifest: dict):
        self.amendment = copy.deepcopy(manifest)
        self.original_config = copy.deepcopy(config)
        self.balance = opening_balance(manifest, config)
        if Path(directory).resolve() != Path(manifest["continuation_directory"]).resolve():
            raise ValueError("continuation output directory differs from the amendment")
        if Path(directory).resolve() == Path(manifest["source_directory"]).resolve():
            raise ValueError("continuation must not overwrite the original run")
        if manifest["max_retry_attempts"] != 1:
            raise ValueError("only one explicitly authorized retry is supported")
        internal = copy.deepcopy(config)
        # The original $75 cap remains authoritative. This suballocation
        # reserves all prior known and uncertain charges exactly once.
        internal["max_usd"] = self.balance["new_segment_maximum_usd"]
        super().__init__(internal, directory)
        balance_path = Path(directory) / "opening_balance.json"
        if balance_path.exists():
            if json.loads(balance_path.read_text()) != self.balance:
                raise ValueError("existing continuation opening balance changed")
        else:
            frozen._atomic_json(balance_path, self.balance)

    def __call__(self, task: dict, role: str, episode_id: str) -> dict:
        logical = self.amendment["logical_episode_id"]
        physical = self.amendment["physical_episode_id"]
        if episode_id == logical or ("_retry" in episode_id and episode_id != physical):
            raise ValueError("this physical retry was not authorized")
        if episode_id == physical and (role != "challenger" or task["task_id"] != "live_014"):
            raise ValueError("authorized retry must be challenger on the original fourteenth draw")
        receipt = super().__call__(task, role, episode_id)
        return {**receipt, "execution_segment": str(self.directory.resolve()),
                "original_unknown_charge_still_reserved_usd": self.balance["original_unknown_reserved_usd"]}


def main(argv: list[str] | None = None) -> int:
    # Import only when the CLI is used, permitting independent no-network
    # tests while the controller extension is being audited.
    from experiments.growing_bandits.empirical import live_stopping_resume

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute", action="store_true", help="execute the authorized continuation")
    action.add_argument("--preflight", action="store_true", help="check app/browser startup without inference")
    args = parser.parse_args(argv)
    config, _tasks, directory = live_stopping_resume.initialize_continuation(
        args.manifest, args.manifest_sha256)
    manifest = json.loads(args.manifest.read_text())
    evaluator = AuthorizedContinuationEvaluator(config, directory, manifest)
    try:
        if args.preflight:
            print(json.dumps(evaluator.prepare(), indent=2))
        else:
            state = live_stopping_resume.run_continuation(
                args.manifest, args.manifest_sha256, evaluator)
            print(json.dumps(state, indent=2))
    finally:
        evaluator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
