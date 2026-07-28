"""xcap command line.

    python -m xcap.cli phase0-fetch     # pull exchange + ticker lists (active AND delisted)
    python -m xcap.cli phase0-build     # raw archive -> parquet (no network)
    python -m xcap.cli phase0-qa        # bias + integrity gate
    python -m xcap.cli status           # ledger and budget summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import ensure_dirs, load_config
from .eodhd.budget import Budget
from .jobs.phase0_universe import fetch_universe
from .ledger import Ledger
from .qa.phase0_checks import format_report, run_checks
from .transform.universe import build_all


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx logs full request URLs at INFO, which would write the API token into
    # every log file and terminal scrollback.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def cmd_phase0_fetch(args: argparse.Namespace) -> int:
    cfg = load_config()
    ledger = Ledger()
    print(f"token {cfg.redacted_token} | budget {cfg.daily_call_budget:,}/day | "
          f"{cfg.rate_per_min} req/min | concurrency {cfg.concurrency}")
    try:
        stats = asyncio.run(fetch_universe(cfg, ledger, force=args.force))
    finally:
        spent = ledger.spent_today(Budget.gmt_day())
        print(f"\ncalls spent today: {spent:,} / {cfg.daily_call_budget:,}")
        ledger.close()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_phase0_build(args: argparse.Namespace) -> int:
    ledger = Ledger()
    try:
        manifest = build_all(ledger)
    finally:
        ledger.close()
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_phase0_qa(args: argparse.Namespace) -> int:
    ledger = Ledger()
    try:
        checks = run_checks(ledger)
    finally:
        ledger.close()
    print("\nPhase 0 quality gate\n")
    print(format_report(checks))
    print()
    return 1 if any(c.level == "FAIL" for c in checks) else 0


def cmd_status(args: argparse.Namespace) -> int:
    ledger = Ledger()
    try:
        print("\nrequests by endpoint/status")
        for r in ledger.summary():
            mb = (r["bytes"] or 0) / 1e6
            print(f"  {r['endpoint']:<24} {r['status']:<16} "
                  f"n={r['n']:>7,}  calls={r['calls'] or 0:>8,}  {mb:>9.1f} MB")
        print("\ncall budget by GMT day")
        for r in ledger.budget_history():
            print(f"  {r['gmt_day']}  {r['calls_spent']:,}")
        failures = ledger.failures()
        if failures:
            print(f"\n{len(failures)} unrecovered requests")
            for r in failures[:20]:
                print(f"  {r['endpoint']}/{r['key']}  {r['status']} "
                      f"http={r['http_status']}  {(r['error'] or '')[:80]}")
        print()
    finally:
        ledger.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xcap", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("phase0-fetch", help="fetch exchange and ticker lists")
    p.add_argument("--force", action="store_true",
                   help="re-fetch even if the ledger already has a successful record")
    p.set_defaults(func=cmd_phase0_fetch)

    p = sub.add_parser("phase0-build", help="build parquet from the raw archive")
    p.set_defaults(func=cmd_phase0_build)

    p = sub.add_parser("phase0-qa", help="run the Phase 0 quality gate")
    p.set_defaults(func=cmd_phase0_qa)

    p = sub.add_parser("status", help="ledger and budget summary")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    ensure_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
