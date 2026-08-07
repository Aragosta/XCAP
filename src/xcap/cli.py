"""xcap command line.

    python -m xcap.cli phase0-fetch     # pull exchange + ticker lists (active AND delisted)
    python -m xcap.cli phase0-build     # raw archive -> parquet (no network)
    python -m xcap.cli phase0-qa        # bias + integrity gate
    python -m xcap.cli probe-history    # measure real EOD history depth on a sample
    python -m xcap.cli probe-delisting  # test a start year for survivorship-bias safety
    python -m xcap.cli phase3-fetch     # equity fundamentals, 10 calls/symbol
    python -m xcap.cli coverage         # write PROGRESS.md: downloaded vs remaining
    python -m xcap.cli status           # ledger and budget summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

from .backup import backup as do_backup, verify_raw
from .config import ensure_dirs, load_config
from .eodhd.budget import Budget
from .corpactions import build_adjustments, reconcile
from .jobs.phase0_universe import fetch_universe
from .jobs.phase1_prices import fetch_prices
from .jobs.phase2_reference import fetch_reference
from .jobs.phase3_fundamentals import fetch_fundamentals
from .jobs.phase3_fundamentals import report as phase3_report
from .jobs.probe_delisting import probe_delisting, verdict
from .jobs.probe_entitlements import probe_entitlements
from .jobs.probe_history import probe_history
from .ledger import Ledger
from .plan import report as plan_report
from .universe import START_DATE
from .qa.phase0_checks import format_report, run_checks
from .qa.coverage import write_report
from .qa.phase1_checks import format_report as fmt1, run_checks as checks1
from .transform.fundamentals import build_all as build_fundamentals
from .transform.prices import build_all as build_prices
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


@contextmanager
def _fetch_session():
    """Every network command opens the same way and, crucially, closes the same
    way: the spend line is printed on the way out even when the job raises, so a
    run that dies mid-flight still says what it cost."""
    cfg = load_config()
    print(f"token {cfg.redacted_token} | budget {cfg.daily_call_budget:,}/day | "
          f"{cfg.rate_per_min} req/min | concurrency {cfg.concurrency}")
    with Ledger() as ledger:
        try:
            yield cfg, ledger
        finally:
            print(f"\ncalls spent today: {ledger.spent_today(Budget.gmt_day()):,} "
                  f"/ {cfg.daily_call_budget:,}")


def cmd_phase0_fetch(args: argparse.Namespace) -> int:
    with _fetch_session() as (cfg, ledger):
        stats = asyncio.run(fetch_universe(cfg, ledger, force=args.force))
    print(json.dumps(stats, indent=2))
    return 0


def cmd_phase0_build(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        manifest = build_all(ledger)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_phase0_qa(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        checks = run_checks(ledger)
    print("\nPhase 0 quality gate\n")
    print(format_report(checks))
    print()
    return 1 if any(c.level == "FAIL" for c in checks) else 0


def cmd_probe_history(args: argparse.Namespace) -> int:
    with _fetch_session() as (cfg, ledger):
        report = asyncio.run(probe_history(
            cfg, ledger, exchange=args.exchange,
            per_stratum=args.per_stratum, seed=args.seed,
        ))
    print(f"\nEOD history depth — {report['exchange']} "
          f"(sampled {report['sampled']}, with data {report['with_data']}, "
          f"no data {report['no_data']})\n")
    width = max(len(k) for k in report["by_stratum"]) if report["by_stratum"] else 10
    print(f"  {'stratum'.ljust(width)}    n  earliest   p25  median  med.months")
    for name, s in report["by_stratum"].items():
        print(f"  {name.ljust(width)} {s['n']:>4}  {s['earliest']:>8}  "
              f"{s['p25_first_year']:>4}  {s['median_first_year']:>6}  {s['median_months']:>10}")
    print("\n  first observation by decade")
    for decade, n in report["first_year_histogram"].items():
        print(f"    {decade}  {'#' * min(n, 60)} {n}")
    print()
    return 0


def cmd_probe_delisting(args: argparse.Namespace) -> int:
    with _fetch_session() as (cfg, ledger):
        report = asyncio.run(probe_delisting(
            cfg, ledger, n=args.sample, seed=args.seed,
        ))
    print(f"\nDelisting coverage — US {'/'.join(report['types'])}")
    print(f"  population {report['population']:,} | sampled {report['sampled']} | "
          f"with data {report['with_data']} | scale x{report['scale_factor']}")
    print(f"  earliest delisting in archive: {report['earliest_delisting']}\n")
    print("  year  sampled  implied")
    for year, v in report["delisting_year_histogram"].items():
        print(f"  {year}  {v['sampled']:>7}  {v['implied_population']:>7}  "
              f"{'#' * min(v['sampled'], 60)}")
    print()
    for start in args.start_years:
        level, msg = verdict(report, start)
        print(f"  start {start}: {level} — {msg}")
    print()
    return 1 if any(verdict(report, s)[0] == "FAIL" for s in args.start_years) else 0


def cmd_phase1_fetch(args: argparse.Namespace) -> int:
    with _fetch_session() as (cfg, ledger):
        stats = asyncio.run(fetch_prices(
            cfg, ledger, which=args.endpoints, limit=args.limit,
            seed_sample=args.sample_seed,
        ))
    print(json.dumps(stats, indent=2))
    if stats["budget_exhausted"]:
        print("\nDaily budget exhausted. Re-run after midnight GMT; the ledger "
              "resumes without re-spending on what is already fetched.")
    return 0


def cmd_phase1_build(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        manifest = build_prices(ledger, start_date=args.start_date)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_phase1_adjust(args: argparse.Namespace) -> int:
    print(json.dumps(build_adjustments(), indent=2))
    return 0


def cmd_phase1_reconcile(args: argparse.Namespace) -> int:
    report = reconcile(tolerance=args.tolerance, sample_securities=args.sample)
    print(f"\nAdjustment reconciliation (tolerance {report['tolerance']:.1%})\n")
    print(f"  bars compared          {report['bars_compared']:,}")
    print(f"  within tolerance       {report['within_tolerance']:,} "
          f"({report['pct_within']:.3f}%)")
    print(f"  outside tolerance      {report['outside_tolerance']:,}")
    print(f"  median relative error  {report['median_rel_err']:.2e}")
    print(f"  p99 relative error     {report['p99_rel_err']:.2e}")
    print(f"  securities             {report['securities']:,} "
          f"({report['securities_with_mismatch']:,} with mismatches)")
    if report["worst_securities"]:
        print("\n  worst securities (security_id, bad bars / bars, max err)")
        for w in report["worst_securities"][:10]:
            print(f"    {w['security_id']:>7}  {w['bad_bars']:>6}/{w['bars']:<6} "
                  f"{w['max_rel_err']:.3f}")
    print()
    return 0


def cmd_phase1_qa(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        checks = checks1(ledger)
    print("\nPhase 1 quality gate\n")
    print(fmt1(checks))
    print()
    return 1 if any(c.level == "FAIL" for c in checks) else 0


def cmd_phase2_fetch(args: argparse.Namespace) -> int:
    with _fetch_session() as (cfg, ledger):
        stats = asyncio.run(fetch_reference(cfg, ledger, which=args.which))
    print(json.dumps(stats, indent=2))
    return 0


def cmd_phase3_fetch(args: argparse.Namespace) -> int:
    if args.blocks_only:
        with Ledger() as ledger:
            print(phase3_report(ledger, args.only))
        return 0

    with _fetch_session() as (cfg, ledger):
        stats = asyncio.run(fetch_fundamentals(
            cfg, ledger, only=args.only, max_calls=args.max_calls,
            sample=args.sample, seed=args.seed,
        ))
    print(json.dumps(stats, indent=2))
    if stats["stopped"]:
        print(f"\nStopped: {stats['stopped']}\nRe-run to resume; blocks already "
              "resolved cost nothing.")
    return 0


def cmd_phase3_build(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        manifest = build_fundamentals(ledger)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        _, md = write_report(ledger)
    print(md)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        print(plan_report(ledger, split_by_venue=args.split_by_venue))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        res = verify_raw(ledger, full=not args.fast)
    mode = "existence only" if args.fast else "full sha256"
    print(f"\nRaw archive verification ({mode})\n")
    print(f"  checked   {res.checked:,}")
    print(f"  ok        {res.ok:,}")
    print(f"  missing   {len(res.missing):,}")
    print(f"  corrupt   {len(res.corrupt):,}")
    for label, items in (("missing", res.missing), ("corrupt", res.corrupt)):
        for x in items[:10]:
            print(f"    {label}: {x}")
    print(f"\n  {'CLEAN' if res.clean else 'FAILED'}\n")
    return 0 if res.clean else 1


def cmd_backup(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
        res = do_backup(Path(args.dest), ledger, verify=not args.no_verify)
    print(json.dumps(res, indent=2))
    v = res.get("verify")
    if v and not v["clean"]:
        print("\nBACKUP VERIFICATION FAILED - do not rely on this copy.")
        return 1
    return 0


def cmd_probe_entitlements(args: argparse.Namespace) -> int:
    with _fetch_session() as (cfg, ledger):
        report = asyncio.run(probe_entitlements(cfg, ledger))
    print(f"\nEndpoint entitlements ({report['accessible']}/{report['total']} accessible)\n")
    width = max(len(r["endpoint"]) for r in report["probes"])
    for r in report["probes"]:
        mark = "ok " if r["accessible"] else "NO "
        print(f"  [{mark}] {r['endpoint'].ljust(width)}  {r['status']:<12} {r['shape'][:70]}")
    print()
    for r in report["probes"]:
        if not r["accessible"]:
            print(f"  inaccessible: {r['endpoint']} -- {r['why']}")
    print()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with Ledger() as ledger:
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

    p = sub.add_parser("probe-history", help="measure real EOD history depth on a sample")
    p.add_argument("--exchange", default="US")
    p.add_argument("--per-stratum", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_probe_history)

    p = sub.add_parser("probe-delisting",
                       help="test whether the delisted archive supports a given start year")
    p.add_argument("--sample", type=int, default=1000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--start-years", type=int, nargs="+", default=[1995, 2000, 2005])
    p.set_defaults(func=cmd_probe_delisting)

    p = sub.add_parser("phase1-fetch", help="fetch EOD, splits and dividends")
    p.add_argument("--endpoints", nargs="+", default=["eod", "splits", "dividends"],
                   choices=["eod", "splits", "dividends"])
    p.add_argument("--limit", type=int, default=None, help="cap the universe size")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="take a deterministic random subsample instead of the first N")
    p.set_defaults(func=cmd_phase1_fetch)

    p = sub.add_parser("phase1-build", help="build price parquet from the raw archive")
    p.add_argument("--start-date", default=START_DATE)
    p.set_defaults(func=cmd_phase1_build)

    p = sub.add_parser("phase1-adjust", help="compute corporate-action adjustment factors")
    p.set_defaults(func=cmd_phase1_adjust)

    p = sub.add_parser("phase1-reconcile",
                       help="verify rebuilt adjusted prices against the vendor's")
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--sample", type=int, default=None)
    p.set_defaults(func=cmd_phase1_reconcile)

    p = sub.add_parser("phase1-qa", help="run the Phase 1 quality gate")
    p.set_defaults(func=cmd_phase1_qa)

    p = sub.add_parser("phase2-fetch", help="fetch reference and non-equity datasets")
    p.add_argument("--which", nargs="+",
                   default=["indices", "exchange-details", "earnings", "macro", "noneq-eod"],
                   choices=["indices", "exchange-details", "earnings", "macro", "noneq-eod"])
    p.set_defaults(func=cmd_phase2_fetch)

    p = sub.add_parser("phase3-fetch", help="fetch equity fundamentals (10 calls/symbol)")
    p.add_argument("--only", nargs="+", default=None,
                   help="restrict to blocks matching a venue or type "
                        "(e.g. --only NASDAQ NYSE, or --only 'Common Stock')")
    p.add_argument("--max-calls", type=int, default=None,
                   help="cap what this run spends, leaving headroom under the daily cap")
    p.add_argument("--sample", type=int, default=None,
                   help="fetch a stratified sample across type x listing status instead "
                        "of the universe, to measure vendor coverage before committing")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--blocks-only", action="store_true",
                   help="print per-block download state and exit, spending nothing")
    p.set_defaults(func=cmd_phase3_fetch)

    p = sub.add_parser("phase3-build", help="build fundamentals parquet from the raw archive")
    p.set_defaults(func=cmd_phase3_build)

    p = sub.add_parser("coverage",
                       help="write data/catalog/PROGRESS.md: what is downloaded and what remains")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("plan", help="cost the remaining work against today's budget")
    p.add_argument("--split-by-venue", action="store_true",
                   help="break equity blocks into whole per-venue blocks")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("probe-entitlements",
                       help="test which endpoints this token can actually reach")
    p.set_defaults(func=cmd_probe_entitlements)

    p = sub.add_parser("verify", help="check the raw archive against ledger checksums")
    p.add_argument("--fast", action="store_true",
                   help="check existence only, skip rehashing")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("backup", help="mirror the raw archive + catalog, then verify the copy")
    p.add_argument("dest", help="destination directory")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(func=cmd_backup)

    p = sub.add_parser("status", help="ledger and budget summary")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    ensure_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
