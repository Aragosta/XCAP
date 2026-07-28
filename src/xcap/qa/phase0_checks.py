"""Phase 0 quality gate.

The point of these checks is not tidiness — it is to catch bias while the
subscription is still active and the data can still be re-pulled. The
survivorship check is the one that matters most: if the delisted share of a
long-history equity universe looks small, the bias is still in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from ..config import PARQUET_DIR
from ..ledger import Ledger

# For US equities over a multi-decade window, securities that have ceased
# trading typically outnumber survivors. A low share means the delisted pass
# did not work, not that the market is unusually stable.
US_DELISTED_MIN_SHARE = 0.30


@dataclass
class Check:
    name: str
    level: str          # PASS | WARN | FAIL
    detail: str


def run_checks(ledger: Ledger) -> list[Check]:
    sec = PARQUET_DIR / "securities.parquet"
    exch = PARQUET_DIR / "exchanges.parquet"
    if not sec.exists():
        return [Check("artifacts", "FAIL", "securities.parquet missing; run `phase0 build`")]

    con = duckdb.connect()
    con.execute(f"CREATE VIEW securities AS SELECT * FROM read_parquet('{sec}')")
    con.execute(f"CREATE VIEW exchanges AS SELECT * FROM read_parquet('{exch}')")
    checks: list[Check] = []

    total, n_exch = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT source_exchange) FROM securities"
    ).fetchone()
    checks.append(Check("universe size", "PASS" if total > 0 else "FAIL",
                        f"{total:,} securities across {n_exch} exchange codes"))

    # --- survivorship -------------------------------------------------
    delisted, = con.execute("SELECT COUNT(*) FROM securities WHERE is_delisted").fetchone()
    share = delisted / total if total else 0.0
    checks.append(Check("delisted coverage (global)",
                        "PASS" if delisted > 0 else "FAIL",
                        f"{delisted:,} delisted ({share:.1%} of universe)"))

    us_total, us_delisted = con.execute(
        "SELECT COUNT(*), COUNT(*) FILTER (WHERE is_delisted) "
        "FROM securities WHERE source_exchange='US'"
    ).fetchone()
    if us_total:
        us_share = us_delisted / us_total
        level = "PASS" if us_share >= US_DELISTED_MIN_SHARE else "FAIL"
        checks.append(Check(
            "survivorship bias (US)", level,
            f"{us_delisted:,}/{us_total:,} US securities delisted ({us_share:.1%}); "
            f"expected >= {US_DELISTED_MIN_SHARE:.0%}",
        ))
    else:
        checks.append(Check("survivorship bias (US)", "FAIL", "no US securities found"))

    overlap, = con.execute(
        "SELECT COUNT(*) FROM securities WHERE listed_active AND listed_delisted"
    ).fetchone()
    checks.append(Check("active/delisted overlap", "PASS" if overlap == 0 else "WARN",
                        f"{overlap:,} securities in both lists (re-listings or vendor noise)"))

    # --- identity integrity -------------------------------------------
    dupes, = con.execute(
        "SELECT COUNT(*) FROM (SELECT api_ticker FROM securities "
        "GROUP BY api_ticker HAVING COUNT(*) > 1)"
    ).fetchone()
    checks.append(Check("api_ticker uniqueness", "PASS" if dupes == 0 else "FAIL",
                        f"{dupes:,} duplicated tickers"))

    id_ok, = con.execute(
        "SELECT COUNT(*) = COUNT(DISTINCT security_id) FROM securities"
    ).fetchone()
    checks.append(Check("security_id uniqueness", "PASS" if id_ok else "FAIL",
                        "unique" if id_ok else "collisions present"))

    # Ticker recycling cannot be measured here: the vendor exposes one row per
    # (exchange, code), so a symbol reused by a second company after the first
    # delisted collapses into a single row. Detecting it needs trade-date gaps
    # from Phase 1 EOD history. Flagged so it is not mistaken for "checked".
    checks.append(Check("ticker recycling", "WARN",
                        "not measurable from listing data — deferred to Phase 1 "
                        "(detect via gaps in EOD trade dates per code)"))

    # --- metadata completeness ----------------------------------------
    isin_null, = con.execute(
        "SELECT COUNT(*) FROM securities WHERE isin IS NULL OR isin=''"
    ).fetchone()
    checks.append(Check("ISIN completeness", "WARN" if isin_null / max(total, 1) > 0.5 else "PASS",
                        f"{isin_null:,} ({isin_null/max(total,1):.1%}) missing ISIN — "
                        "Phase 0b ID-mapping should fill these"))

    types = con.execute(
        "SELECT type, COUNT(*) n FROM securities GROUP BY 1 ORDER BY n DESC LIMIT 8"
    ).fetchall()
    checks.append(Check("instrument types", "PASS",
                        ", ".join(f"{t or 'NULL'}={n:,}" for t, n in types)))

    # --- fetch completeness -------------------------------------------
    empty_lists = [r["key"] for r in ledger.rows("exchange-symbol-list", status="empty")]
    checks.append(Check("empty symbol lists", "PASS" if not empty_lists else "WARN",
                        f"{len(empty_lists)} exchange lists returned no rows: "
                        f"{', '.join(empty_lists[:10])}" if empty_lists else "none"))

    failures = ledger.failures()
    failed_names = ", ".join(f"{r['endpoint']}/{r['key']}" for r in failures[:10])
    checks.append(Check("fetch failures", "PASS" if not failures else "FAIL",
                        f"{len(failures)} unrecovered requests: {failed_names}"
                        if failures else "none"))

    con.close()
    return checks


def format_report(checks: list[Check]) -> str:
    icon = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}
    width = max(len(c.name) for c in checks)
    lines = [f"  [{icon[c.level]}] {c.name.ljust(width)}  {c.detail}" for c in checks]
    worst = "FAIL" if any(c.level == "FAIL" for c in checks) else \
            "WARN" if any(c.level == "WARN" for c in checks) else "PASS"
    lines.append("")
    lines.append(f"  gate: {worst}")
    return "\n".join(lines)
