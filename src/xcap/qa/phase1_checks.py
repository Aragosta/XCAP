"""Phase 1 quality gate: coverage, price sanity, and corporate-action integrity.

The reconciliation between locally rebuilt adjusted prices and the vendor's
adjusted_close is the centrepiece. Where the two disagree, one of them is
wrong, and the observed causes are worth naming because they are different
problems with different fixes:

1. Ticker recycling / spliced series. Splits or dividends dated outside the
   security's own EOD range mean the price series and the action series
   describe different companies that shared a symbol. Both our factors and the
   vendor's are then meaningless for that security.
2. Vendor missed a split. adjusted_close equals raw close across a series that
   demonstrably has a split. Here the local rebuild is correct and the vendor
   is not -- which is the whole argument for storing raw prices.
3. Incomplete distributions. ETFs in particular pay distributions that the
   vendor folds into adjusted_close but does not expose through /div, so the
   local factor legitimately cannot match.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from ..config import DATA_DIR, PARQUET_DIR
from ..ledger import Ledger
from ..universe import START_DATE, select

# Below this, the local rebuild disagrees with the vendor too often to treat
# either series as trustworthy without triage.
MIN_PCT_WITHIN_TOLERANCE = 95.0


@dataclass
class Check:
    name: str
    level: str
    detail: str


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DATA_DIR / '_duckdb_tmp'}'")
    con.execute(f"CREATE VIEW eod AS SELECT * FROM read_parquet('{PARQUET_DIR}/eod/**/*.parquet')")
    con.execute(f"CREATE VIEW adj AS SELECT * FROM read_parquet('{PARQUET_DIR}/adjustments/**/*.parquet')")
    con.execute(f"CREATE VIEW splits AS SELECT * FROM read_parquet('{PARQUET_DIR}/splits.parquet')")
    con.execute(f"CREATE VIEW divs AS SELECT * FROM read_parquet('{PARQUET_DIR}/dividends.parquet')")
    con.execute(f"CREATE VIEW secs AS SELECT * FROM read_parquet('{PARQUET_DIR}/securities.parquet')")
    return con


def run_checks(ledger: Ledger) -> list[Check]:
    if not (PARQUET_DIR / "eod").exists():
        return [Check("artifacts", "FAIL", "eod/ missing; run `phase1-build`")]

    con = _connect()
    checks: list[Check] = []
    universe = {s.security_id for s in select()}

    # ---- coverage ----------------------------------------------------
    rows, secs, dmin, dmax = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT security_id), MIN(date), MAX(date) FROM eod"
    ).fetchone()
    checks.append(Check("eod volume", "PASS",
                        f"{rows:,} bars, {secs:,} securities, {dmin} to {dmax}"))

    checks.append(Check("start-date floor",
                        "PASS" if str(dmin) >= START_DATE else "FAIL",
                        f"earliest bar {dmin} (floor {START_DATE})"))

    fetched = {r["key"] for r in ledger.rows("eod", status="ok")}
    attempted = fetched | {r["key"] for r in ledger.rows("eod", status="empty")}
    tickers = {s.api_ticker for s in select()}
    unattempted = tickers - attempted
    checks.append(Check("universe coverage",
                        "PASS" if not unattempted else "WARN",
                        f"{len(attempted):,}/{len(tickers):,} universe securities fetched; "
                        f"{len(unattempted):,} not yet attempted"))

    failures = [r for r in ledger.failures() if r["endpoint"] in ("eod", "splits", "dividends")]
    checks.append(Check("fetch failures", "PASS" if not failures else "FAIL",
                        f"{len(failures)} unrecovered" if failures else "none"))

    # Securities present in the universe but with no usable bars, and why.
    with_data, = con.execute("SELECT COUNT(DISTINCT security_id) FROM eod").fetchone()
    checks.append(Check("securities with bars", "PASS",
                        f"{with_data:,} of {len(universe):,} universe securities have "
                        f"bars on or after {START_DATE}"))

    delisted_share, = con.execute(
        "SELECT COUNT(DISTINCT e.security_id) FILTER (WHERE s.is_delisted) * 1.0 "
        "/ COUNT(DISTINCT e.security_id) FROM eod e JOIN secs s USING (security_id)"
    ).fetchone()
    checks.append(Check("survivorship (with data)",
                        "PASS" if delisted_share >= 0.30 else "FAIL",
                        f"{delisted_share:.1%} of securities carrying bars are delisted"))

    # ---- price integrity ---------------------------------------------
    dupes, = con.execute(
        "SELECT COUNT(*) FROM (SELECT security_id, date FROM eod "
        "GROUP BY 1,2 HAVING COUNT(*) > 1)"
    ).fetchone()
    checks.append(Check("duplicate bars", "PASS" if dupes == 0 else "FAIL",
                        f"{dupes:,} duplicated (security_id, date) pairs"))

    nonpos, = con.execute(
        "SELECT COUNT(*) FROM eod WHERE close <= 0 OR open <= 0 OR high <= 0 OR low <= 0"
    ).fetchone()
    checks.append(Check("non-positive prices", "PASS" if nonpos == 0 else "WARN",
                        f"{nonpos:,} bars ({nonpos/max(rows,1):.4%})"))

    ohlc, = con.execute(
        "SELECT COUNT(*) FROM eod WHERE high < low OR close > high OR close < low "
        "OR open > high OR open < low"
    ).fetchone()
    checks.append(Check("OHLC consistency", "PASS" if ohlc == 0 else "WARN",
                        f"{ohlc:,} bars violate low <= {{open,close}} <= high "
                        f"({ohlc/max(rows,1):.4%})"))

    # Returns above +100% that no split on that date explains. A handful is
    # normal for microcaps; a large count means missing split data.
    jumps, = con.execute("""
        WITH r AS (
            SELECT e.security_id, e.date, e.close,
                   LAG(e.close) OVER (PARTITION BY e.security_id ORDER BY e.date) AS prev
            FROM eod e
        )
        SELECT COUNT(*) FROM r
        LEFT JOIN splits s USING (security_id, date)
        WHERE prev > 0 AND close / prev > 2.0 AND s.security_id IS NULL
    """).fetchone()
    checks.append(Check("unexplained price jumps", "PASS" if jumps / max(rows, 1) < 0.001 else "WARN",
                        f"{jumps:,} bars >100% up with no split on that date "
                        f"({jumps/max(rows,1):.4%})"))

    # ---- corporate-action integrity ----------------------------------
    # Splits/dividends dated outside a security's own EOD range: the price
    # series and the action series describe different entities sharing a
    # ticker. This is the ticker-recycling detector deferred from Phase 0.
    recycled, total_with_events = con.execute("""
        WITH span AS (SELECT security_id, MIN(date) lo, MAX(date) hi FROM eod GROUP BY 1),
        ev AS (
            SELECT security_id, date FROM splits
            UNION ALL SELECT security_id, date FROM divs
        ),
        flagged AS (
            SELECT e.security_id,
                   COUNT(*) FILTER (WHERE e.date > s.hi OR e.date < s.lo) AS outside
            FROM ev e JOIN span s USING (security_id) GROUP BY 1
        )
        SELECT COUNT(*) FILTER (WHERE outside > 0), COUNT(*) FROM flagged
    """).fetchone()
    share = recycled / max(total_with_events, 1)
    checks.append(Check("spliced / recycled tickers",
                        "PASS" if share < 0.02 else "WARN",
                        f"{recycled:,}/{total_with_events:,} securities ({share:.1%}) have "
                        "corporate actions dated outside their own price history"))

    neg_factor, = con.execute(
        "SELECT COUNT(*) FROM adj WHERE price_factor <= 0 OR split_factor <= 0 "
        "OR price_factor IS NULL OR split_factor IS NULL"
    ).fetchone()
    checks.append(Check("adjustment factors valid", "PASS" if neg_factor == 0 else "FAIL",
                        f"{neg_factor:,} non-positive or null factors"))

    last_bar_unit, = con.execute("""
        WITH last AS (
            SELECT security_id, MAX(date) d FROM eod GROUP BY 1
        )
        SELECT COUNT(*) FROM last l JOIN adj a ON a.security_id = l.security_id AND a.date = l.d
        WHERE abs(a.price_factor - 1.0) > 1e-9
    """).fetchone()
    checks.append(Check("factor anchored at 1.0", "PASS" if last_bar_unit == 0 else "FAIL",
                        f"{last_bar_unit:,} securities whose latest bar has factor != 1.0"))

    # ---- reconciliation ----------------------------------------------
    within, outside = con.execute("""
        SELECT COUNT(*) FILTER (WHERE abs(e.close * a.price_factor - e.vendor_adjusted_close)
                                     / e.vendor_adjusted_close <= 0.01),
               COUNT(*) FILTER (WHERE abs(e.close * a.price_factor - e.vendor_adjusted_close)
                                     / e.vendor_adjusted_close > 0.01)
        FROM eod e JOIN adj a USING (security_id, date)
        WHERE e.vendor_adjusted_close > 0
    """).fetchone()
    pct = 100.0 * within / max(within + outside, 1)
    checks.append(Check("adjustment reconciliation",
                        "PASS" if pct >= MIN_PCT_WITHIN_TOLERANCE else "WARN",
                        f"{pct:.2f}% of bars within 1% of vendor adjusted_close "
                        f"(target >= {MIN_PCT_WITHIN_TOLERANCE}%)"))

    con.close()
    return checks


def format_report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"  [{c.level}] {c.name.ljust(width)}  {c.detail}" for c in checks]
    worst = "FAIL" if any(c.level == "FAIL" for c in checks) else \
            "WARN" if any(c.level == "WARN" for c in checks) else "PASS"
    return "\n".join(lines + ["", f"  gate: {worst}"])
