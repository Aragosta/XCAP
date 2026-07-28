"""Locally computed corporate-action adjustment factors.

Why this exists: a back-adjusted price series is only correct as of the moment
it was downloaded. Every later split or dividend rewrites the entire history.
Storing the vendor's adjusted_close and cancelling the subscription therefore
freezes a series that silently drifts out of date. Storing RAW OHLCV plus the
event tables and recomputing factors here keeps the dataset correct forever,
and makes as-of-date reconstruction possible.

Convention (CRSP / Yahoo style). For an event with ex-date t:

    split with ratio r  ->  f(t) = 1 / r
    cash dividend d     ->  f(t) = 1 - d / close(previous trading day)

The factor applied to a bar at date u is the product of f(t) over every event
with ex-date t > u, so the most recent bar always has factor 1.0. The product
is evaluated in log space via window functions, which keeps it O(n) and
out-of-core.

    adjusted_close(u) = close(u) * price_factor(u)
"""

from __future__ import annotations

import json
import logging
import shutil

import duckdb

from .config import CATALOG_DIR, DATA_DIR, PARQUET_DIR

log = logging.getLogger("xcap.corpactions")

_FACTOR_SQL = """
WITH bars AS (
    SELECT security_id, date, close,
           LAG(close) OVER (PARTITION BY security_id ORDER BY date) AS prev_close
    FROM read_parquet('{eod}/**/*.parquet')
),
split_f AS (
    SELECT security_id, date,
           -- Several splits can share an ex-date; compound them.
           exp(SUM(ln(1.0 / ratio))) AS f
    FROM read_parquet('{splits}')
    WHERE ratio IS NOT NULL AND ratio > 0
    GROUP BY 1, 2
),
div_f AS (
    -- The dividend must be expressed in the same share terms as the price it
    -- is divided by. close here is RAW (unadjusted for splits), so it pairs
    -- with unadjusted_value, not with value -- value is already restated into
    -- post-split terms and would understate the factor for any security that
    -- split after the payment.
    SELECT d.security_id, d.date,
           exp(SUM(ln(1.0 - COALESCE(d.unadjusted_value, d.value) / b.prev_close))) AS f
    FROM read_parquet('{dividends}') d
    JOIN bars b USING (security_id, date)
    WHERE COALESCE(d.unadjusted_value, d.value) IS NOT NULL
      AND b.prev_close IS NOT NULL AND b.prev_close > 0
      AND COALESCE(d.unadjusted_value, d.value) > 0
      -- Guard against bad vendor rows implying a >=95% yield in one payment,
      -- which would produce a non-positive or absurd factor.
      AND COALESCE(d.unadjusted_value, d.value) / b.prev_close < 0.95
    GROUP BY 1, 2
),
events AS (
    SELECT b.security_id, b.date,
           COALESCE(s.f, 1.0) AS sf,
           COALESCE(s.f, 1.0) * COALESCE(v.f, 1.0) AS pf
    FROM bars b
    LEFT JOIN split_f s USING (security_id, date)
    LEFT JOIN div_f   v USING (security_id, date)
)
SELECT security_id, date,
       exp(SUM(ln(sf)) OVER w_all - SUM(ln(sf)) OVER w_cum) AS split_factor,
       exp(SUM(ln(pf)) OVER w_all - SUM(ln(pf)) OVER w_cum) AS price_factor,
       CAST(year(date) AS INTEGER) AS year
FROM events
WINDOW
    w_all AS (PARTITION BY security_id),
    w_cum AS (PARTITION BY security_id ORDER BY date
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
"""


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DATA_DIR / '_duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order=false")
    return con


#: Used when dividends have not been downloaded yet. price_factor then equals
#: split_factor, i.e. price-return only. Never silently: build_adjustments
#: reports dividends_included=False so downstream code and the QA gate can see
#: that total-return reconstruction is not yet possible.
_NO_DIVIDENDS_SQL = """
div_f AS (
    SELECT NULL::INTEGER AS security_id, NULL::DATE AS date, 1.0 AS f WHERE FALSE
),"""


def build_adjustments() -> dict:
    """Compute and persist split and total-return factors for every bar."""
    out = PARQUET_DIR / "adjustments"
    if out.exists():
        shutil.rmtree(out)

    dividends_path = PARQUET_DIR / "dividends.parquet"
    has_dividends = dividends_path.exists()

    sql = _FACTOR_SQL.format(
        eod=PARQUET_DIR / "eod",
        splits=PARQUET_DIR / "splits.parquet",
        dividends=dividends_path,
    )
    if not has_dividends:
        log.warning("dividends.parquet absent: factors are SPLIT-ONLY "
                    "(price return, not total return)")
        start = sql.index("div_f AS (")
        end = sql.index("events AS (")
        sql = sql[:start] + _NO_DIVIDENDS_SQL.strip() + "\n" + sql[end:]
    con = _connect()
    con.execute(f"""
        COPY ({sql} ORDER BY date, security_id) TO '{out}'
        (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION zstd, OVERWRITE_OR_IGNORE)
    """)
    rows, = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out}/**/*.parquet')"
    ).fetchone()
    con.close()

    files = sorted(out.rglob("*.parquet"))
    return {"path": str(out.relative_to(DATA_DIR)), "rows": rows,
            "files": len(files), "bytes": sum(f.stat().st_size for f in files),
            "dividends_included": has_dividends,
            "factor_meaning": "total return" if has_dividends else "price return (split-only)"}


def reconcile(tolerance: float = 0.01, sample_securities: int | None = None) -> dict:
    """Compare locally rebuilt adjusted closes against the vendor's.

    This is the corporate-action data-quality check. A security whose rebuilt
    series diverges from the vendor's has a missing or wrong split/dividend in
    at least one of the two, and it must be investigated *before* the
    subscription is cancelled.
    """
    con = _connect()
    where = ""
    if sample_securities:
        con.execute(
            f"""CREATE TEMP TABLE picked AS
                SELECT DISTINCT security_id FROM read_parquet('{PARQUET_DIR}/eod/**/*.parquet')
                USING SAMPLE {sample_securities} ROWS"""
        )
        where = "WHERE e.security_id IN (SELECT security_id FROM picked)"

    con.execute(f"""
        CREATE TEMP TABLE cmp AS
        SELECT e.security_id, e.date,
               e.close * a.price_factor AS rebuilt,
               e.vendor_adjusted_close  AS vendor
        FROM read_parquet('{PARQUET_DIR}/eod/**/*.parquet') e
        JOIN read_parquet('{PARQUET_DIR}/adjustments/**/*.parquet') a
          USING (security_id, date)
        {where}
    """)

    total, comparable = con.execute(
        "SELECT COUNT(*), COUNT(*) FILTER (WHERE vendor IS NOT NULL AND vendor > 0) FROM cmp"
    ).fetchone()

    con.execute("""
        CREATE TEMP TABLE diffs AS
        SELECT security_id, date, rebuilt, vendor,
               abs(rebuilt - vendor) / vendor AS rel_err
        FROM cmp WHERE vendor IS NOT NULL AND vendor > 0 AND rebuilt IS NOT NULL
    """)

    stats = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE rel_err <= {tolerance}) AS within,
            COUNT(*) FILTER (WHERE rel_err >  {tolerance}) AS outside,
            median(rel_err) AS median_err,
            quantile_cont(rel_err, 0.99) AS p99_err,
            max(rel_err) AS max_err
        FROM diffs
    """).fetchone()

    worst = con.execute(f"""
        SELECT security_id,
               COUNT(*) FILTER (WHERE rel_err > {tolerance}) AS bad_bars,
               COUNT(*) AS bars,
               max(rel_err) AS max_err
        FROM diffs GROUP BY 1
        HAVING bad_bars > 0
        ORDER BY bad_bars DESC LIMIT 25
    """).fetchall()

    n_secs, n_bad_secs = con.execute(f"""
        SELECT COUNT(DISTINCT security_id),
               COUNT(DISTINCT security_id) FILTER (WHERE rel_err > {tolerance})
        FROM diffs
    """).fetchone()
    con.close()

    within, outside, median_err, p99_err, max_err = stats
    report = {
        "tolerance": tolerance,
        "bars_compared": comparable,
        "bars_joined": total,
        "within_tolerance": within,
        "outside_tolerance": outside,
        "pct_within": round(100.0 * within / max(within + outside, 1), 4),
        "median_rel_err": median_err,
        "p99_rel_err": p99_err,
        "max_rel_err": max_err,
        "securities": n_secs,
        "securities_with_mismatch": n_bad_secs,
        "worst_securities": [
            {"security_id": s, "bad_bars": b, "bars": n, "max_rel_err": e}
            for s, b, n, e in worst
        ],
    }
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "adjustment_reconciliation.json").write_text(json.dumps(report, indent=2))
    return report
