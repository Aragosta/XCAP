"""Raw fundamentals JSON -> parquet.

Financials and shares-outstanding are the two datasets large enough (tens of
millions of rows across 32,525 documents) to need the same staged,
out-of-core approach as EOD. General facts and coverage are one row per
security and built directly.

Gated on Phase 3 block completeness the same way phase1.build_all gates on
the fetch block: a parquet built from a partially downloaded universe looks
complete to anything that reads it, which is worse than no parquet at all.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CATALOG_DIR, DATA_DIR, PARQUET_DIR
from ..db import connect
from ..eodhd.client import EodhdClient
from ..ledger import Ledger
from ..schemas.fundamentals import (
    FUNDAMENTALS_COVERAGE, FUNDAMENTALS_FINANCIALS,
    FUNDAMENTALS_GENERAL, FUNDAMENTALS_SHARES,
)
from ..universe import select
from . import ROWS_PER_STAGE_FILE, sha256_file

log = logging.getLogger("xcap.transform.fundamentals")

STAGING = DATA_DIR / "_staging" / "fundamentals"

STATEMENTS = (
    ("Income_Statement", "income_statement"),
    ("Balance_Sheet", "balance_sheet"),
    ("Cash_Flow", "cash_flow"),
)

# Fields carried into FUNDAMENTALS_FINANCIALS are never inferred from the
# statement -- every vendor-reported key becomes a row, aside from the
# metadata fields extracted separately (date, filing_date, currency_symbol).
_STATEMENT_META = {"date", "filing_date", "currency_symbol"}


def _f(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i(value: object) -> int | None:
    v = _f(value)
    return int(v) if v is not None else None


def _resolved_universe(ledger: Ledger) -> tuple[list, bool]:
    """Universe securities with a terminal fundamentals answer, and whether
    every security in the universe has one (the completeness gate)."""
    universe = select()
    done = ledger.resolved("fundamentals")
    resolved = [s for s in universe if s.api_ticker in done]
    return resolved, len(resolved) >= len(universe)


def _load(row) -> dict | None:
    if not row["raw_path"]:
        return None
    try:
        return json.loads(EodhdClient.read_raw(Path(row["raw_path"])))
    except Exception as exc:  # noqa: BLE001 - never abort the build over one bad file
        log.error("unreadable raw for %s: %s", row["key"], exc)
        return None


# ------------------------------------------------------------ coverage + general

def build_general_and_coverage(ledger: Ledger, sec_by_ticker: dict) -> dict:
    ok_rows = {r["key"]: r for r in ledger.rows("fundamentals", status="ok")}
    universe = select()

    cov: dict[str, list] = {f.name: [] for f in FUNDAMENTALS_COVERAGE}
    gen: dict[str, list] = {f.name: [] for f in FUNDAMENTALS_GENERAL}

    for n, sec in enumerate(universe, 1):
        row = ok_rows.get(sec.api_ticker)
        is_etf = sec.type == "ETF"
        if row is None:
            # empty / not_found -- no document at all, still gets a coverage row.
            cov["security_id"].append(sec.security_id)
            cov["api_ticker"].append(sec.api_ticker)
            cov["type"].append(sec.type)
            cov["is_delisted"].append(sec.is_delisted)
            cov["is_etf"].append(is_etf)
            cov["has_financials"].append(False)
            cov["n_annual_periods"].append(0)
            cov["n_quarterly_periods"].append(0)
            cov["first_fiscal_year"].append(None)
            cov["last_fiscal_year"].append(None)
            cov["has_shares_outstanding"].append(False)
            cov["updated_at"].append(None)
            cov["doc_bytes"].append(0)
            continue

        d = _load(row)
        if d is None:
            continue
        g = d.get("General") or {}
        h = d.get("Highlights") or {}
        v = d.get("Valuation") or {}
        ss = d.get("SharesStats") or {}
        t = d.get("Technicals") or {}
        inc = ((d.get("Financials") or {}).get("Income_Statement") or {})
        yearly, quarterly = inc.get("yearly") or {}, inc.get("quarterly") or {}
        shares = d.get("outstandingShares") or {}
        has_shares = bool((shares.get("annual") or {})) or bool((shares.get("quarterly") or {}))

        fys = sorted(yearly)  # ISO date strings sort correctly
        cov["security_id"].append(sec.security_id)
        cov["api_ticker"].append(sec.api_ticker)
        cov["type"].append(sec.type)
        cov["is_delisted"].append(sec.is_delisted)
        cov["is_etf"].append(is_etf)
        cov["has_financials"].append(bool(yearly))
        cov["n_annual_periods"].append(len(yearly))
        cov["n_quarterly_periods"].append(len(quarterly))
        cov["first_fiscal_year"].append(fys[0] if fys else None)
        cov["last_fiscal_year"].append(fys[-1] if fys else None)
        cov["has_shares_outstanding"].append(has_shares)
        cov["updated_at"].append(g.get("UpdatedAt"))
        cov["doc_bytes"].append(row["bytes"] or 0)

        if is_etf:
            continue  # ETFs carry ETF_Data instead; no General/Highlights facts to speak of

        gen["security_id"].append(sec.security_id)
        gen["api_ticker"].append(sec.api_ticker)
        gen["name"].append(g.get("Name"))
        gen["sector"].append(g.get("Sector"))
        gen["industry"].append(g.get("Industry"))
        gen["gic_sector"].append(g.get("GicSector"))
        gen["gic_industry"].append(g.get("GicIndustry"))
        gen["isin"].append(g.get("ISIN"))
        gen["cik"].append(g.get("CIK"))
        gen["ipo_date"].append(g.get("IPODate") or None)
        # Vendor delisting flag/date. Taken from the fundamentals doc, not from
        # `sec.is_delisted`: that one is derived from which exchange-symbol-list the
        # ticker appeared on and carries no date at all.
        gen["is_delisted"].append(bool(g.get("IsDelisted")))
        gen["delisted_date"].append(g.get("DelistedDate") or None)
        gen["fiscal_year_end"].append(g.get("FiscalYearEnd"))
        gen["full_time_employees"].append(_i(g.get("FullTimeEmployees")))
        gen["market_cap"].append(_f(h.get("MarketCapitalization")))
        gen["ebitda"].append(_f(h.get("EBITDA")))
        gen["pe_ratio"].append(_f(h.get("PERatio")))
        gen["peg_ratio"].append(_f(h.get("PEGRatio")))
        gen["book_value"].append(_f(h.get("BookValue")))
        gen["dividend_share"].append(_f(h.get("DividendShare")))
        gen["dividend_yield"].append(_f(h.get("DividendYield")))
        gen["earnings_share"].append(_f(h.get("EarningsShare")))
        gen["profit_margin"].append(_f(h.get("ProfitMargin")))
        gen["operating_margin_ttm"].append(_f(h.get("OperatingMarginTTM")))
        gen["return_on_assets_ttm"].append(_f(h.get("ReturnOnAssetsTTM")))
        gen["return_on_equity_ttm"].append(_f(h.get("ReturnOnEquityTTM")))
        gen["revenue_ttm"].append(_f(h.get("RevenueTTM")))
        gen["revenue_per_share_ttm"].append(_f(h.get("RevenuePerShareTTM")))
        gen["gross_profit_ttm"].append(_f(h.get("GrossProfitTTM")))
        gen["diluted_eps_ttm"].append(_f(h.get("DilutedEpsTTM")))
        gen["trailing_pe"].append(_f(v.get("TrailingPE")))
        gen["forward_pe"].append(_f(v.get("ForwardPE")))
        gen["price_sales_ttm"].append(_f(v.get("PriceSalesTTM")))
        gen["price_book_mrq"].append(_f(v.get("PriceBookMRQ")))
        gen["enterprise_value"].append(_f(v.get("EnterpriseValue")))
        gen["shares_outstanding"].append(_f(ss.get("SharesOutstanding")))
        gen["shares_float"].append(_f(ss.get("SharesFloat")))
        gen["percent_insiders"].append(_f(ss.get("PercentInsiders")))
        gen["percent_institutions"].append(_f(ss.get("PercentInstitutions")))
        gen["beta"].append(_f(t.get("Beta")))
        gen["week_52_high"].append(_f(t.get("52WeekHigh")))
        gen["week_52_low"].append(_f(t.get("52WeekLow")))
        gen["updated_at"].append(g.get("UpdatedAt"))

        if n % 5000 == 0:
            log.info("  general/coverage %d/%d", n, len(universe))

    def _write(cols, schema, name):
        # date32 fields were appended as ISO strings or None; cast via DuckDB.
        table = pa.table(cols)
        path = PARQUET_DIR / f"{name}.parquet"
        con = connect()
        date_cols = [f.name for f in schema if f.type == pa.date32()]
        select_cols = ", ".join(
            f"CAST({c} AS DATE) AS {c}" if c in date_cols else c
            for c in cols
        )
        con.register("t", table)
        con.execute(f"COPY (SELECT {select_cols} FROM t ORDER BY security_id) "
                    f"TO '{path}' (FORMAT PARQUET, COMPRESSION zstd)")
        con.close()
        return {"path": str(path.relative_to(DATA_DIR)), "rows": table.num_rows,
                "bytes": path.stat().st_size, "sha256": sha256_file(path)}

    return {
        "coverage": _write(cov, FUNDAMENTALS_COVERAGE, "fundamentals_coverage"),
        "general": _write(gen, FUNDAMENTALS_GENERAL, "fundamentals_general"),
    }


# ------------------------------------------------------------ financials (long, staged)

def build_financials(ledger: Ledger, sec_by_ticker: dict) -> dict:
    stage_dir = STAGING / "financials"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    cols: dict[str, list] = {f.name: [] for f in FUNDAMENTALS_FINANCIALS}
    part = 0
    total_rows = 0
    securities_with_data = 0

    def flush() -> None:
        nonlocal cols, part
        if not cols["security_id"]:
            return
        pq.write_table(pa.table(cols), stage_dir / f"part-{part:05d}.parquet",
                       compression="zstd", compression_level=3)
        part += 1
        cols = {f.name: [] for f in FUNDAMENTALS_FINANCIALS}

    ok_rows = ledger.rows("fundamentals", status="ok")
    log.info("parsing financials from %d fundamentals documents", len(ok_rows))

    for n, row in enumerate(ok_rows, 1):
        sec = sec_by_ticker.get(row["key"])
        if sec is None or sec.type == "ETF":
            continue
        d = _load(row)
        if d is None:
            continue
        fin = d.get("Financials") or {}
        got_any = False
        for src_key, statement_name in STATEMENTS:
            stmt = fin.get(src_key) or {}
            for period_type, periods in (("annual", stmt.get("yearly") or {}),
                                         ("quarterly", stmt.get("quarterly") or {})):
                for period_end, fields in periods.items():
                    if not isinstance(fields, dict):
                        continue
                    filing_date = fields.get("filing_date") or None
                    # A minority of vendor records carry a filing_date that
                    # predates the period it reports on -- physically
                    # impossible for a real filing, and consistent with the
                    # vendor occasionally misaligning filing_date against the
                    # adjacent fiscal year. Drop rather than trust it: an
                    # impossible filing_date is a look-ahead risk (§3.3) if
                    # anything ever joins on it naively, and the alternative
                    # -- keeping a wrong-but-plausible date -- is worse than
                    # keeping no date at all.
                    if filing_date and filing_date < period_end:
                        filing_date = None
                    currency = fields.get("currency_symbol")
                    for item, value in fields.items():
                        if item in _STATEMENT_META:
                            continue
                        fv = _f(value)
                        if fv is None:
                            continue  # absence is an absent row, not a NULL row
                        cols["security_id"].append(sec.security_id)
                        cols["statement"].append(statement_name)
                        cols["period_type"].append(period_type)
                        cols["period_end"].append(period_end)
                        cols["filing_date"].append(filing_date)
                        cols["currency"].append(currency)
                        cols["item"].append(item)
                        cols["value"].append(fv)
                        got_any = True
        if got_any:
            securities_with_data += 1
            total_rows_local = len(cols["security_id"])
        if len(cols["security_id"]) >= ROWS_PER_STAGE_FILE:
            total_rows += len(cols["security_id"])
            flush()
        if n % 5000 == 0:
            log.info("  financials %d/%d documents, %d rows staged",
                     n, len(ok_rows), total_rows + len(cols["security_id"]))

    total_rows += len(cols["security_id"])
    flush()
    log.info("staged %d financial line-item rows from %d securities",
             total_rows, securities_with_data)

    out = PARQUET_DIR / "fundamentals_financials.parquet"
    con = connect()
    if total_rows:
        con.execute(f"""
            COPY (
                SELECT security_id, statement, period_type,
                       CAST(period_end AS DATE) AS period_end,
                       CAST(filing_date AS DATE) AS filing_date,
                       currency, item, value
                FROM read_parquet('{stage_dir}/*.parquet')
                ORDER BY security_id, statement, period_type, period_end
            ) TO '{out}' (FORMAT PARQUET, COMPRESSION zstd)
        """)
    else:
        pq.write_table(pa.table({f.name: [] for f in FUNDAMENTALS_FINANCIALS},
                                schema=FUNDAMENTALS_FINANCIALS), out)
    con.close()
    shutil.rmtree(stage_dir, ignore_errors=True)

    return {"path": str(out.relative_to(DATA_DIR)), "rows": total_rows,
            "securities": securities_with_data, "bytes": out.stat().st_size,
            "sha256": sha256_file(out)}


# ------------------------------------------------------------ shares outstanding (long)

def build_shares(ledger: Ledger, sec_by_ticker: dict) -> dict:
    cols: dict[str, list] = {f.name: [] for f in FUNDAMENTALS_SHARES}
    ok_rows = ledger.rows("fundamentals", status="ok")

    for row in ok_rows:
        sec = sec_by_ticker.get(row["key"])
        if sec is None or sec.type == "ETF":
            continue
        d = _load(row)
        if d is None:
            continue
        shares = d.get("outstandingShares") or {}
        for period_type in ("annual", "quarterly"):
            for _, rec in (shares.get(period_type) or {}).items():
                if not isinstance(rec, dict):
                    continue
                dt = rec.get("dateFormatted") or rec.get("date")
                sh = _i(rec.get("shares"))
                if not dt or sh is None:
                    continue
                cols["security_id"].append(sec.security_id)
                cols["period_type"].append(period_type)
                cols["date"].append(dt)
                cols["shares"].append(sh)

    out = PARQUET_DIR / "fundamentals_shares.parquet"
    con = connect()
    if cols["security_id"]:
        table = pa.table(cols)
        con.register("t", table)
        con.execute(f"""
            COPY (SELECT security_id, period_type, CAST(date AS DATE) AS date, shares
                  FROM t ORDER BY security_id, date)
            TO '{out}' (FORMAT PARQUET, COMPRESSION zstd)
        """)
    else:
        pq.write_table(pa.table({f.name: [] for f in FUNDAMENTALS_SHARES},
                                schema=FUNDAMENTALS_SHARES), out)
    con.close()
    return {"path": str(out.relative_to(DATA_DIR)), "rows": len(cols["security_id"]),
            "bytes": out.stat().st_size, "sha256": sha256_file(out)}


# ------------------------------------------------------------ orchestration

def build_all(ledger: Ledger) -> dict:
    """Build every Phase 3 dataset, gated on full universe resolution."""
    universe = select()
    resolved, is_complete = _resolved_universe(ledger)

    if not is_complete:
        reason = f"{len(resolved):,}/{len(universe):,} securities resolved"
        log.warning("skipping fundamentals build: block incomplete (%s)", reason)
        for name in ("fundamentals_coverage", "fundamentals_general",
                     "fundamentals_financials", "fundamentals_shares"):
            (PARQUET_DIR / f"{name}.parquet").unlink(missing_ok=True)
        return {"skipped": reason}

    sec_by_ticker = {s.api_ticker: s for s in universe}
    datasets = {}
    datasets.update(build_general_and_coverage(ledger, sec_by_ticker))
    datasets["financials"] = build_financials(ledger, sec_by_ticker)
    datasets["shares"] = build_shares(ledger, sec_by_ticker)

    manifest = {"universe": len(universe), "datasets": datasets}
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "phase3_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
