"""Arrow schemas for Phase 3 fundamentals.

Financials and shares outstanding are stored LONG (one row per line item per
period) rather than wide. The vendor's own item set is not stable across
companies or years -- a wide schema would either drop fields silently or need
constant maintenance. Long format survives that for free: an absent item is
an absent row, never a column that has to be added later.

Snapshot fields (Highlights, Valuation, Technicals, ...) have no vintage --
see docs/DATA_RETRIEVAL.md §3.6 -- so they get exactly one row per security,
clearly named to make that current-only status impossible to miss.
"""

from __future__ import annotations

import pyarrow as pa

#: One row per security with a fundamentals document. The single source of
#: truth for "do we have usable data for this security", independent of
#: whether it has financial statements at all (ETFs never do; see
#: fundamentals.py transform for FUNDAMENTALS_FINANCIALS coverage).
FUNDAMENTALS_COVERAGE = pa.schema([
    pa.field("security_id", pa.int32(), nullable=False),
    pa.field("api_ticker", pa.string(), nullable=False),
    pa.field("type", pa.string()),          # Common Stock / ETF / Preferred Stock
    pa.field("is_delisted", pa.bool_()),
    pa.field("is_etf", pa.bool_(), nullable=False),
    pa.field("has_financials", pa.bool_(), nullable=False),
    pa.field("n_annual_periods", pa.int32()),
    pa.field("n_quarterly_periods", pa.int32()),
    pa.field("first_fiscal_year", pa.date32()),
    pa.field("last_fiscal_year", pa.date32()),
    pa.field("has_shares_outstanding", pa.bool_(), nullable=False),
    pa.field("updated_at", pa.string()),    # vendor's own last-update stamp, as given
    pa.field("doc_bytes", pa.int64()),
])

#: Current-snapshot company facts. No history, no vintage -- see §3.6.
FUNDAMENTALS_GENERAL = pa.schema([
    pa.field("security_id", pa.int32(), nullable=False),
    pa.field("api_ticker", pa.string(), nullable=False),
    pa.field("name", pa.string()),
    pa.field("sector", pa.string()),
    pa.field("industry", pa.string()),
    pa.field("gic_sector", pa.string()),
    pa.field("gic_industry", pa.string()),
    pa.field("isin", pa.string()),
    pa.field("cik", pa.string()),
    pa.field("ipo_date", pa.date32()),
    # The vendor's own delisting date, the only non-circular source for one: inferring
    # it from the last price print cannot tell a delisting from the end of the download.
    # Populated for ~100% of names the vendor flags delisted, median 0 days from the
    # last print. Feeds `delist_dates` in research/BACKTEST.py.
    pa.field("is_delisted", pa.bool_()),
    pa.field("delisted_date", pa.date32()),
    pa.field("fiscal_year_end", pa.string()),
    pa.field("full_time_employees", pa.int64()),
    # Highlights / Valuation / SharesStats / Technicals -- current only.
    pa.field("market_cap", pa.float64()),
    pa.field("ebitda", pa.float64()),
    pa.field("pe_ratio", pa.float64()),
    pa.field("peg_ratio", pa.float64()),
    pa.field("book_value", pa.float64()),
    pa.field("dividend_share", pa.float64()),
    pa.field("dividend_yield", pa.float64()),
    pa.field("earnings_share", pa.float64()),
    pa.field("profit_margin", pa.float64()),
    pa.field("operating_margin_ttm", pa.float64()),
    pa.field("return_on_assets_ttm", pa.float64()),
    pa.field("return_on_equity_ttm", pa.float64()),
    pa.field("revenue_ttm", pa.float64()),
    pa.field("revenue_per_share_ttm", pa.float64()),
    pa.field("gross_profit_ttm", pa.float64()),
    pa.field("diluted_eps_ttm", pa.float64()),
    pa.field("trailing_pe", pa.float64()),
    pa.field("forward_pe", pa.float64()),
    pa.field("price_sales_ttm", pa.float64()),
    pa.field("price_book_mrq", pa.float64()),
    pa.field("enterprise_value", pa.float64()),
    pa.field("shares_outstanding", pa.float64()),
    pa.field("shares_float", pa.float64()),
    pa.field("percent_insiders", pa.float64()),
    pa.field("percent_institutions", pa.float64()),
    pa.field("beta", pa.float64()),
    pa.field("week_52_high", pa.float64()),
    pa.field("week_52_low", pa.float64()),
    pa.field("updated_at", pa.string()),
])

#: Long: one row per (security, statement, period type, period end, item).
#: filing_date is what makes this point-in-time-safe -- see §3.3. Never index
#: on `date` (period end) alone; a Q4 figure isn't knowable until filed.
FUNDAMENTALS_FINANCIALS = pa.schema([
    pa.field("security_id", pa.int32(), nullable=False),
    pa.field("statement", pa.string(), nullable=False),   # income_statement / balance_sheet / cash_flow
    pa.field("period_type", pa.string(), nullable=False), # annual / quarterly
    pa.field("period_end", pa.date32(), nullable=False),
    pa.field("filing_date", pa.date32()),
    pa.field("currency", pa.string()),
    pa.field("item", pa.string(), nullable=False),
    pa.field("value", pa.float64(), nullable=False),
])

#: Long: shares outstanding over time (not just the current snapshot).
FUNDAMENTALS_SHARES = pa.schema([
    pa.field("security_id", pa.int32(), nullable=False),
    pa.field("period_type", pa.string(), nullable=False),  # annual / quarterly
    pa.field("date", pa.date32(), nullable=False),
    pa.field("shares", pa.int64(), nullable=False),
])
