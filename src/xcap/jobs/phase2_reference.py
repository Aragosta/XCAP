"""Phase 2 — reference and non-equity datasets.

Five independent pulls, each cheap enough to complete fully:

  index constituents     ~50 requests x 10   point-in-time membership (2008+)
  exchange details        ~70 requests x 1   trading hours + market holidays
  earnings calendar      ~320 requests x 1   actual vs estimate, monthly chunks
  macro indicators    ~1,290 requests x 10   39 World Bank series, 33 regions
  non-equity EOD      ~9,800 requests x 1    forex, crypto, bonds, money rates

Each sub-job is separately resumable through the ledger, so a partial run
costs nothing to resume.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from ..config import Config, PARQUET_DIR
from ..db import query
from ..eodhd.budget import BudgetExceeded
from ..eodhd.client import EodhdClient, RequestSpec, tally
from ..ledger import Ledger

log = logging.getLogger("xcap.phase2")

# ---- index constituents ------------------------------------------------

#: Headline indices worth point-in-time membership. Anything matching a prefix
#: below is added too, which picks up the S&P sector indices.
INDEX_WHITELIST = {
    "GSPC", "DJI", "IXIC", "NDX", "RUT", "RUI", "RUA", "MID", "SML", "OEX",
    "NYA", "XAX", "VIX", "W5000", "DJT", "DJU", "SOX", "BKX", "HSI", "N225",
    "FTSE", "GDAXI", "FCHI", "STOXX50E", "SSMI", "AEX", "IBEX", "BFX",
}
INDEX_PREFIXES = ("SP500-", "DJ-")

MACRO_INDICATORS = [
    "gdp_current_usd", "gdp_per_capita_usd", "gdp_growth_annual", "gni_usd",
    "gni_per_capita_usd", "gni_ppp_usd", "gni_per_capita_ppp_usd",
    "gross_capital_formation_percent_gdp", "agriculture_value_added_percent_gdp",
    "industry_value_added_percent_gdp", "services_value_added_percent_gdp",
    "inflation_consumer_prices_annual", "consumer_price_index",
    "inflation_gdp_deflator_annual", "real_interest_rate",
    "net_trades_goods_services", "exports_of_goods_services_percent_gdp",
    "imports_of_goods_services_percent_gdp", "merchandise_trade_percent_gdp",
    "high_technology_exports_percent_total", "debt_percent_gdp",
    "revenue_excluding_grants_percent_gdp", "cash_surplus_deficit_percent_gdp",
    "total_debt_service_percent_gni", "population_total",
    "population_growth_annual", "net_migration", "life_expectancy",
    "fertility_rate", "prevalence_hiv_total", "unemployment_total_percent",
    "income_share_lowest_twenty", "poverty_poverty_lines_percent_population",
    "market_cap_domestic_companies_percent_gdp", "mobile_subscriptions_per_hundred",
    "internet_users_per_hundred", "startup_procedures_register",
    "co2_emissions_tons_per_capita", "surface_area_km",
]

#: Major economies plus World Bank aggregates.
MACRO_COUNTRIES = [
    "USA", "CHN", "JPN", "DEU", "IND", "GBR", "FRA", "ITA", "BRA", "CAN",
    "RUS", "MEX", "AUS", "KOR", "ESP", "IDN", "NLD", "SAU", "TUR", "CHE",
    "POL", "BEL", "SWE", "IRL", "ARG", "NOR", "AUT", "ZAF", "DNK", "SGP",
    "WLD", "EUU", "OED",
]

#: Non-equity exchanges pulled in full. Each series is 1 call for all history.
NONEQUITY_EXCHANGES = ["FOREX", "CC", "GBOND", "MONEY"]

EARNINGS_START = date(2000, 1, 1)


def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    out, cur = [], start.replace(day=1)
    while cur <= end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        out.append((cur, min(nxt - timedelta(days=1), end)))
        cur = nxt
    return out


def _tickers(exchange: str) -> list[str]:
    rows = query(
        f"""SELECT api_ticker FROM read_parquet('{PARQUET_DIR / "securities.parquet"}')
            WHERE source_exchange = ? ORDER BY api_ticker""",
        [exchange],
    )
    return [r[0] for r in rows]


async def _run(client: EodhdClient, name: str, specs: list[RequestSpec]) -> dict:
    counts = {"ok": 0, "empty": 0, "not_found": 0, "failed": 0, "cached": 0}
    CHUNK = 250
    for i in range(0, len(specs), CHUNK):
        try:
            results = await client.fetch_all(specs[i:i + CHUNK])
        except BudgetExceeded as exc:
            log.warning("%s: stopping, %s", name, exc)
            counts["budget_exhausted"] = True
            break
        tally(results, counts)
        log.info("%-18s %d/%d  ok=%d empty=%d 404=%d fail=%d",
                 name, min(i + CHUNK, len(specs)), len(specs),
                 counts["ok"], counts["empty"], counts["not_found"], counts["failed"])
    return counts


async def fetch_reference(cfg: Config, ledger: Ledger, *,
                          which: list[str] | None = None) -> dict:
    which = which or ["indices", "exchange-details", "earnings", "macro", "noneq-eod"]
    stats: dict[str, dict] = {}

    async with EodhdClient(cfg, ledger) as client:
        # --- index constituents -------------------------------------
        if "indices" in which:
            res = await client.fetch(
                RequestSpec("exchange-symbol-list", "/exchange-symbol-list/INDX", "INDX.active")
            )
            codes: list[str] = []
            if res.ok and res.body:
                for row in json.loads(res.body):
                    c = row.get("Code") or ""
                    if c in INDEX_WHITELIST or c.startswith(INDEX_PREFIXES):
                        codes.append(c)
            log.info("indices matched: %d", len(codes))
            stats["indices"] = await _run(client, "indices", [
                RequestSpec("fundamentals-index", f"/fundamentals/{c}.INDX", f"{c}.INDX")
                for c in sorted(set(codes))
            ])

        # --- exchange trading hours and holidays --------------------
        if "exchange-details" in which:
            ex = [r[0] for r in query(
                f"SELECT code FROM read_parquet('{PARQUET_DIR / 'exchanges.parquet'}') ORDER BY code"
            )]
            stats["exchange-details"] = await _run(client, "exchange-details", [
                RequestSpec("exchange-details", f"/exchange-details/{c}", c) for c in ex
            ])

        # --- historical earnings calendar ---------------------------
        if "earnings" in which:
            months = _month_ranges(EARNINGS_START, date.today())
            stats["earnings"] = await _run(client, "earnings", [
                RequestSpec("calendar-earnings", "/calendar/earnings",
                      f"{a.isoformat()}_{b.isoformat()}",
                      {"from": a.isoformat(), "to": b.isoformat()})
                for a, b in months
            ])

        # --- macro indicators ---------------------------------------
        if "macro" in which:
            stats["macro"] = await _run(client, "macro", [
                RequestSpec("macro-indicator", f"/macro-indicator/{c}", f"{c}.{ind}",
                      {"indicator": ind})
                for c in MACRO_COUNTRIES for ind in MACRO_INDICATORS
            ])

        # --- non-equity EOD -----------------------------------------
        if "noneq-eod" in which:
            specs = []
            for exch in NONEQUITY_EXCHANGES:
                for t in _tickers(exch):
                    # Separate endpoint namespace from equity EOD so the two
                    # never mix in the raw archive or the parquet build.
                    specs.append(RequestSpec("eod-nonequity", f"/eod/{t}", t,
                                       {"period": "d", "order": "a"}))
            log.info("non-equity series: %d", len(specs))
            stats["noneq-eod"] = await _run(client, "noneq-eod", specs)

    return stats
