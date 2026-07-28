"""Probe which endpoints this token can actually reach, and what they return.

Plan descriptions are marketing documents; the token is the ground truth. This
hits every candidate endpoint once with a known-good symbol and records the
status code and payload shape, so Phase 2+ scoping is based on measured access
rather than on what the pricing page claims is "included".

Cheap by design (~200 calls), except bulk-fundamentals which costs 100 on its
own and is probed anyway because it is the single biggest cost lever in the
whole project: 0.2 calls/symbol versus 10.
"""

from __future__ import annotations

import json
import logging

from ..config import CATALOG_DIR, Config
from ..eodhd.client import EodhdClient, FatalApiError, RequestSpec
from ..ledger import Ledger

log = logging.getLogger("xcap.entitlements")

# (endpoint name, path, params, why it matters)
PROBES: list[tuple[str, str, dict, str]] = [
    ("fundamentals", "/fundamentals/AAPL.US", {},
     "per-symbol fundamentals: 10 calls each"),
    ("bulk-fundamentals", "/bulk-fundamentals/NASDAQ", {"limit": 1, "offset": 0},
     "batch fundamentals: 100 calls per 500 symbols (50x cheaper)"),
    ("fundamentals-index", "/fundamentals/GSPC.INDX", {},
     "index constituents + HistoricalTickerComponents"),
    ("fundamentals-etf", "/fundamentals/SPY.US", {},
     "ETF holdings and allocations"),
    ("historical-market-cap", "/historical-market-cap/AAPL.US", {},
     "market cap time series (US only, weekly from 2021-07)"),
    ("insider", "/sec-filings/AAPL.US/form4", {"page[limit]": 1},
     "Form 4 insider transactions, back to 2000"),
    ("macro-indicator", "/macro-indicator/USA", {"indicator": "gdp_current_usd"},
     "39 World Bank series per country, from 1960"),
    ("calendar-earnings", "/calendar/earnings", {"symbols": "AAPL.US"},
     "earnings actual vs estimate, 1 call"),
    ("calendar-ipos", "/calendar/ipos", {},
     "IPO calendar from 2015"),
    ("news", "/news", {"s": "AAPL.US", "limit": 1},
     "news + sentiment, 5 calls per request"),
    ("sentiments", "/sentiments", {"s": "AAPL.US"},
     "aggregated sentiment series"),
    ("options", "/options/AAPL.US", {},
     "US options chains with greeks"),
    ("intraday", "/intraday/AAPL.US", {"interval": "5m"},
     "intraday bars, 5 calls per request"),
    ("exchange-details", "/exchange-details/US", {},
     "trading hours + market holidays, needed for calendar QA"),
    ("technical-splitadj", "/technical/AAPL.US", {"function": "splitadjusted", "period": "d"},
     "split-adjusted OHLC without dividend adjustment"),
    ("eod-forex", "/eod/EURUSD.FOREX", {},
     "FX history"),
    ("eod-crypto", "/eod/BTC-USD.CC", {},
     "crypto history"),
    ("eod-bond", "/eod/US10Y.GBOND", {},
     "government bond yields"),
    ("screener", "/screener", {"sort": "market_capitalization.desc", "limit": 1},
     "screener API"),
    ("search", "/search/apple", {},
     "symbol search"),
]


def _shape(body: bytes | None) -> str:
    if not body:
        return "empty"
    try:
        obj = json.loads(body)
    except Exception:  # noqa: BLE001
        return f"non-json ({len(body)}b)"
    if isinstance(obj, list):
        keys = list(obj[0].keys())[:6] if obj and isinstance(obj[0], dict) else []
        return f"list[{len(obj)}] {keys}"
    if isinstance(obj, dict):
        return f"dict keys={list(obj.keys())[:8]}"
    return type(obj).__name__


async def probe_entitlements(cfg: Config, ledger: Ledger) -> dict:
    results: list[dict] = []
    async with EodhdClient(cfg, ledger) as client:
        for name, path, params, why in PROBES:
            spec = RequestSpec(endpoint=f"probe-{name}", path=path, key="probe",
                               params=params)
            try:
                res = await client.fetch(spec)
                status, http = res.status, res.http_status
                shape = _shape(res.body)
            except FatalApiError as exc:
                status, http, shape = "forbidden", 403, str(exc)[:120]
            results.append({
                "endpoint": name, "path": path, "why": why,
                "status": status, "http": http, "shape": shape,
                "accessible": status in ("ok", "empty"),
            })
            log.info("%-24s %-12s %s", name, status, shape[:90])

    report = {"probes": results,
              "accessible": sum(1 for r in results if r["accessible"]),
              "total": len(results)}
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "entitlements.json").write_text(json.dumps(report, indent=2))
    return report
