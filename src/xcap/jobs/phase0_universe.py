"""Phase 0 — build the security universe.

This runs first and its output is frozen. Every later phase iterates this table,
so if the universe shifts mid-extraction, coverage accounting becomes
meaningless. Cost is ~2 calls per exchange plus 1, i.e. a rounding error against
the daily budget.

The delisted pass is the single most important step in the whole project: it is
what separates a survivorship-bias-free dataset from a worthless one.
"""

from __future__ import annotations

import json
import logging

from ..config import Config
from ..eodhd.client import EodhdClient, tally
from ..eodhd.endpoints import exchange_symbol_list, exchanges_list
from ..ledger import Ledger

log = logging.getLogger("xcap.phase0")


async def fetch_universe(cfg: Config, ledger: Ledger, *, force: bool = False) -> dict:
    async with EodhdClient(cfg, ledger) as client:
        # 1. Exchange directory.
        res = await client.fetch(exchanges_list(), force=force)
        if not res.ok:
            raise SystemExit(f"could not fetch exchanges-list: {res.status} {res.error}")
        exchanges = json.loads(res.body)
        codes = sorted({e["Code"] for e in exchanges if e.get("Code")})

        # "US" is a virtual exchange spanning NYSE/NASDAQ/AMEX/BATS and is how
        # US tickers are addressed everywhere else in the API. Make sure it is
        # present even if the directory enumerates the venues separately.
        if "US" not in codes:
            codes.append("US")
            codes.sort()

        log.info("exchanges: %d (from cache: %s)", len(codes), res.from_cache)

        # 2. Active + delisted ticker lists for every exchange.
        specs = [
            exchange_symbol_list(code, delisted=delisted)
            for code in codes
            for delisted in (False, True)
        ]
        results = await client.fetch_all(specs, force=force)

    counts = tally(results)

    log.info(
        "symbol lists: %d ok, %d empty, %d not_found, %d failed",
        counts["ok"], counts["empty"], counts["not_found"], counts["failed"],
    )
    return {"exchanges": len(codes), "requests": len(specs), **counts}
