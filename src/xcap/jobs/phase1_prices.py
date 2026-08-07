"""Phase 1 — EOD prices, splits and dividends for the tradable US universe.

Three endpoints at 1 call each, over ~32.5k securities, so a complete pull is
~97.5k calls: right at the edge of the default 100k/day cap. The job is chunked
and resumable, and stops cleanly when the budget runs out rather than dying
mid-flight — re-run it the next GMT day and it picks up exactly where it
stopped, spending nothing on what it already has.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from ..config import Config
from ..eodhd import endpoints
from ..eodhd.budget import BudgetExceeded
from ..eodhd.client import EodhdClient, RequestSpec, tally
from ..ledger import Ledger
from ..universe import Security, select

log = logging.getLogger("xcap.phase1")

BUILDERS: dict[str, Callable[[str], RequestSpec]] = {
    "eod": endpoints.eod,
    "splits": endpoints.splits,
    "dividends": endpoints.dividends,
}

CHUNK = 500


async def fetch_prices(
    cfg: Config,
    ledger: Ledger,
    *,
    which: list[str] | None = None,
    limit: int | None = None,
    seed_sample: int | None = None,
) -> dict:
    which = which or ["eod", "splits", "dividends"]
    universe: list[Security] = select()
    if seed_sample:
        # Deterministic subsample for smoke-testing the pipeline end to end
        # before committing tens of thousands of calls.
        import random
        universe = random.Random(seed_sample).sample(universe, min(limit or 200, len(universe)))
    elif limit:
        universe = universe[:limit]

    log.info("universe: %d securities | endpoints: %s", len(universe), ", ".join(which))

    stats: dict[str, dict[str, int]] = {
        name: {"ok": 0, "empty": 0, "not_found": 0, "failed": 0, "cached": 0}
        for name in which
    }
    budget_hit = False

    async with EodhdClient(cfg, ledger) as client:
        for name in which:
            if budget_hit:
                break
            build = BUILDERS[name]
            started = time.monotonic()
            done = 0
            for i in range(0, len(universe), CHUNK):
                chunk = universe[i:i + CHUNK]
                specs = [build(s.api_ticker) for s in chunk]
                try:
                    results = await client.fetch_all(specs)
                except BudgetExceeded as exc:
                    log.warning("stopping: %s", exc)
                    budget_hit = True
                    break

                tally(results, stats[name])

                done += len(chunk)
                rate = done / max(time.monotonic() - started, 1e-9)
                remaining = (len(universe) - done) / max(rate, 1e-9)
                log.info(
                    "%s  %d/%d  (%.0f/s, ~%.0f min left)  ok=%d empty=%d 404=%d fail=%d",
                    name, done, len(universe), rate, remaining / 60,
                    stats[name]["ok"], stats[name]["empty"],
                    stats[name]["not_found"], stats[name]["failed"],
                )

    return {
        "universe": len(universe),
        "endpoints": stats,
        "budget_exhausted": budget_hit,
    }
