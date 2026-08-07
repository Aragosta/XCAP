"""The Phase 1 target universe: US tradable listings.

Defined in one place so the fetch job, the parquet build, and the QA gate all
agree on what "the universe" means. Changing this definition changes coverage
accounting, so it is a deliberate, versioned decision rather than an inline
filter repeated across modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PARQUET_DIR
from .db import query, quoted

#: Real exchange listings. Excludes OTC tiers (PINK, OTCQB, OTCGREY, ...) and
#: NMFQS mutual-fund quote lines, which together make up two thirds of the raw
#: US symbol count but are not a tradable equity universe.
VENUES = ("NYSE", "NASDAQ", "AMEX", "NYSE MKT", "NYSE ARCA", "BATS")

TYPES = ("Common Stock", "ETF", "Preferred Stock")

#: Dataset floor. Set by the delisting-coverage probe: the vendor's delisted
#: archive begins ~1997-98, so no earlier start year is survivorship-bias free.
#: See data/catalog/delisting_probe_US.json and xcap.jobs.probe_delisting.
START_DATE = "2000-01-01"


@dataclass(frozen=True)
class Security:
    security_id: int
    api_ticker: str
    type: str
    venue: str
    is_delisted: bool


def select(*, exchange: str = "US") -> list[Security]:
    """Every security in the Phase 1 universe, delisted included.

    Delisted names are pulled unconditionally and are never filtered by date at
    fetch time: deciding membership from data you have not downloaded yet
    rebuilds the exact survivorship bias this universe exists to avoid.
    """
    rows = query(
        f"""
        SELECT security_id, api_ticker, type, venue, is_delisted
        FROM read_parquet('{PARQUET_DIR / "securities.parquet"}')
        WHERE source_exchange = ?
          AND venue IN ({quoted(VENUES)})
          AND type  IN ({quoted(TYPES)})
        ORDER BY security_id
        """,
        [exchange],
    )
    return [Security(int(a), b, c, d, bool(e)) for a, b, c, d, e in rows]
