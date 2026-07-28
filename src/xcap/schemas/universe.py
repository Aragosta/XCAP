"""Explicit Arrow schemas for the Phase 0 datasets.

Schemas are declared, never inferred. Inference across 150k heterogeneous rows
silently produces different column types on different runs, which is exactly the
kind of drift that is impossible to debug once the subscription is gone.
"""

from __future__ import annotations

import pyarrow as pa

EXCHANGES = pa.schema([
    pa.field("code", pa.string(), nullable=False),
    pa.field("name", pa.string()),
    pa.field("operating_mic", pa.string()),
    pa.field("country", pa.string()),
    pa.field("currency", pa.string()),
    pa.field("country_iso2", pa.string()),
    pa.field("country_iso3", pa.string()),
    pa.field("snapshot_date", pa.date32(), nullable=False),
])

SECURITIES = pa.schema([
    # Stable internal key. Never join on ticker: symbols are recycled after
    # delisting, so a ticker is not a security identity.
    pa.field("security_id", pa.int32(), nullable=False),
    # Exchange code used in the API path (e.g. "US"), which is what you must
    # append to Code to address the security in later endpoints.
    pa.field("source_exchange", pa.string(), nullable=False),
    pa.field("code", pa.string(), nullable=False),
    # Ticker as passed to /eod, /fundamentals, etc.
    pa.field("api_ticker", pa.string(), nullable=False),
    # Actual venue reported in the listing (e.g. "NASDAQ" when source is "US").
    pa.field("venue", pa.string()),
    pa.field("name", pa.string()),
    pa.field("country", pa.string()),
    pa.field("currency", pa.string()),
    pa.field("type", pa.string()),
    pa.field("isin", pa.string()),
    # Which source list(s) the security appeared in. is_delisted is derived:
    # present in the delisted list and absent from the active one.
    pa.field("listed_active", pa.bool_(), nullable=False),
    pa.field("listed_delisted", pa.bool_(), nullable=False),
    pa.field("is_delisted", pa.bool_(), nullable=False),
    pa.field("snapshot_date", pa.date32(), nullable=False),
])
