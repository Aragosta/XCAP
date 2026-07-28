"""Arrow schemas for Phase 1 price and corporate-action datasets."""

from __future__ import annotations

import pyarrow as pa

#: Daily bars. OHLCV is stored exactly as the vendor returns it: RAW, adjusted
#: for neither splits nor dividends. vendor_adjusted_close is kept only so the
#: locally-computed adjustment factors can be reconciled against it — never
#: use it directly, because a back-adjusted series frozen on disk goes stale
#: the moment another corporate action happens.
EOD = pa.schema([
    pa.field("security_id", pa.int32(), nullable=False),
    pa.field("api_ticker", pa.string(), nullable=False),
    pa.field("date", pa.date32(), nullable=False),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("vendor_adjusted_close", pa.float64()),
    # Volume is split-adjusted by the vendor even though OHLC is not.
    pa.field("volume", pa.int64()),
])

SPLITS = pa.schema([
    pa.field("security_id", pa.int32(), nullable=False),
    pa.field("api_ticker", pa.string(), nullable=False),
    pa.field("date", pa.date32(), nullable=False),
    pa.field("split_to", pa.float64()),      # new shares
    pa.field("split_from", pa.float64()),    # per old shares
    pa.field("ratio", pa.float64()),         # split_to / split_from
    pa.field("raw", pa.string()),            # vendor string, kept for audit
])

DIVIDENDS = pa.schema([
    pa.field("security_id", pa.int32(), nullable=False),
    pa.field("api_ticker", pa.string(), nullable=False),
    pa.field("date", pa.date32(), nullable=False),          # ex-dividend date
    pa.field("value", pa.float64()),                        # split-adjusted
    pa.field("unadjusted_value", pa.float64()),
    pa.field("currency", pa.string()),
    pa.field("declaration_date", pa.date32()),
    pa.field("record_date", pa.date32()),
    pa.field("payment_date", pa.date32()),
    pa.field("period", pa.string()),
])

#: Locally computed, point-in-time-safe adjustment factors.
#: adj_close = close * price_factor. Rebuild rather than store adjusted prices.
ADJUSTMENTS = pa.schema([
    pa.field("security_id", pa.int32(), nullable=False),
    pa.field("date", pa.date32(), nullable=False),
    pa.field("split_factor", pa.float64()),   # splits only
    pa.field("price_factor", pa.float64()),   # splits + dividends (total return)
])
