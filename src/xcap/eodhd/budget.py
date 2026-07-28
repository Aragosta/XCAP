"""API call-cost accounting and request pacing.

Costs mirror https://eodhd.com/financial-apis/api-limits. They are charged per
HTTP *attempt*, not per logical job, because a retried request is a request the
vendor counts again.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import monotonic

from ..ledger import Ledger

#: Call cost per request, by logical endpoint.
ENDPOINT_COST = {
    "exchanges-list": 1,
    "exchange-symbol-list": 1,
    "eod": 1,
    "splits": 1,
    "dividends": 1,
    "live": 1,
    "intraday": 5,
    "technical": 5,
    "news": 5,
    "fundamentals": 10,
    "options": 10,
    "bulk-eod": 100,
    "bulk-fundamentals": 100,
    # Entitlement probes: cost mirrors the real endpoint each one stands in for.
    "probe-fundamentals": 10, "probe-bulk-fundamentals": 100,
    "probe-fundamentals-index": 10, "probe-fundamentals-etf": 10,
    "probe-historical-market-cap": 10, "probe-insider": 10,
    "probe-macro-indicator": 10, "probe-options": 10,
    "probe-news": 5, "probe-sentiments": 5, "probe-intraday": 5,
    "probe-technical-splitadj": 5,
}


class BudgetExceeded(RuntimeError):
    """Raised when the configured daily call budget would be exceeded."""


class Budget:
    """Tracks call spend against the daily cap, persisted across restarts."""

    def __init__(self, ledger: Ledger, daily_limit: int) -> None:
        self.ledger = ledger
        self.daily_limit = daily_limit
        self._lock = asyncio.Lock()

    @staticmethod
    def gmt_day() -> str:
        # EODHD resets subscription limits at midnight GMT.
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def charge(self, calls: int) -> int:
        """Charge `calls` against today's budget, or raise if it would overrun."""
        async with self._lock:
            day = self.gmt_day()
            spent = self.ledger.spent_today(day)
            if spent + calls > self.daily_limit:
                raise BudgetExceeded(
                    f"daily budget {self.daily_limit} would be exceeded "
                    f"({spent} spent, {calls} requested on {day})"
                )
            return self.ledger.spend(day, calls)

    def remaining(self) -> int:
        return self.daily_limit - self.ledger.spent_today(self.gmt_day())


class RateLimiter:
    """Token bucket over a rolling minute."""

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.refill_per_s = per_minute / 60.0
        self.updated = monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.updated) * self.refill_per_s
                )
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.refill_per_s
            await asyncio.sleep(wait)
