"""Budget planner: what remains, what it costs, and what fits in today's cap.

Answers one question precisely -- can the queued work finish inside the
remaining daily allowance -- by counting outstanding requests from the ledger
rather than from estimates. Blocks are listed in execution order with a
running cumulative total, so the point at which the budget would run out is
visible before anything is spent.

Blocks are the unit of completion. A block that starts should finish, because
a half-downloaded dataset is worse than an absent one: it looks complete to
anything that reads the parquet. Where a block is too large to guarantee that,
`split_by_venue` divides the equity work into whole per-venue blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from .config import PARQUET_DIR, load_config
from .eodhd.budget import ENDPOINT_COST, Budget
from .jobs.phase2_reference import (
    INDEX_WHITELIST, MACRO_COUNTRIES, MACRO_INDICATORS,
    NONEQUITY_EXCHANGES, EARNINGS_START,
)
from .ledger import Ledger
from .universe import select

from datetime import date


@dataclass
class Block:
    name: str
    endpoint: str
    total: int
    done: int
    cost_per: int
    note: str = ""
    subblocks: list["Block"] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(self.total - self.done, 0)

    @property
    def calls(self) -> int:
        return self.remaining * self.cost_per


def _done(ledger: Ledger, endpoint: str) -> set[str]:
    """Keys already resolved -- ok, empty or 404 all mean 'do not re-spend'."""
    keys: set[str] = set()
    for status in ("ok", "empty", "not_found"):
        keys |= {r["key"] for r in ledger.rows(endpoint, status=status)}
    return keys


def _nonequity_tickers() -> list[str]:
    con = duckdb.connect()
    marks = ",".join(f"'{e}'" for e in NONEQUITY_EXCHANGES)
    rows = con.execute(
        f"""SELECT api_ticker FROM read_parquet('{PARQUET_DIR / "securities.parquet"}')
            WHERE source_exchange IN ({marks})"""
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def _month_count(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def build_blocks(ledger: Ledger, *, split_by_venue: bool = False) -> list[Block]:
    universe = select()
    blocks: list[Block] = []

    for endpoint, label in (("eod", "Equity EOD"), ("splits", "Equity splits")):
        done = _done(ledger, endpoint)
        b = Block(label, endpoint, len(universe),
                  sum(1 for s in universe if s.api_ticker in done),
                  ENDPOINT_COST.get(endpoint, 1))
        if split_by_venue:
            by_venue: dict[str, list] = {}
            for s in universe:
                by_venue.setdefault(s.venue or "?", []).append(s)
            for venue, secs in sorted(by_venue.items(), key=lambda kv: -len(kv[1])):
                b.subblocks.append(Block(
                    f"{label} · {venue}", endpoint, len(secs),
                    sum(1 for s in secs if s.api_ticker in done),
                    ENDPOINT_COST.get(endpoint, 1),
                ))
        blocks.append(b)

    idx_done = _done(ledger, "fundamentals-index")
    blocks.append(Block("Index constituents", "fundamentals-index",
                        len(INDEX_WHITELIST) + 15, len(idx_done), 10,
                        "whitelist + SP500-/DJ- prefixes; total approximate"))

    con = duckdb.connect()
    n_exch, = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{PARQUET_DIR / 'exchanges.parquet'}')"
    ).fetchone()
    con.close()
    blocks.append(Block("Exchange hours + holidays", "exchange-details",
                        n_exch, len(_done(ledger, "exchange-details")), 1))

    blocks.append(Block("Earnings calendar", "calendar-earnings",
                        _month_count(EARNINGS_START, date.today()),
                        len(_done(ledger, "calendar-earnings")), 1,
                        "monthly chunks"))

    blocks.append(Block("Macro indicators", "macro-indicator",
                        len(MACRO_COUNTRIES) * len(MACRO_INDICATORS),
                        len(_done(ledger, "macro-indicator")), 10,
                        f"{len(MACRO_COUNTRIES)} regions x {len(MACRO_INDICATORS)} series"))

    neq = _nonequity_tickers()
    neq_done = _done(ledger, "eod-nonequity")
    blocks.append(Block("Non-equity EOD", "eod-nonequity", len(neq),
                        sum(1 for t in neq if t in neq_done), 1,
                        "/".join(NONEQUITY_EXCHANGES)))

    return blocks


def report(ledger: Ledger, *, split_by_venue: bool = False) -> str:
    cfg = load_config()
    spent = ledger.spent_today(Budget.gmt_day())
    available = cfg.daily_call_budget - spent
    blocks = build_blocks(ledger, split_by_venue=split_by_venue)

    lines = [
        "",
        f"Budget plan for GMT day {Budget.gmt_day()}",
        f"  daily cap        {cfg.daily_call_budget:>9,}",
        f"  spent so far     {spent:>9,}",
        f"  available        {available:>9,}",
        "",
        f"  {'block':<34}{'remaining':>10}{'cost':>6}{'calls':>10}{'cumulative':>12}  fits",
    ]
    cumulative = 0
    for b in blocks:
        cumulative += b.calls
        fits = "yes" if cumulative <= available else "NO"
        lines.append(f"  {b.name:<34}{b.remaining:>10,}{b.cost_per:>6}"
                     f"{b.calls:>10,}{cumulative:>12,}  {fits}")
        for sb in b.subblocks:
            if sb.remaining:
                lines.append(f"      {sb.name:<30}{sb.remaining:>10,}{sb.cost_per:>6}"
                             f"{sb.calls:>10,}")

    total = sum(b.calls for b in blocks)
    lines += [
        "",
        f"  total queued     {total:>9,}",
        f"  headroom         {available - total:>9,}"
        f"  ({100.0 * (available - total) / max(available, 1):.1f}% of available)",
        "",
    ]
    if total <= available:
        lines.append("  VERDICT: everything queued fits inside today's remaining budget.")
    else:
        over = total - available
        lines.append(f"  VERDICT: over by {over:,} calls. Blocks marked NO will not "
                     "complete today.")
        lines.append("  Run with --split-by-venue and fetch whole venues per day so "
                     "each block lands complete.")
    lines.append("")
    return "\n".join(lines)
