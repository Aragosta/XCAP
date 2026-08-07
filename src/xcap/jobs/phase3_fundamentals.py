"""Phase 3 — equity fundamentals for the US universe.

10 calls per symbol across 32,525 securities is 325,250 calls: more than three
full days at the 100k/day cap, and by far the largest block of work in the
project. `/bulk-fundamentals` would cost 50x less but is 403 on this plan (see
data/catalog/entitlements.json), so per-symbol is the only route.

Two consequences shape this module:

**Blocks are (type, venue), active and delisted together.** A block that is
fetched active-only would be survivorship-biased in exactly the way Phase 0
exists to prevent, and it would look complete to anything reading it. Whole
venues land together or not at all. NASDAQ common stock (14,266 names,
142,660 calls) cannot fit one day at any ordering; it is the single block that
necessarily spans two, and the parquet build gate is what keeps that safe.

**Order is by information value, not convenience.** Common stock first, then
ETFs, then preferred: if the subscription ends early, what is missing should be
the part a factor library can most afford to lose.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..config import Config
from ..eodhd import endpoints
from ..eodhd.budget import ENDPOINT_COST, BudgetExceeded
from ..eodhd.client import EodhdClient
from ..ledger import Ledger
from ..universe import Security, select

log = logging.getLogger("xcap.phase3")

#: Fetch order by security type. Common stock carries the financial statements
#: every factor depends on; preferred lines mostly mirror their issuer.
TYPE_ORDER = ("Common Stock", "ETF", "Preferred Stock")

#: Symbols per request batch. Deliberately small: each response is ~100KB-1MB,
#: and a 100-symbol batch is 1,000 calls, which is a fine enough granularity to
#: stop a run against a spend cap.
CHUNK = 100

COST = ENDPOINT_COST["fundamentals"]


@dataclass(frozen=True)
class Block:
    """A whole (type, venue) slice of the universe — the unit of completion."""

    name: str
    type: str
    venue: str
    securities: list[Security]

    @property
    def calls(self) -> int:
        return len(self.securities) * COST


def blocks(universe: list[Security] | None = None) -> list[Block]:
    """Every fetch block, in execution order."""
    universe = universe if universe is not None else select()
    grouped: dict[tuple[str, str], list[Security]] = {}
    for s in universe:
        grouped.setdefault((s.type, s.venue or "?"), []).append(s)

    ordered = sorted(
        grouped.items(),
        key=lambda kv: (TYPE_ORDER.index(kv[0][0]) if kv[0][0] in TYPE_ORDER
                        else len(TYPE_ORDER), -len(kv[1])),
    )
    return [
        Block(f"{typ} · {venue}", typ, venue, sorted(secs, key=lambda s: s.security_id))
        for (typ, venue), secs in ordered
    ]


def outstanding(ledger: Ledger, block: Block) -> list[Security]:
    """Securities in `block` with no terminal answer yet."""
    done: set[str] = set()
    for status in ("ok", "empty", "not_found"):
        done |= {r["key"] for r in ledger.rows("fundamentals", status=status)}
    return [s for s in block.securities if s.api_ticker not in done]


def _select_blocks(only: list[str] | None) -> list[Block]:
    """Blocks matching `only` (a venue, a type, or a full block name)."""
    plan = blocks()
    if not only:
        return plan
    wanted = {o.lower() for o in only}
    picked = [b for b in plan if b.venue.lower() in wanted
              or b.type.lower() in wanted or b.name.lower() in wanted]
    if not picked:
        raise SystemExit(f"no blocks match {only!r}")
    return picked


def _stratum(s: Security) -> str:
    return f"{s.type}/{'delisted' if s.is_delisted else 'active'}"


async def fetch_fundamentals(
    cfg: Config,
    ledger: Ledger,
    *,
    only: list[str] | None = None,
    max_calls: int | None = None,
    sample: int | None = None,
    seed: int | None = None,
) -> dict:
    """Fetch fundamentals block by block, resuming from the ledger.

    `max_calls` caps what *this run* spends, on top of the daily budget — the
    vendor's own counter runs slightly ahead of the ledger, so leaving headroom
    is how a run ends on a chunk boundary instead of a burst of 402s.

    `sample` takes a stratified sample across type x listing status instead of
    the full universe: the cheap way to measure how much of the universe the
    vendor actually answers for before committing 325,250 calls.
    """
    plan = _select_blocks(only)
    if sample:
        plan = [_sampled(plan, sample, seed or 13)]

    stats = {"ok": 0, "empty": 0, "not_found": 0, "failed": 0, "cached": 0}
    by_stratum: dict[str, dict[str, int]] = {}
    by_block: list[dict] = []
    spent = 0
    stopped = None

    async with EodhdClient(cfg, ledger) as client:
        for block in plan:
            todo = outstanding(ledger, block)
            if not todo:
                log.info("%s — complete (%d symbols)", block.name, len(block.securities))
                by_block.append({"block": block.name, "symbols": len(block.securities),
                                 "fetched": 0, "state": "complete"})
                continue

            log.info("%s — %d/%d outstanding, %d calls",
                     block.name, len(todo), len(block.securities), len(todo) * COST)
            started = time.monotonic()
            fetched = 0

            for i in range(0, len(todo), CHUNK):
                chunk = todo[i:i + CHUNK]
                # Cost the batch actually about to be sent, not a full chunk —
                # otherwise a short final batch is refused by a cap that would
                # have covered it, and the block stops needlessly incomplete.
                if max_calls is not None and spent + len(chunk) * COST > max_calls:
                    stopped = f"run cap reached ({spent:,}/{max_calls:,} calls)"
                    break
                try:
                    results = await client.fetch_all(
                        [endpoints.fundamentals(s.api_ticker) for s in chunk])
                except BudgetExceeded as exc:
                    stopped = str(exc)
                    break

                for sec, r in zip(chunk, results):
                    bucket = r.status if r.status in stats else "failed"
                    stats[bucket] += 1
                    if r.from_cache:
                        stats["cached"] += 1
                    else:
                        spent += COST
                    strat = by_stratum.setdefault(
                        _stratum(sec),
                        {"ok": 0, "empty": 0, "not_found": 0, "failed": 0})
                    strat[bucket if bucket in strat else "failed"] += 1

                fetched += len(chunk)
                rate = fetched / max(time.monotonic() - started, 1e-9)
                log.info("%s  %d/%d  (%.1f/s, ~%.0f min left)  ok=%d empty=%d "
                         "404=%d fail=%d  spent=%d",
                         block.name, fetched, len(todo), rate,
                         (len(todo) - fetched) / max(rate, 1e-9) / 60,
                         stats["ok"], stats["empty"], stats["not_found"],
                         stats["failed"], spent)

            remaining = len(outstanding(ledger, block))
            by_block.append({
                "block": block.name, "symbols": len(block.securities),
                "fetched": fetched, "remaining": remaining,
                "state": "complete" if not remaining else "partial",
            })
            if stopped:
                log.warning("stopping: %s", stopped)
                break

    return {
        "blocks": by_block,
        "totals": stats,
        "by_stratum": by_stratum,
        "calls_spent": spent,
        "stopped": stopped,
    }


def _sampled(plan: list[Block], n: int, seed: int) -> Block:
    """One synthetic block: a stratified sample across type x listing status.

    Sampling the strata evenly rather than proportionally is deliberate — the
    question is whether the vendor answers for *delisted* names at all, and a
    proportional sample of a mostly-active stratum cannot answer it.
    """
    import random

    rng = random.Random(seed)
    strata: dict[str, list[Security]] = {}
    for b in plan:
        for s in b.securities:
            strata.setdefault(_stratum(s), []).append(s)
    per = max(n // max(len(strata), 1), 1)
    picked: list[Security] = []
    for name in sorted(strata):
        pool = strata[name]
        picked += rng.sample(pool, min(per, len(pool)))
    return Block(f"stratified sample (n={len(picked)}, seed={seed})",
                 "sample", "sample", picked)


def report(ledger: Ledger, only: list[str] | None = None) -> str:
    """Per-block download state, for deciding what to run next."""
    lines = ["", "Phase 3 — equity fundamentals by block", "",
             f"  {'block':<28}{'symbols':>9}{'done':>8}{'left':>8}{'calls left':>12}  state"]
    total_left = 0
    for b in _select_blocks(only):
        left = len(outstanding(ledger, b))
        total_left += left * COST
        done = len(b.securities) - left
        state = "complete" if not left else ("not started" if done == 0 else "partial")
        lines.append(f"  {b.name:<28}{len(b.securities):>9,}{done:>8,}{left:>8,}"
                     f"{left * COST:>12,}  {state}")
    lines += ["", f"  calls remaining for a complete Phase 3: {total_left:,}",
              f"  days at 100,000/day: {total_left / 100_000:.1f}", ""]
    return "\n".join(lines)
