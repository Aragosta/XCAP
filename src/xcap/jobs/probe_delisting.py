"""Test whether the delisted archive is deep enough to support a chosen start year.

This is the check that decides whether a backtest window is survivorship-bias
free. Having *a* delisted list is not sufficient: if the vendor only retains
securities that stopped trading after some cutoff, then any window starting
before that cutoff is missing its failures, and the delisted share of the
universe will still look reassuringly large.

Method: sample delisted securities, read the last observed trade date, and
histogram it by year. A genuine US equity archive should show delistings
spread across every year, at a rate of several hundred per year. A cliff in
the histogram is the truncation point, and no start year before it is safe.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter

import duckdb

from ..config import CATALOG_DIR, Config, PARQUET_DIR
from ..eodhd.client import EodhdClient
from ..jobs.probe_history import eod_probe
from ..ledger import Ledger

log = logging.getLogger("xcap.probe_delisting")

TRADABLE_VENUES = ("NYSE", "NASDAQ", "AMEX", "NYSE MKT", "NYSE ARCA", "BATS")


def sample_delisted(types: list[str], n: int, seed: int) -> tuple[list[dict], int]:
    con = duckdb.connect()
    venues = ",".join(f"'{v}'" for v in TRADABLE_VENUES)
    typs = ",".join(f"'{t}'" for t in types)
    rows = con.execute(
        f"""
        SELECT api_ticker, type, venue
        FROM read_parquet('{PARQUET_DIR / "securities.parquet"}')
        WHERE source_exchange='US' AND is_delisted
          AND venue IN ({venues}) AND type IN ({typs})
        ORDER BY api_ticker
        """
    ).fetchall()
    con.close()
    population = len(rows)
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, population))
    return [{"ticker": t, "type": ty, "venue": v} for t, ty, v in picked], population


async def probe_delisting(
    cfg: Config, ledger: Ledger, *,
    types: list[str] | None = None, n: int = 1000, seed: int = 11,
) -> dict:
    types = types or ["Common Stock"]
    sample, population = sample_delisted(types, n, seed)
    log.info("sampling %d of %d delisted securities", len(sample), population)

    async with EodhdClient(cfg, ledger) as client:
        results = await client.fetch_all([eod_probe(s["ticker"]) for s in sample])

    by_ticker = {r.spec.key: r for r in results}
    obs: list[dict] = []
    for s in sample:
        res = by_ticker[s["ticker"]]
        first = last = None
        if res.ok and res.body:
            series = json.loads(res.body)
            if series:
                first, last = series[0].get("date"), series[-1].get("date")
        obs.append({**s, "status": res.status, "first_date": first, "last_date": last})

    with_data = [o for o in obs if o["last_date"]]
    scale = population / len(with_data) if with_data else 0.0

    last_year = Counter(int(o["last_date"][:4]) for o in with_data)
    first_year = Counter(int(o["first_date"][:4]) for o in with_data if o["first_date"])

    years = sorted(last_year)
    histogram = {
        str(y): {
            "sampled": last_year[y],
            "implied_population": round(last_year[y] * scale),
        }
        for y in years
    }

    report = {
        "types": types,
        "population": population,
        "sampled": len(sample),
        "with_data": len(with_data),
        "no_data": len(obs) - len(with_data),
        "scale_factor": round(scale, 2),
        "earliest_delisting": min(years) if years else None,
        "delisting_year_histogram": histogram,
        "first_year_histogram": {str(y): first_year[y] for y in sorted(first_year)},
        "observations": obs,
    }
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "delisting_probe_US.json").write_text(json.dumps(report, indent=2))
    return report


def verdict(report: dict, start_year: int) -> tuple[str, str]:
    """Is `start_year` supportable given the observed delisting distribution?"""
    hist = report["delisting_year_histogram"]
    pre = sum(v["implied_population"] for y, v in hist.items() if int(y) < start_year + 5)
    early = {y: v for y, v in hist.items() if start_year <= int(y) < start_year + 10}
    covered_years = sum(1 for v in early.values() if v["sampled"] > 0)

    if report["earliest_delisting"] is None:
        return "FAIL", "no delisting dates recovered"
    if report["earliest_delisting"] > start_year:
        return "FAIL", (
            f"earliest delisting in the archive is {report['earliest_delisting']}, "
            f"after the requested start year {start_year}"
        )
    if covered_years < 8:
        return "FAIL", (
            f"only {covered_years}/10 years in {start_year}-{start_year+9} contain any "
            "delisting — the archive is truncated inside the requested window"
        )
    return "PASS", (
        f"delistings observed in {covered_years}/10 years from {start_year}; "
        f"~{pre:,} securities delisted before {start_year + 5}"
    )
