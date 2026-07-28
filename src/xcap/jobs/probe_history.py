"""Measure how far back EODHD history actually goes, on a stratified sample.

The vendor claims "30+ years". That is a claim about their best-covered
securities, not about the universe, and the difference decides how much of a
backtest window is real. This probes a sample cheaply (1 call per symbol, using
period=m to keep payloads small) and reports the distribution of first trade
dates by stratum.

Results land under the 'eod-probe' endpoint so they never collide with the
Phase 1 'eod' archive.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict

import duckdb

from ..config import CATALOG_DIR, Config, PARQUET_DIR
from ..eodhd.client import EodhdClient, RequestSpec
from ..ledger import Ledger

log = logging.getLogger("xcap.probe")

# Strata worth measuring separately: coverage differs enormously between a
# NYSE common stock and an OTC fund share class.
STRATA_TYPES = ["Common Stock", "ETF", "Mutual Fund", "FUND", "Preferred Stock"]


def eod_probe(ticker: str) -> RequestSpec:
    return RequestSpec(
        endpoint="eod-probe",
        path=f"/eod/{ticker}",
        key=ticker,
        params={"period": "m", "order": "a"},
    )


def sample_universe(exchange: str, per_stratum: int, seed: int) -> list[dict]:
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT api_ticker, type, venue, is_delisted
        FROM read_parquet('{PARQUET_DIR / "securities.parquet"}')
        WHERE source_exchange = ?
        """,
        [exchange],
    ).fetchall()
    con.close()

    buckets: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for ticker, typ, venue, delisted in rows:
        if typ in STRATA_TYPES:
            buckets[(typ, bool(delisted))].append(
                {"ticker": ticker, "type": typ, "venue": venue, "is_delisted": bool(delisted)}
            )

    rng = random.Random(seed)
    sample: list[dict] = []
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda r: r["ticker"])
        sample.extend(rng.sample(pool, min(per_stratum, len(pool))))
    return sample


async def probe_history(
    cfg: Config, ledger: Ledger, *, exchange: str = "US",
    per_stratum: int = 40, seed: int = 7,
) -> dict:
    sample = sample_universe(exchange, per_stratum, seed)
    log.info("probing %d securities (%d strata)", len(sample), len(sample) // max(per_stratum, 1))

    async with EodhdClient(cfg, ledger) as client:
        results = await client.fetch_all([eod_probe(s["ticker"]) for s in sample])

    by_ticker = {r.spec.key: r for r in results}
    observations: list[dict] = []
    for s in sample:
        res = by_ticker[s["ticker"]]
        first = last = None
        n = 0
        if res.ok and res.body:
            series = json.loads(res.body)
            if series:
                n = len(series)
                first, last = series[0].get("date"), series[-1].get("date")
        observations.append({**s, "status": res.status, "first_date": first,
                             "last_date": last, "months": n})

    with_data = [o for o in observations if o["first_date"]]
    report = {
        "exchange": exchange,
        "sampled": len(observations),
        "with_data": len(with_data),
        "no_data": len(observations) - len(with_data),
        "by_stratum": {},
        "first_year_histogram": {},
        "observations": observations,
    }

    strata: dict[str, list[dict]] = defaultdict(list)
    for o in with_data:
        strata[f"{o['type']} / {'delisted' if o['is_delisted'] else 'active'}"].append(o)

    for name, obs in sorted(strata.items()):
        years = sorted(int(o["first_date"][:4]) for o in obs)
        report["by_stratum"][name] = {
            "n": len(obs),
            "earliest": min(years),
            "p25_first_year": years[len(years) // 4],
            "median_first_year": years[len(years) // 2],
            "median_months": sorted(o["months"] for o in obs)[len(obs) // 2],
        }

    decades = Counter(f"{int(o['first_date'][:4]) // 10 * 10}s" for o in with_data)
    report["first_year_histogram"] = dict(sorted(decades.items()))

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / f"history_probe_{exchange}.json").write_text(json.dumps(report, indent=2))
    return report
