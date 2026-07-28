# XCAP — EODHD archival extraction

> **Design and goals: [docs/DATA_RETRIEVAL.md](docs/DATA_RETRIEVAL.md)** — why the pipeline
> is built this way, every bias control and its evidence, and the operating
> procedure. Live download state: [`data/catalog/PROGRESS.md`](data/catalog/PROGRESS.md).

One-shot extraction of EODHD financial data into analysis-ready parquet, built
on the assumption that **the subscription gets cancelled and there is no second
pull**. Every design decision follows from that.

## Operating principles

1. **Raw bytes are the asset.** Every response is written to `data/_raw`
   (zstd, sha256-checksummed) *before* parsing. Parsers can be rewritten; the
   subscription cannot be un-cancelled.
2. **Never store derived data you can't rebuild.** `adjusted_close` is
   back-adjusted, so a frozen copy silently goes stale. Store raw OHLCV plus
   splits and dividends separately and compute adjustment factors as-of any
   date.
3. **Coverage is a query, not a guess.** The ledger records every HTTP attempt,
   its outcome, and its call cost, so "did we get everything?" has a real answer.
4. **Bias is checked while the subscription is live**, when problems can still
   be re-pulled.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env    # add your EODHD_API_TOKEN
```

## Phase 0 — the security universe

```bash
.venv/bin/python -m xcap.cli phase0-fetch    # ~141 API calls
.venv/bin/python -m xcap.cli phase0-build    # raw -> parquet, no network
.venv/bin/python -m xcap.cli phase0-qa       # bias + integrity gate
.venv/bin/python -m xcap.cli status          # ledger + budget summary
```

Fetches the exchange directory, then **both** the active and the `delisted=1`
ticker list for every exchange. The delisted pass is the single most important
step in the project — it is what separates a survivorship-bias-free dataset from
a worthless one.

Result as of the 2026-07-28 snapshot: **322,006 securities across 70 exchanges,
39.4% delisted globally, 53.3% delisted within US**.

Phase 0 output is frozen. Every later phase iterates `securities.parquet`, so if
the universe shifts mid-extraction, coverage accounting stops meaning anything.

## History depth (measured, not claimed)

```bash
.venv/bin/python -m xcap.cli probe-history --exchange US   # 400 calls
```

Stratified sample of 400 US securities, all of which returned data. Deepest
history observed is **1962-01-02** (long-lived NYSE names: IBM, GE, KO, XOM,
JNJ). But the *median* US common stock starts around **2013**, with a median of
133 months of history.

| stratum | earliest | median first year |
|---|---|---|
| Common Stock / active | 1994 | 2013 |
| Common Stock / delisted | 1993 | 2010 |
| ETF / active | 2000 | 2024 |
| FUND / active | 1986 | 2009 |
| Mutual Fund / active | 2021 | 2025 |

Caveat: this cannot distinguish "the company IPO'd in 2013" from "the vendor
only backfilled to 2013". Separating those requires comparing against known
listing dates.

## Datasets

| File | Rows | Notes |
|---|---|---|
| `data/parquet/exchanges.parquet` | 70 | exchange directory |
| `data/parquet/securities.parquet` | 322,006 | active + delisted, deduplicated |

`securities.parquet` carries a stable `security_id`. **Join on it, never on
ticker** — symbols are recycled after delisting, so a ticker is not an identity.
`source_exchange` is the code used in API paths (`US`); `venue` is the actual
listing venue (`NASDAQ`).

## Call budget

Costs mirror [EODHD's published table](https://eodhd.com/financial-apis/api-limits)
and are charged per HTTP *attempt* (a retry is a request the vendor counts
again). Spend is persisted per GMT day and the run aborts rather than overrun
the configured cap.

| Endpoint | Cost |
|---|---|
| EOD / splits / dividends / symbol lists | 1 |
| Intraday, technicals, news | 5 |
| Fundamentals, options | 10 |
| Bulk EOD (exchange-day), bulk fundamentals (≤500 symbols) | 100 |

Note: full EOD history for one symbol is **1 call**, so per-symbol beats
Bulk-EOD (100 calls per exchange *per day*) by ~10× for historical backfill.
Bulk fundamentals is the inverse: 0.2 calls/symbol vs 10 individually.

## Roadmap

- **Phase 0b** — ID mapping (ISIN/CUSIP/FIGI/CIK); 37.7% of securities currently lack an ISIN
- **Phase 1** — EOD history + splits + dividends, all 322k securities (~450k calls)
- **Phase 2** — fundamentals, bulk for active, `symbols=` batching for delisted
- **Phase 3** — point-in-time index constituents (separate marketplace product)
- **Phase 4** — intraday, scoped to a defined universe
- **Phase 5** — news, insider, macro, options

## Before cancelling

Coverage query clean; adjustment reconstruction reconciles against vendor
`adjusted_close`; every parquet reads back and matches its manifest checksum;
raw archive backed up to a second location. The raw archive is the asset — the
parquet is rebuildable, the raws are not.

Also verify EODHD's terms on retaining and using data after the subscription
ends before relying on this plan.
