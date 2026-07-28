# XCAP data retrieval — process and goals

Status document for the EODHD extraction. Describes *why* the pipeline is built
the way it is and *how* to operate it. Live counts are not duplicated here —
they live in [`data/catalog/PROGRESS.md`](../data/catalog/PROGRESS.md), which is
regenerated from disk by `python -m xcap.cli coverage`.

---

## 1. Goal and the constraint that shapes everything

Acquire a complete, bias-free historical dataset for algorithmic trading
research from a single EODHD **All-In-One** subscription, then **cancel the
subscription**.

That last clause is not a footnote — it inverts every normal data-engineering
tradeoff:

| Normal pipeline | This pipeline |
|---|---|
| Re-fetch when something looks wrong | There is no re-fetch. Ever. |
| Store the convenient derived form | Store the source form; derive on demand |
| Fix data quality later | Find data quality problems *before* cancelling |
| Partial coverage is fine, it backfills tomorrow | Partial coverage is permanent |

Everything below follows from this. The subscription is a time-boxed window
onto a vendor's database, and the job is to get everything of value through
that window before it closes, in a form that stays correct afterwards.

### Success criteria

1. **Complete** — every security in the defined universe, delisted included,
   with an explicit recorded reason for anything absent.
2. **Bias-free** — no survivorship, look-ahead, or corporate-action leakage.
   Where a bias cannot be eliminated, it is measured and documented.
3. **Durable** — correct in five years without vendor access.
4. **Few parquet files** — a handful of datasets, not thousands of fragments.
5. **Auditable** — provenance from any parquet row back to the exact bytes the
   vendor returned.

---

## 2. Two-layer architecture

```
                  EODHD API
                      │
                      ▼
   ┌──────────────────────────────────────────┐
   │  LAYER 1  data/_raw/                     │   ← the asset. Irreplaceable.
   │  exact vendor bytes, zstd, sha256        │
   │  written BEFORE anything parses them     │
   └──────────────────────────────────────────┘
                      │  offline, no network, repeatable
                      ▼
   ┌──────────────────────────────────────────┐
   │  LAYER 2  data/parquet/                  │   ← derived. Disposable.
   │  typed, sorted, partitioned, analysis-   │
   │  ready. Rebuilt from Layer 1 on demand   │
   └──────────────────────────────────────────┘
```

### Layer 1 — raw archive

Every HTTP response is written to `data/_raw/{endpoint}/{key}.{hash}.json.zst`
and checksummed **before** any parser sees it. Writes are atomic (temp file then
rename), so a raw file either exists complete or does not exist.

This exists because parsers have bugs. One was already found and fixed here: the
dividend adjustment used the vendor's split-adjusted `value` against a raw
close, mixing share terms. The fix cost a rebuild, not a re-purchase. Without
Layer 1 it would have cost the dataset.

### Layer 2 — parquet

Rebuilt at any time, offline:

| command | produces |
|---|---|
| `phase0-build` | `exchanges.parquet`, `securities.parquet` |
| `phase1-build` | `eod/`, `splits.parquet`, `dividends.parquet` |
| `phase1-adjust` | `adjustments/` |

Deleting `data/parquet/` loses nothing but CPU time.

### Layer boundary rule

**Nothing derived is ever stored where the source could be.** The critical case
is `adjusted_close`. A back-adjusted price series is only valid as of the moment
it was downloaded — the next split or dividend rewrites the entire history.
AAPL's first bar illustrates it:

```
1980-12-12   close 28.7392   adjusted_close 0.0982
```

That `0.0982` is a snapshot, not a fact. Frozen on disk and the subscription
cancelled, it silently becomes wrong with no way to notice.

So the parquet stores **raw OHLCV only**. `vendor_adjusted_close` is retained
solely for reconciliation. Adjustment factors are computed locally in
`xcap/corpactions.py` from raw prices plus the split and dividend tables, which
makes adjusted prices reconstructable forever and as-of any date.

---

## 3. Bias controls

The reason for the whole exercise. Each control is implemented and, where
possible, measured.

### 3.1 Survivorship bias — controlled

Every exchange is enumerated twice: active tickers, and `delisted=1`. Both are
merged into `securities.parquet` with `is_delisted` derived from which list a
security appeared in.

Measured: **53.3% of US securities are delisted** (58,717 of 110,209). A vendor
pull without the delisted pass would silently drop every company that failed.

### 3.2 Archive truncation — measured, and it set the start date

Having a delisted list is *not sufficient*. If the vendor only retains
securities that died after some cutoff, the delisted share still looks healthy
while every earlier failure is missing.

`probe-delisting` sampled 1,000 delisted US common stocks and read their last
trade date:

```
earliest delisting anywhere in the archive: 1998
0 of 1000 sampled delistings before 1998
224 of 1000 have their first price observation in 1997  ← backfill wall
```

With n=1000 and zero observations, the 95% upper bound on the true pre-1998
rate is 0.3% — at most ~49 of 16,460 securities, against a reality of several
thousand US delistings in 1995–1997. That is a hard archive cutoff, not noise.

**Consequence: the dataset floor is 2000-01-01.** A 1995 start was requested and
rejected on this evidence — it would have contained the survivors of 1995–1997
but almost none of the failures, inflating returns in exactly the period used to
validate a strategy. 1998 is the boundary year and visibly partial; 2000 clears
it with margin and fully contains the dot-com bust.

Full history is still archived raw, because a date-limited EOD call costs the
same single API call. The floor is applied at build time only, so the decision
stays reversible.

### 3.3 Look-ahead bias — controlled by construction

- Fundamentals are indexed on `filing_date`, never `period_end`. A Q4 figure
  with period end 31 Dec was not knowable until it was filed in February.
- Universe membership is never decided from data not yet downloaded. Phase 1
  fetches all 32,525 securities regardless of whether they traded after 2000 —
  filtering at fetch time would rebuild the survivorship bias just removed.

### 3.4 Corporate-action leakage — controlled and reconciled

Adjustment factors are computed locally, CRSP/Yahoo convention, in log space via
window functions:

```
split ratio r  →  f(t) = 1 / r
cash dividend  →  f(t) = 1 − d / close(previous trading day)

factor(u) = Π f(t) for all events with ex-date t > u        (latest bar = 1.0)
adjusted_close(u) = close(u) × price_factor(u)
```

`phase1-reconcile` then compares the rebuild against the vendor's
`adjusted_close`. Disagreement means one of the two is wrong, and finding out
which is only possible while the subscription is live.

### 3.5 Ticker recycling — detected

Ticker symbols are reused after delisting, so **a ticker is not an identity**.
All datasets key on a stable internal `security_id`; joining on ticker is a bug.

Detector (Phase 1 QA): corporate actions dated outside a security's own EOD
range mean the price series and the action series describe different companies.
Real example — `PDII.US` has price history ending 2015-12-22 but splits dated
2016 and 2020.

### 3.6 Restatement / point-in-time fundamentals — **not solvable**

EODHD serves *current* fundamentals, not filing vintages. A 2015 figure later
restated shows the restated number. There is no fix within this vendor. Recorded
as a known dataset property so no strategy unknowingly depends on it.

### 3.7 Index membership before 2008 — **biased, documented**

`/fundamentals/{INDEX}.INDX` returns `HistoricalTickerComponents` with real
membership windows, and needs no separate marketplace subscription. But every
recorded `EndDate` is 2008-09-16 or later: 315 removals where 2000–2026 should
show ~570. **Pre-2008 removals are absent.**

Use index membership from 2008. For 2000–2008, build the universe from market
cap and liquidity instead.

### 3.8 Delisting returns — flagged, not solved

Delisted series simply stop. A backtest liquidating at the last observed price
overstates returns. Terminal observations are identifiable so the strategy layer
can apply an explicit delisting-return assumption.

---

## 4. What the vendor actually provides

Measured with `probe-entitlements` against the real token — 17 of 20 endpoints
accessible. Two findings drive the plan:

- **`bulk-fundamentals` returns 403.** It requires the Extended Fundamentals
  plan. Fundamentals therefore cost **10 calls/symbol, not 0.2** — a 50× swing
  that sets the whole Phase 2 schedule.
- **Index constituents are reachable via the ordinary fundamentals endpoint**,
  no marketplace subscription needed (subject to §3.7).

### Cost model

Charged per HTTP *attempt* — a retry is a request the vendor counts again.

| endpoint | calls |
|---|---:|
| EOD / splits / dividends / symbol lists / exchange details / calendar | 1 |
| intraday, technicals, news | 5 |
| fundamentals, options, macro indicators, market cap, insider | 10 |
| bulk EOD (one exchange-day), bulk fundamentals (≤500 symbols) | 100 |

Caps: **100,000 calls/day** (raisable), **1,000 requests/min** (we run at 500
after observing 429s at 800).

Non-obvious consequence: one EOD call returns a security's *entire* history, so
per-symbol backfill beats Bulk-EOD — which costs 100 calls per exchange **per
day** — by roughly 10×.

### Measured history depth

Deepest is **1962-01-02** (long-lived NYSE names: IBM, GE, KO, XOM, JNJ). But
the *median* US common stock starts around **2013** with 133 months of history.
A 1962-start backtest exists only for a few hundred survivors, which is itself a
survivorship-biased sample.

---

## 5. Universe

**32,525 US securities** — 12,556 active, 19,969 delisted.

- Venues: NYSE, NASDAQ, AMEX, NYSE MKT, NYSE ARCA, BATS
- Types: Common Stock, ETF, Preferred Stock

Deliberately excludes OTC tiers (PINK, OTCQB, OTCGREY…) and NMFQS mutual-fund
quote lines. Those are two thirds of the 110,209 raw US symbol count but are not
a tradable equity universe. Defined once in `xcap/universe.py` so the fetch,
build and QA layers cannot disagree about what "the universe" means.

---

## 6. Phases

| phase | dataset | calls | status |
|---|---|---:|---|
| 0 | Exchange directory + active/delisted ticker lists | 141 | done |
| 0 | History-depth and delisting-coverage probes | ~1,400 | done |
| 1 | Equity EOD, full history | 32,525 | done |
| 1 | Splits | 32,525 | running |
| 1 | Dividends | 32,525 | deferred |
| 2 | Index constituents (point-in-time) | ~430 | queued |
| 2 | Exchange trading hours + holidays | 70 | queued |
| 2 | Historical earnings calendar | ~320 | queued |
| 2 | Macro indicators, 33 regions × 39 series | 12,870 | queued |
| 2 | Forex / crypto / bond / money EOD | 9,819 | queued |
| 3 | Equity fundamentals | 325,250 | not started |

Deliberately **out of scope** (cost far exceeds value at 10 calls/symbol or
unbounded volume): intraday (~650k calls), insider Form 4 (~147k, only 4,800
issuers covered), news (unbounded), historical market cap (US-only, weekly, from
2021 — derive from shares outstanding × price instead), options (path 404s;
likely a separate subscription).

### Block completion

**A block that starts should finish.** A half-downloaded dataset is worse than an
absent one because it looks complete to whatever reads the parquet — a factor
return computed on a universe that silently stops at the letter M.

`xcap plan` costs remaining work from the ledger against the remaining daily
allowance and shows a running cumulative, so budget exhaustion is visible before
anything is spent. `plan --split-by-venue` divides equity work into whole venue
blocks (NASDAQ, NYSE, ARCA, BATS, NYSE MKT, AMEX). Phase 3 fundamentals
*requires* this mode: at 325,250 calls it spans ~4 days, so each day must end on
complete venues.

---

## 7. Operating the pipeline

```bash
python -m xcap.cli plan [--split-by-venue]   # cost remaining work vs today's budget
python -m xcap.cli coverage                  # write PROGRESS.md
python -m xcap.cli status                    # ledger + budget summary

python -m xcap.cli phase0-fetch   /  phase0-build  /  phase0-qa
python -m xcap.cli phase1-fetch   /  phase1-build  /  phase1-qa
python -m xcap.cli phase1-adjust  /  phase1-reconcile
python -m xcap.cli phase2-fetch --which indices exchange-details earnings macro noneq-eod

python -m xcap.cli probe-history      # measured EOD depth by stratum
python -m xcap.cli probe-delisting    # test a start year for survivorship safety
```

### Resumption

Every HTTP attempt is recorded in `data/catalog/ledger.sqlite`, keyed
`(endpoint, key, params_hash)` with status, http code, attempts, call cost,
bytes, sha256 and raw path.

Re-running any fetch skips what is already resolved and **spends no calls on
it**. `ok`, `empty` and `not_found` are all terminal — an empty splits response
is a real answer, not a failure, and is never re-fetched. Interrupted runs cost
nothing to resume; the budget guard stops cleanly at the daily cap rather than
dying mid-flight.

### Parquet layout

Few datasets, sorted for the dominant access pattern, zstd-compressed, explicit
schemas with **zero type inference** — inference across heterogeneous rows
silently yields different column types on different runs.

| dataset | partitioning | sort |
|---|---|---|
| `eod/` | `year=` | `date, security_id` (cross-sectional reads) |
| `adjustments/` | `year=` | `date, security_id` |
| `splits.parquet`, `dividends.parquet` | none | `security_id, date` |
| `securities.parquet`, `exchanges.parquet` | none | `security_id` |

---

## 8. Pre-cancellation gate

Do not cancel until **all** of these hold:

1. `coverage` shows every universe security either has data or a recorded
   reason for not having it.
2. `phase1-qa` passes; every WARN is explained rather than merely tolerated.
3. `phase1-reconcile` — disagreements against vendor `adjusted_close` are
   triaged into the three known causes (spliced ticker, vendor error, missing
   ETF distributions), not left unexplained.
4. Every parquet reads back end-to-end and matches its manifest checksum.
5. **`data/_raw/` is backed up to a second location** — external disk or S3.
   It is gitignored and currently exists on one machine. That ~1.5 GB *is* the
   purchase; the parquet is rebuildable, the raws are not.
6. Vendor terms on retaining and using data after the subscription ends have
   been read and confirmed for this plan.

---

## 9. Known limitations

Carried deliberately; each is measured or documented rather than hidden.

| limitation | impact | mitigation |
|---|---|---|
| Delisted archive starts ~1997–98 | No survivorship-safe window before 2000 | Floor at 2000-01-01 |
| No point-in-time fundamentals | Restatements invisible | Documented; no vendor-side fix |
| Index removals absent before 2008 | Index universe biased 2000–2008 | Use membership from 2008; build own universe earlier |
| ETF distributions not exposed via `/div` | ETF total return not fully rebuildable | Source distributions elsewhere, or scope ETFs out |
| Vendor `adjusted_close` wrong for some securities | — | Detected by reconciliation; local rebuild is authoritative |
| Ticker recycling in vendor data | Price and action series can describe different firms | Detector in Phase 1 QA; key on `security_id` |
| `bulk-fundamentals` not entitled | Fundamentals cost 50× more | Per-venue blocks across ~4 days |
