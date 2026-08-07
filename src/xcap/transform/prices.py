"""Raw price/corporate-action JSON -> partitioned parquet.

EOD is far too large to hold in memory, so it streams into staging files and
DuckDB performs the out-of-core sort and partitioning. Dates stay as strings
through staging (cheap) and are cast once at finalise time.

Reads only from the raw archive. Safe to re-run after the subscription ends.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CATALOG_DIR, DATA_DIR, PARQUET_DIR
from ..eodhd.client import EodhdClient
from ..db import connect, query
from ..ledger import Ledger
from ..schemas.prices import DIVIDENDS, SPLITS
from ..universe import START_DATE, select
from . import ROWS_PER_STAGE_FILE, sha256_file

log = logging.getLogger("xcap.transform.prices")

STAGING = DATA_DIR / "_staging"

STAGE_EOD = pa.schema([
    pa.field("security_id", pa.int32()),
    pa.field("api_ticker", pa.string()),
    pa.field("date", pa.string()),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("vendor_adjusted_close", pa.float64()),
    pa.field("volume", pa.float64()),   # vendor sometimes emits floats
])


#: An isolated print is a whole OHLC bar belonging to a different instrument, spliced
#: into a series and unwinding on the next bar. Every other check passes it: the bar is
#: internally consistent, split_factor is 1.0, and a plain jump filter cannot tell it
#: from a real move. The reversal is the tell. Measured on the VENDOR ADJUSTED series so
#: a genuine split, already absorbed there, never trips it. Same rule as the
#: "isolated price prints" check in xcap.qa.phase1_checks, which counts what this drops.
_SPIKE_LN = 0.80        # >2.2x move
_UNWIND_FRAC = 0.15     # ...that retraces to within 15% of where it started


def _neutralize_isolated_prints(series: list[dict]) -> int:
    """NaN the OHLCV of bars that spike and immediately unwind. Returns count.

    The bar is blanked rather than deleted so the trading date survives; downstream a
    null close reads as "no print", which is what a bad tick actually is.

    "Next bar" means the next bar that PRINTS, matching the QA check: a bad tick
    sitting next to a non-printing bar is still a bad tick.

    The rule is symmetric -- it sees "middle disagrees with both ends, ends agree" and
    cannot say which of the three is the liar. Around two spikes the GOOD bar between
    them matches that shape too, so within a pass candidates are taken largest jump
    first and never two in a row: that blanks the spikes and spares what sits between.

    Blanking then changes who is adjacent to whom, and a bad print hiding behind a
    worse one only becomes visible once the worse one is gone (AEZ.US: 20.76, 0.035,
    8.6, 23.40 -- both middle bars are junk, but only the 0.035 is visible at first).
    So repeat until a pass finds nothing.
    """
    adj = [_as_float(b.get("adjusted_close")) for b in series]
    live = [i for i, p in enumerate(adj) if p is not None and p > 0]
    total = 0

    while True:
        candidates = []
        for j in range(1, len(live) - 1):
            prev, p, nxt = adj[live[j - 1]], adj[live[j]], adj[live[j + 1]]
            jump = abs(math.log(p / prev))
            if jump > _SPIKE_LN and abs(math.log(nxt / prev)) < _UNWIND_FRAC * jump:
                candidates.append((jump, j))
        if not candidates:
            return total

        taken: set[int] = set()
        for _, j in sorted(candidates, reverse=True):
            if j - 1 not in taken and j + 1 not in taken:
                taken.add(j)

        for j in taken:
            for k in ("open", "high", "low", "close", "adjusted_close", "volume"):
                series[live[j]][k] = None
        live = [i for j, i in enumerate(live) if j not in taken]
        total += len(taken)


#: A spliced ticker carries bars from more than one company: the exchange reissued a
#: dead symbol, or the vendor folded two listings onto one key. PVX.US interleaves
#: ~4000, ~9.15 and ~0.09 regimes; every bar is a real price for SOME instrument, so
#: nothing here is a bad print -- the series is only wrong as an assembly. It cannot be
#: cleaned bar by bar, and pct_change across a regime boundary invents a 47,000x return.
#:
#: The series is cut at the first regime change and everything after is discarded. Cut
#: FORWARD, never retroactively: a splice in 2020 says nothing about whether the name
#: was tradeable in 2015, and dropping its whole history would be deciding 2015
#: universe membership on 2020 information -- lookahead, worth +0.24%/yr on the liquid
#: US panel. Keeping the pre-splice segment also keeps the older company, which is real.
#:
#: The residual bias is survivorship: 94% of spliced names are delisted, because
#: recycling happens when a company dies. Bounded at +0.46%/yr equal-weighted, but it
#: scales with concentration -- re-measure before trusting it on a concentrated book.
#: Measured on the vendor adjusted series so splits, already absorbed, never trip it.
_SPLICE_LN = 2.0        # >7.4x in one session and it does not come back
_SPLIT_GUARD_DAYS = 5   # a step this close to a split is the split, not a splice


def _split_windows() -> dict[str, set[str]]:
    """api_ticker -> ISO dates within _SPLIT_GUARD_DAYS of one of its splits.

    The vendor's adjusted_close normally absorbs splits, so most never reach the
    splice rule. Where its adjustment is incomplete a 1:10 reverse split looks exactly
    like a regime change, and reverse splits cluster in distressed names -- precisely
    where throwing away the rest of the history would bias hardest. Splits are built
    before EOD, so the dates are on hand.
    """
    path = PARQUET_DIR / "splits.parquet"
    if not path.exists():
        return {}
    t = pq.read_table(path, columns=["api_ticker", "date"])
    out: dict[str, set[str]] = {}
    offsets = range(-_SPLIT_GUARD_DAYS, _SPLIT_GUARD_DAYS + 1)
    for tk, d in zip(t["api_ticker"].to_pylist(), t["date"].to_pylist()):
        if d is not None:
            out.setdefault(tk, set()).update((d + timedelta(days=k)).isoformat()
                                             for k in offsets)
    return out


def _splice_cut(series: list[dict], split_days: set[str] = frozenset()) -> int:
    """Index of the first bar of a foreign regime, or len(series) if the ticker is clean.

    Run AFTER _neutralize_isolated_prints, which removes the spikes that do come back;
    what is left at this size is a step to a different instrument's price level.
    """
    adj = [_as_float(b.get("adjusted_close")) for b in series]
    live = [i for i, p in enumerate(adj) if p is not None and p > 0]
    for j in range(1, len(live)):
        i = live[j]
        if (abs(math.log(adj[i] / adj[live[j - 1]])) > _SPLICE_LN
                and series[i].get("date") not in split_days):
            return i
    return len(series)


def _sanitize(series: list[dict]) -> tuple[int, int]:
    """Blank unusable bars and repair impossible ones. Returns (blanked, clamped).

    A zero in OHLC is the vendor's "no print", not a price, and 15,581 of them sit
    INSIDE a series where a return taken off close goes to -100% and then to infinity.
    Blanked to null rather than deleted: same shape _neutralize_isolated_prints leaves
    behind, so the trading date survives and everything downstream already reads it as
    "did not print".

    A high/low that fails to bracket open/close cannot be right, but nothing here can
    say which of the four is the liar. Widening the range to the bar's own extremes is
    the only repair that invents no number the session did not print.
    """
    blanked = clamped = 0
    for bar in series:
        o, h, l, c = (_as_float(bar.get(k)) for k in ("open", "high", "low", "close"))
        if any(p is not None and p <= 0 for p in (o, h, l, c)):
            for k in ("open", "high", "low", "close", "adjusted_close", "volume"):
                bar[k] = None
            blanked += 1
            continue
        live = [p for p in (o, h, l, c) if p is not None]
        if live and (h, l) != (max(live), min(live)):
            bar["high"], bar["low"] = max(live), min(live)
            clamped += 1
    return blanked, clamped


#: A hole this long is not a trading halt, it is a second listing on the same symbol.
#: The price rule below cannot see it: INTW.US carries a dead 2000 company at ~$10 and
#: the 2024 GraniteShares 2x Long INTC ETF at ~$12, so there is no step to detect, only
#: an 8,687-day hole between them.
_MAX_GAP_DAYS = 365
#: A segment this short beside such a hole is vendor debris, not a listing: SVRA.US
#: prints once on 2000-07-26 and then not again until 2002-06-11.
_STRAY_BARS = 5


def _listing_span(series: list[dict]) -> tuple[int, int]:
    """Half-open slice of `series` covering one listing.

    Stray segments are dropped wherever they sit -- each one hands a backtest a
    security that existed for a single day years before it was born. Of the segments
    left, the EARLIEST is kept: the same forward-only cut as _splice_cut and for the
    same reason, that dropping the early history would decide who was tradeable then
    on what the ticker went on to do later.

    When every segment is short the longest one is kept instead of the first, so the
    result never still contains a hole. A genuinely short history is not debris and is
    never emptied -- one uninterrupted run of two bars is one segment and survives
    whole -- but a 7-bar ticker split either side of a 13-year hole is still two
    listings, and only one of them can be it.
    """
    bounds = [0]
    prev = None
    for i, bar in enumerate(series):
        d = _opt_date(bar.get("date"))
        if d is None:
            continue
        if prev is not None and (d - prev).days > _MAX_GAP_DAYS:
            bounds.append(i)
        prev = d
    bounds.append(len(series))

    segments = list(zip(bounds, bounds[1:]))
    for lo, hi in segments:
        if hi - lo > _STRAY_BARS:
            return lo, hi
    return max(segments, key=lambda s: (s[1] - s[0], -s[0]))


def _ticker_to_id() -> dict[str, int]:
    rows = query(
        f"SELECT api_ticker, security_id FROM read_parquet('{PARQUET_DIR / 'securities.parquet'}')"
    )
    return {t: int(i) for t, i in rows}


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- EOD

def build_eod(ledger: Ledger, start_date: str = START_DATE) -> dict:
    ids = _ticker_to_id()
    split_days = _split_windows()
    stage_dir = STAGING / "eod"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    cols: dict[str, list] = {f.name: [] for f in STAGE_EOD}
    part = 0
    total_rows = 0
    dropped_pre_start = 0
    neutralized = 0
    blanked = 0
    clamped = 0
    regapped = 0
    dropped_other_listing = 0
    spliced_securities = 0
    dropped_after_splice = 0
    securities = 0

    def flush() -> None:
        nonlocal cols, part
        if not cols["date"]:
            return
        pq.write_table(
            pa.table(cols, schema=STAGE_EOD),
            stage_dir / f"part-{part:05d}.parquet",
            compression="zstd", compression_level=3,
        )
        part += 1
        cols = {f.name: [] for f in STAGE_EOD}

    rows = ledger.rows("eod", status="ok")
    log.info("parsing %d EOD series (floor %s)", len(rows), start_date)

    for n, row in enumerate(rows, 1):
        ticker = row["key"]
        sec_id = ids.get(ticker)
        if sec_id is None:
            log.warning("ticker %s not in securities.parquet; skipping", ticker)
            continue
        try:
            series = json.loads(EodhdClient.read_raw(Path(row["raw_path"])))
        except Exception as exc:  # noqa: BLE001 - record and continue, never abort the build
            log.error("unreadable raw for %s: %s", ticker, exc)
            continue

        # Apply the floor FIRST. Every rule below decides what one ticker's history
        # is, and letting them see bars the dataset does not keep makes them answer
        # for the wrong window: a symbol used 1995-1998 and reissued in 2005 has its
        # earliest listing entirely below the floor, so preferring it discards the
        # 2005 history and leaves the security with no bars at all.
        n_bars = len(series)
        series = [b for b in series if b.get("date") and b["date"] >= start_date]
        dropped_pre_start += n_bars - len(series)

        b, c = _sanitize(series)
        blanked += b
        clamped += c

        lo, hi = _listing_span(series)
        if (lo, hi) != (0, len(series)):
            regapped += 1
            dropped_other_listing += len(series) - (hi - lo)
            series = series[lo:hi]

        neutralized += _neutralize_isolated_prints(series)

        cut = _splice_cut(series, split_days.get(ticker, frozenset()))
        if cut < len(series):
            spliced_securities += 1
            dropped_after_splice += len(series) - cut
            series = series[:cut]

        for bar in series:
            cols["security_id"].append(sec_id)
            cols["api_ticker"].append(ticker)
            cols["date"].append(bar["date"])
            cols["open"].append(_as_float(bar.get("open")))
            cols["high"].append(_as_float(bar.get("high")))
            cols["low"].append(_as_float(bar.get("low")))
            cols["close"].append(_as_float(bar.get("close")))
            cols["vendor_adjusted_close"].append(_as_float(bar.get("adjusted_close")))
            cols["volume"].append(_as_float(bar.get("volume")))

        total_rows += len(series)
        if series:
            securities += 1
        if len(cols["date"]) >= ROWS_PER_STAGE_FILE:
            flush()
        if n % 5000 == 0:
            log.info("  %d/%d series, %d rows staged", n, len(rows), total_rows)

    flush()
    log.info("staged %d rows from %d securities (%d bars dropped before %s, "
             "%d non-positive bars blanked, %d OHLC ranges clamped, "
             "%d bars dropped as a second listing in %d securities, "
             "%d isolated prints neutralized, %d bars dropped after a splice in "
             "%d securities)",
             total_rows, securities, dropped_pre_start, start_date, blanked, clamped,
             dropped_other_listing, regapped, neutralized,
             dropped_after_splice, spliced_securities)

    out = PARQUET_DIR / "eod"
    if out.exists():
        shutil.rmtree(out)

    con = connect()
    con.execute(f"""
        COPY (
            SELECT security_id, api_ticker, CAST(date AS DATE) AS date,
                   open, high, low, close, vendor_adjusted_close,
                   CAST(volume AS BIGINT) AS volume,
                   CAST(substr(date, 1, 4) AS INTEGER) AS year
            FROM read_parquet('{stage_dir}/*.parquet')
            ORDER BY date, security_id
        ) TO '{out}'
        (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION zstd, OVERWRITE_OR_IGNORE)
    """)
    con.close()
    shutil.rmtree(stage_dir, ignore_errors=True)

    files = sorted(out.rglob("*.parquet"))
    return {
        "path": str(out.relative_to(DATA_DIR)),
        "rows": total_rows,
        "securities": securities,
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "dropped_before_start": dropped_pre_start,
        "blanked_non_positive": blanked,
        "clamped_ohlc": clamped,
        "second_listing_securities": regapped,
        "dropped_second_listing": dropped_other_listing,
        "neutralized_isolated_prints": neutralized,
        "spliced_securities_truncated": spliced_securities,
        "dropped_after_splice": dropped_after_splice,
        "start_date": start_date,
    }


# ------------------------------------------------- splits & dividends

def _parse_split(raw: str) -> tuple[float | None, float | None, float | None]:
    """'2.000000/1.000000' -> (to, from, ratio)."""
    if not raw or "/" not in raw:
        return None, None, None
    to_s, _, from_s = raw.partition("/")
    to_v, from_v = _as_float(to_s), _as_float(from_s)
    if not to_v or not from_v:
        return to_v, from_v, None
    return to_v, from_v, to_v / from_v


def build_splits(ledger: Ledger) -> dict:
    ids = _ticker_to_id()
    cols: dict[str, list] = {f.name: [] for f in SPLITS}
    for row in ledger.rows("splits", status="ok"):
        ticker = row["key"]
        sec_id = ids.get(ticker)
        if sec_id is None:
            continue
        for ev in json.loads(EodhdClient.read_raw(Path(row["raw_path"]))):
            raw = ev.get("split")
            to_v, from_v, ratio = _parse_split(raw)
            cols["security_id"].append(sec_id)
            cols["api_ticker"].append(ticker)
            cols["date"].append(date.fromisoformat(ev["date"]))
            cols["split_to"].append(to_v)
            cols["split_from"].append(from_v)
            cols["ratio"].append(ratio)
            cols["raw"].append(raw)
    table = pa.table(cols, schema=SPLITS).sort_by([("security_id", "ascending"),
                                                   ("date", "ascending")])
    path = PARQUET_DIR / "splits.parquet"
    pq.write_table(table, path, compression="zstd", compression_level=9)
    return {"path": str(path.relative_to(DATA_DIR)), "rows": table.num_rows,
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _opt_date(value: object) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def build_dividends(ledger: Ledger) -> dict:
    ids = _ticker_to_id()
    cols: dict[str, list] = {f.name: [] for f in DIVIDENDS}
    for row in ledger.rows("dividends", status="ok"):
        ticker = row["key"]
        sec_id = ids.get(ticker)
        if sec_id is None:
            continue
        for ev in json.loads(EodhdClient.read_raw(Path(row["raw_path"]))):
            d = _opt_date(ev.get("date"))
            if d is None:
                continue
            cols["security_id"].append(sec_id)
            cols["api_ticker"].append(ticker)
            cols["date"].append(d)
            cols["value"].append(_as_float(ev.get("value")))
            cols["unadjusted_value"].append(_as_float(ev.get("unadjustedValue")))
            cols["currency"].append(ev.get("currency"))
            cols["declaration_date"].append(_opt_date(ev.get("declarationDate")))
            cols["record_date"].append(_opt_date(ev.get("recordDate")))
            cols["payment_date"].append(_opt_date(ev.get("paymentDate")))
            cols["period"].append(ev.get("period"))
    table = pa.table(cols, schema=DIVIDENDS).sort_by([("security_id", "ascending"),
                                                      ("date", "ascending")])
    path = PARQUET_DIR / "dividends.parquet"
    pq.write_table(table, path, compression="zstd", compression_level=9)
    return {"path": str(path.relative_to(DATA_DIR)), "rows": table.num_rows,
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def trim_events_to_listing(name: str) -> int:
    """Drop events dated outside a security's own listing. Returns rows removed.

    A ticker's event history covers every company that ever used the symbol, so once
    the price series has been cut to a single listing the rest of that history belongs
    to somebody else. CRY.US is the GraniteShares YieldBOOST CRCL ETF, first bar
    2026-04-28, and it carried a dividend dated 2000-12-28 -- CryoLife's. 22,014
    dividends and 2,368 splits sat under a security_id whose company had stopped
    trading years earlier. None of them reached the adjustment factors, because there
    is no bar on those dates for them to apply to, but they are another company's
    payments filed under this one and an event study reads them.

    Both edges of the DATASET are exempt, because at both the price history is ours
    and not the security's. A security still trading in the last session keeps its
    scheduled future events -- nothing has replaced it yet. A security whose first bar
    sits at the dataset floor keeps everything before it: its real history runs back
    past the floor and we simply do not store that part. Only where the price series
    starts well inside the window is its own start known, and only there does an
    earlier event have to belong to someone else.
    """
    path = PARQUET_DIR / f"{name}.parquet"
    if not path.exists():
        return 0

    eod = PARQUET_DIR / "eod"
    tmp = path.with_suffix(f".{name}.tmp")
    con = connect()
    before, = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()
    con.execute(f"""
        COPY (
            WITH span AS (
                SELECT security_id, MIN(date) AS lo, MAX(date) AS hi
                FROM read_parquet('{eod}/**/*.parquet') GROUP BY 1
            ),
            edge AS (SELECT MIN(lo) AS floor_, MAX(hi) AS last_session FROM span)
            SELECT e.* FROM read_parquet('{path}') e
            LEFT JOIN span s USING (security_id), edge
            WHERE s.hi IS NULL                          -- no bars at all: nothing to contradict
               OR ((e.date <= s.hi OR s.hi >= edge.last_session - 7)
               AND (e.date >= s.lo OR s.lo <= edge.floor_ + {_MAX_GAP_DAYS}))
            ORDER BY e.security_id, e.date
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION zstd, COMPRESSION_LEVEL 9)
    """)
    after, = con.execute(f"SELECT COUNT(*) FROM read_parquet('{tmp}')").fetchone()
    con.close()
    tmp.replace(path)
    return before - after


def build_all(ledger: Ledger, start_date: str = START_DATE) -> dict:
    """Build every dataset whose download block is COMPLETE.

    Completeness, not mere presence, is the gate. A parquet built from a
    partially downloaded block is worse than no parquet at all: it carries no
    marker of its own incompleteness, so anything reading it computes over a
    silently truncated universe. Blocks still in flight are skipped and any
    stale artefact from an earlier partial build is removed.
    """
    universe_size = len(select())
    datasets: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    for name, builder, filename in (
        ("splits", build_splits, "splits.parquet"),
        ("dividends", build_dividends, "dividends.parquet"),
    ):
        done = len(ledger.resolved(name))
        if done >= universe_size:
            datasets[name] = builder(ledger)
        else:
            reason = f"{done:,}/{universe_size:,} securities resolved"
            log.warning("skipping %s: block incomplete (%s)", filename, reason)
            skipped[name] = reason
            (PARQUET_DIR / filename).unlink(missing_ok=True)

    datasets["eod"] = build_eod(ledger, start_date)

    # After the price series is cut to one listing, and only then, the event tables
    # can be held to the same span.
    for name in ("splits", "dividends"):
        if name in datasets:
            removed = trim_events_to_listing(name)
            path = PARQUET_DIR / f"{name}.parquet"
            datasets[name] |= {"dropped_other_listing": removed,
                               "rows": datasets[name]["rows"] - removed,
                               "bytes": path.stat().st_size,
                               "sha256": sha256_file(path)}
            log.info("%s: dropped %d events belonging to a later listing", name, removed)

    manifest = {"start_date": start_date, "datasets": datasets, "skipped": skipped}
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "phase1_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":  # self-check for the isolated-print screen
    def _bars(*closes):
        return [{"date": f"2020-01-{i + 1:02d}", "open": c, "high": c, "low": c,
                 "close": c, "adjusted_close": c, "volume": 100}
                for i, c in enumerate(closes)]

    # A bad print spikes and unwinds -> neutralized.
    s = _bars(10.0, 10.1, 12_000.0, 10.2, 10.15)
    assert _neutralize_isolated_prints(s) == 1
    assert s[2]["close"] is None and s[2]["volume"] is None
    assert s[1]["close"] == 10.1 and s[3]["close"] == 10.2  # neighbours untouched
    assert s[2]["date"] == "2020-01-03"                     # the row survives

    # A real 3:1 split is absorbed by adjusted_close, so nothing to see.
    assert _neutralize_isolated_prints(_bars(30.0, 30.3, 30.1, 30.2, 30.15)) == 0
    # A genuine repricing that never comes back (takeover, halt-and-crash) is kept.
    assert _neutralize_isolated_prints(_bars(10.0, 10.1, 25.0, 24.8, 24.9)) == 0
    # A doubling is a big move but under the 2.2x bar; not our business.
    assert _neutralize_isolated_prints(_bars(10.0, 20.0, 10.0)) == 0
    # Nulls, zeros and short series must not raise.
    assert _neutralize_isolated_prints(_bars(10.0)) == 0
    assert _neutralize_isolated_prints(_bars()) == 0

    # A zero between the spike and its unwind must not hide the spike: "next" is the
    # next bar that PRINTS. This is the case the QA check counted and a naive
    # adjacent-index scan misses.
    s = _bars(10.0, 12_000.0, 0.0, 10.1, 10.0)
    assert _neutralize_isolated_prints(s) == 1
    assert s[1]["close"] is None and s[2]["close"] == 0.0

    # Two spiked bars back to back are NOT an isolated print: nothing unwinds on the
    # next bar. A two-day plateau is a repricing as far as this rule can tell, and
    # guessing at it would start deleting real moves. Left for QA to report.
    s = _bars(10.0, 12_000.0, 11_000.0, 10.1, 10.0)
    assert _neutralize_isolated_prints(s) == 0

    # Two separate spikes are both caught and the good bar between them is spared.
    # A naive "flag every match" pass eats the 10.1: it disagrees with both of its
    # (bad) neighbours, which agree with each other, so it fits the rule perfectly.
    s = _bars(10.0, 12_000.0, 10.1, 12_100.0, 10.2)
    assert _neutralize_isolated_prints(s) == 2
    assert s[1]["close"] is None and s[3]["close"] is None
    assert s[0]["close"] == 10.0 and s[2]["close"] == 10.1

    # AEZ.US 2008-04-25..30: two adjacent junk bars, the milder one only visible once
    # the wilder one is gone. Both must go, and it must not take the 20.76 or 23.40.
    s = _bars(20.76, 0.035, 8.6, 23.40, 23.40)
    assert _neutralize_isolated_prints(s) == 2
    assert s[1]["close"] is None and s[2]["close"] is None
    assert s[0]["close"] == 20.76 and s[3]["close"] == 23.40

    # ---- splice truncation ----
    # PVX.US: a ~0.09 penny stock and a ~4000 foreign line on the same ticker. Cut at
    # the first regime change, keep everything before it.
    s = _bars(0.09, 0.085, 0.09, 4000.0, 3900.0, 4000.0)
    assert _splice_cut(s) == 3
    # Clean series are never cut, however volatile.
    assert _splice_cut(_bars(10.0, 20.0, 8.0, 30.0, 12.0)) == 5
    assert _splice_cut(_bars(10.0)) == 1
    assert _splice_cut(_bars()) == 0
    # The cut is FORWARD only: a late splice must not touch the early history.
    s = _bars(*([10.0] * 50 + [90_000.0] * 50))
    assert _splice_cut(s) == 50
    # Non-printing bars are skipped when measuring the step, not treated as a regime.
    s = _bars(0.09, 0.09, 0.09, 4000.0)
    s[2]["adjusted_close"] = None
    assert _splice_cut(s) == 3

    # ---- bar sanitation ----
    # A zero is the vendor's "no print", not a price. Blanked, not deleted.
    s = _bars(10.0, 0.0, 10.2)
    assert _sanitize(s) == (1, 0)
    assert s[1]["close"] is None and s[1]["volume"] is None
    assert s[1]["date"] == "2020-01-02" and s[0]["close"] == 10.0

    # high/low that do not bracket open/close are widened to the bar's own extremes.
    s = _bars(10.0)
    s[0].update(open=9.0, high=10.0, low=9.5, close=11.0)   # open<low and close>high
    assert _sanitize(s) == (0, 1)
    assert s[0]["high"] == 11.0 and s[0]["low"] == 9.0
    assert s[0]["open"] == 9.0 and s[0]["close"] == 11.0    # printed values untouched
    assert _sanitize(_bars(10.0, 10.1)) == (0, 0)           # valid bars are left alone
    assert _sanitize([]) == (0, 0)

    # ---- listing spans ----
    def _at(*pairs):
        return [{"date": d, "open": c, "high": c, "low": c, "close": c,
                 "adjusted_close": c, "volume": 100} for d, c in pairs]

    # SVRA.US: one stray print in 2000, then the real series from 2002. The stray goes.
    s = _at(("2000-07-26", 5.0), *[(f"2002-06-{d:02d}", 9.0) for d in range(11, 25)])
    assert _listing_span(s) == (1, len(s))
    # A stray at the END goes too.
    s = _at(*[(f"2002-06-{d:02d}", 9.0) for d in range(11, 25)], ("2019-01-04", 3.0))
    assert _listing_span(s) == (0, len(s) - 1)
    # INTW.US shape: two real listings on one ticker at similar prices, so no price step
    # for _splice_cut to see. Keep the earlier one, forward-only.
    old = [(f"2000-0{m}-03", 10.0) for m in range(1, 10)]
    new = [(f"2024-0{m}-03", 12.0) for m in range(1, 10)]
    assert _listing_span(_at(*old, *new)) == (0, len(old))
    assert _splice_cut(_at(*old, *new)) == len(old) + len(new)   # invisible to the price rule
    # Continuous series are never cut, and a short whole history is never emptied.
    assert _listing_span(_at(*old)) == (0, len(old))
    assert _listing_span(_at(("2020-01-02", 1.0), ("2020-01-03", 1.0))) == (0, 2)
    assert _listing_span([]) == (0, 0)
    # INDZ_old.US: every segment is short, so there is no "substantial" one to prefer.
    # Keeping the first would leave the hole in; the longest is kept instead.
    s = _at(("2006-01-03", 1.0), ("2006-01-04", 1.0),
            ("2019-06-03", 2.0), ("2019-06-04", 2.0), ("2019-06-05", 2.0))
    assert _listing_span(s) == (2, 5)
    # Equal-length short segments fall back to the earlier one.
    assert _listing_span(_at(("2006-01-03", 1.0), ("2019-06-03", 2.0))) == (0, 1)

    # A 1:10 reverse split the vendor failed to absorb looks identical to a splice.
    # With the split on record the history is kept; without it, it is cut. The +/- 5
    # day window is pre-expanded by _split_windows, so this set is every guarded date.
    s = _bars(0.09, 0.085, 0.09, 0.90, 0.88)   # dates are 2020-01-01..05
    assert _splice_cut(s) == 3
    assert _splice_cut(s, {"2020-01-04"}) == len(s)
    assert _splice_cut(s, {"2020-06-01"}) == 3          # unrelated split, still cut

    print("prices self-check ok")
