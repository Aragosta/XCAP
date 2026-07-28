"""Raw price/corporate-action JSON -> partitioned parquet.

EOD is far too large to hold in memory, so it streams into staging files and
DuckDB performs the out-of-core sort and partitioning. Dates stay as strings
through staging (cheap) and are cast once at finalise time.

Reads only from the raw archive. Safe to re-run after the subscription ends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CATALOG_DIR, DATA_DIR, PARQUET_DIR
from ..eodhd.client import EodhdClient
from ..ledger import Ledger
from ..schemas.prices import DIVIDENDS, SPLITS
from ..universe import START_DATE, select

log = logging.getLogger("xcap.transform.prices")

STAGING = DATA_DIR / "_staging"
ROWS_PER_STAGE_FILE = 4_000_000

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


def _ticker_to_id() -> dict[str, int]:
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT api_ticker, security_id FROM read_parquet('{PARQUET_DIR / 'securities.parquet'}')"
    ).fetchall()
    con.close()
    return {t: int(i) for t, i in rows}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    stage_dir = STAGING / "eod"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    cols: dict[str, list] = {f.name: [] for f in STAGE_EOD}
    part = 0
    total_rows = 0
    dropped_pre_start = 0
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

        kept = 0
        for bar in series:
            d = bar.get("date")
            if not d:
                continue
            if d < start_date:
                dropped_pre_start += 1
                continue
            cols["security_id"].append(sec_id)
            cols["api_ticker"].append(ticker)
            cols["date"].append(d)
            cols["open"].append(_as_float(bar.get("open")))
            cols["high"].append(_as_float(bar.get("high")))
            cols["low"].append(_as_float(bar.get("low")))
            cols["close"].append(_as_float(bar.get("close")))
            cols["vendor_adjusted_close"].append(_as_float(bar.get("adjusted_close")))
            cols["volume"].append(_as_float(bar.get("volume")))
            kept += 1

        total_rows += kept
        if kept:
            securities += 1
        if len(cols["date"]) >= ROWS_PER_STAGE_FILE:
            flush()
        if n % 5000 == 0:
            log.info("  %d/%d series, %d rows staged", n, len(rows), total_rows)

    flush()
    log.info("staged %d rows from %d securities (%d bars dropped before %s)",
             total_rows, securities, dropped_pre_start, start_date)

    out = PARQUET_DIR / "eod"
    if out.exists():
        shutil.rmtree(out)

    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DATA_DIR / '_duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order=false")
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
            "bytes": path.stat().st_size, "sha256": _sha(path)}


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
            "bytes": path.stat().st_size, "sha256": _sha(path)}


def _resolved(ledger: Ledger, endpoint: str) -> int:
    """Securities with a terminal answer for `endpoint` -- ok, empty or 404."""
    return sum(len(ledger.rows(endpoint, status=s))
               for s in ("ok", "empty", "not_found"))


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
        done = _resolved(ledger, name)
        if done >= universe_size:
            datasets[name] = builder(ledger)
        else:
            reason = f"{done:,}/{universe_size:,} securities resolved"
            log.warning("skipping %s: block incomplete (%s)", filename, reason)
            skipped[name] = reason
            (PARQUET_DIR / filename).unlink(missing_ok=True)

    datasets["eod"] = build_eod(ledger, start_date)
    manifest = {"start_date": start_date, "datasets": datasets, "skipped": skipped}
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "phase1_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
