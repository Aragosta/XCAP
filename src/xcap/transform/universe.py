"""Raw exchange/symbol JSON -> exchanges.parquet + securities.parquet.

Reads exclusively from the raw archive via the ledger, never from the network,
so it can be re-run any number of times after the subscription ends.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CATALOG_DIR, PARQUET_DIR
from ..eodhd.client import EodhdClient
from ..ledger import Ledger
from ..schemas.universe import EXCHANGES, SECURITIES

log = logging.getLogger("xcap.transform.universe")


def _write(table: pa.Table, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table, path,
        compression="zstd", compression_level=9,
        use_dictionary=True, version="2.6",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.relative_to(PARQUET_DIR.parent)),
        "rows": table.num_rows,
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def build_exchanges(ledger: Ledger, snapshot: date) -> dict:
    rows = ledger.rows("exchanges-list", status="ok")
    if not rows:
        raise SystemExit("no exchanges-list in the raw archive; run `phase0 fetch` first")
    payload = json.loads(EodhdClient.read_raw(Path(rows[0]["raw_path"])))

    cols = {
        "code": [e.get("Code") for e in payload],
        "name": [e.get("Name") for e in payload],
        "operating_mic": [e.get("OperatingMIC") for e in payload],
        "country": [e.get("Country") for e in payload],
        "currency": [e.get("Currency") for e in payload],
        "country_iso2": [e.get("CountryISO2") for e in payload],
        "country_iso3": [e.get("CountryISO3") for e in payload],
        "snapshot_date": [snapshot] * len(payload),
    }
    table = pa.table(cols, schema=EXCHANGES)
    return _write(table, PARQUET_DIR / "exchanges.parquet")


def build_securities(ledger: Ledger, snapshot: date) -> dict:
    """Merge every active/delisted ticker list into one deduplicated universe."""
    rows = ledger.rows("exchange-symbol-list", status="ok")
    if not rows:
        raise SystemExit("no symbol lists in the raw archive; run `phase0 fetch` first")

    # key -> record. A security may legitimately appear in both the active and
    # delisted lists (e.g. re-listings); we keep one row and flag both origins
    # rather than silently dropping either, so QA can quantify the overlap.
    merged: dict[tuple[str, str], dict] = {}

    for row in rows:
        exchange_code, _, kind = row["key"].rpartition(".")
        delisted = kind == "delisted"
        payload = json.loads(EodhdClient.read_raw(Path(row["raw_path"])))
        if not isinstance(payload, list):
            log.warning("unexpected payload shape for %s; skipping", row["key"])
            continue

        for sec in payload:
            code = sec.get("Code")
            if not code:
                continue
            key = (exchange_code, code)
            rec = merged.get(key)
            if rec is None:
                rec = {
                    "source_exchange": exchange_code,
                    "code": code,
                    "api_ticker": f"{code}.{exchange_code}",
                    "venue": sec.get("Exchange"),
                    "name": sec.get("Name"),
                    "country": sec.get("Country"),
                    "currency": sec.get("Currency"),
                    "type": sec.get("Type"),
                    "isin": sec.get("Isin") or sec.get("ISIN"),
                    "listed_active": False,
                    "listed_delisted": False,
                }
                merged[key] = rec
            rec["listed_delisted" if delisted else "listed_active"] = True
            # Prefer non-null metadata from whichever list carries it.
            for src, dst in (("Name", "name"), ("Isin", "isin"), ("Type", "type"),
                             ("Currency", "currency"), ("Country", "country"),
                             ("Exchange", "venue")):
                if rec[dst] in (None, "") and sec.get(src):
                    rec[dst] = sec[src]

    # Deterministic ordering, so security_id is stable across rebuilds of the
    # same raw archive.
    records = sorted(merged.values(), key=lambda r: (r["source_exchange"], r["code"]))
    for i, rec in enumerate(records, start=1):
        rec["security_id"] = i
        rec["is_delisted"] = rec["listed_delisted"] and not rec["listed_active"]
        rec["snapshot_date"] = snapshot

    cols = {f.name: [r[f.name] for r in records] for f in SECURITIES}
    table = pa.table(cols, schema=SECURITIES)
    return _write(table, PARQUET_DIR / "securities.parquet")


def build_all(ledger: Ledger, snapshot: date | None = None) -> dict:
    snapshot = snapshot or date.today()
    manifest = {
        "snapshot_date": snapshot.isoformat(),
        "datasets": {
            "exchanges": build_exchanges(ledger, snapshot),
            "securities": build_securities(ledger, snapshot),
        },
    }
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "phase0_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
