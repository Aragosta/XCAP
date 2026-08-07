"""Raw exchange/symbol JSON -> exchanges.parquet + securities.parquet.

Reads exclusively from the raw archive via the ledger, never from the network,
so it can be re-run any number of times after the subscription ends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
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


#: Exchange test instruments. NASDAQ publishes ZVZZT/ZWZZT/ZXZZT, NYSE ATEST-*/NTEST-*,
#: Bats ZBZX and so on purely to exercise market-data plumbing. They carry synthetic
#: prices (ZWZZT printed 0.0047 -> 120,490.73 in one session) and a liquidity screen
#: selects for them, so they must never reach a backtest.
#:
#: "test" in the name is not the signal -- that catches Advantest, Aehr Test Systems,
#: Biotest and forty others. Three signals that are: a reserved NASDAQ/Bats symbol,
#: a name the exchange wrote about its own plumbing, and a name that is just the ticker
#: repeated back (a real company always has a real name).
_RESERVED_CODE = re.compile(r"^(Z[A-Z]ZZT|ZVZZCNX|ZXYZ(-[A-Z])?|ZBZX)$")
_TESTISH_CODE = re.compile(r"^([A-Z]?TEST(-[A-Z]{1,2})?|TEST[A-Z])(_old)?$")
_TEST_NAME = re.compile(
    r"(NYSE|NASDAQ|NYSE ARCA|BATS|CBOE)\b.*TEST"
    r"|LISTED TEST|TEST STOCK|TEST CONTROL|TEST INSTRUMENT|TE?ST SECURITY|TEST SYMBOL",
    re.I,
)


def _is_test_instrument(rec: dict) -> bool:
    code = rec.get("code") or ""
    name = rec.get("name") or ""
    if _RESERVED_CODE.match(code) or _TEST_NAME.search(name):
        return True
    # A testish code alone is not enough (ZTEST Electronics is a real company), so
    # require the name to add nothing beyond the ticker itself.
    return bool(_TESTISH_CODE.match(code) and _squash(name) == _squash(code))


def _squash(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper()).removesuffix("OLD")


def _existing_ids() -> dict[tuple[str, str], int]:
    """(source_exchange, code) -> security_id from the previous build, if any.

    securities.parquet is its own id registry: there is no separate state file to
    lose, and a rebuild from an unchanged archive is a no-op.
    """
    path = PARQUET_DIR / "securities.parquet"
    if not path.exists():
        return {}
    t = pq.read_table(path, columns=["source_exchange", "code", "security_id"])
    return {(e, c): int(i) for e, c, i in
            zip(t["source_exchange"].to_pylist(), t["code"].to_pylist(),
                t["security_id"].to_pylist())}


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

    # security_id must survive a universe refresh. A bare enumerate() over the sorted
    # universe is a POSITION, not an identity: one new ticker inserted alphabetically
    # shifts every id after it, and any table built before the refresh (eod, splits,
    # dividends) silently starts pointing at a different company. That happened here
    # once already — a 07-31 price build against an 08-04 universe put all 31,513
    # tickers on the wrong id. So carry existing ids forward and only ever append.
    records = sorted(merged.values(), key=lambda r: (r["source_exchange"], r["code"]))
    n_before = len(records)
    records = [r for r in records if not _is_test_instrument(r)]
    if n_before != len(records):
        log.info("dropped %d exchange test instruments", n_before - len(records))

    known = _existing_ids()
    next_id = max(known.values(), default=0) + 1
    for rec in records:
        key = (rec["source_exchange"], rec["code"])
        if key not in known:
            known[key] = next_id
            next_id += 1
        rec["security_id"] = known[key]
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


if __name__ == "__main__":  # self-check for the test-instrument filter
    def _t(code, name):
        return _is_test_instrument({"code": code, "name": name})

    for code, name in [("ZVZZT", "NASDAQ TEST STOCK"), ("ZBZX", "Bats Listed Test"),
                       ("ZXYZ-A", "NASDAQ SYMBOLOGY TEST"), ("ATEST-A", "ATEST-A"),
                       ("NTEST-B", "NTEST.B"), ("PTEST", "PTEST"), ("TESTF", "TESTF"),
                       ("TEST_old", "TEST"), ("ZTEST", "ZTEST"), ("ZVV", "LISTED TEST SYMBOL"),
                       ("ATEST", "Tick Pilot Test Control Common Stock"),
                       ("CBO", "NYSE LISTED TEST STOCK FOR CTS AND CQS")]:
        assert _t(code, name), (code, name)

    # Real companies. Every one of these has "test" in the name or a testish ticker.
    for code, name in [("ADTTF", "Advantest Corporation"), ("AEHR", "Aehr Test Systems"),
                       ("BIO", "Biotest Aktiengesellschaft"), ("USER", "User Testing Inc"),
                       ("ZTSTF", "ZTEST Electronics Inc"), ("INTT", "inTest Corporation"),
                       ("HSCS", "Heart Test Laboratories Inc. Common Stock"),
                       ("CBO", "Cobram Estate Olives Ltd"),
                       ("CBX", "Cortex Business Solutions Inc"),
                       ("2908", "Test Rite International Co Ltd"),
                       ("ATEST", "Advanced Testing Corp")]:   # testish code, real name
        assert not _t(code, name), (code, name)

    assert not _t("", "") and not _t(None, None)
    print("universe self-check ok")
