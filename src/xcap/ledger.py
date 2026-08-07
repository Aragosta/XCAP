"""The job ledger: the single source of truth for what has been fetched and what it cost.

Every HTTP attempt against EODHD is recorded here. This makes the extraction
resumable, stops retries from double-spending the API budget, and turns
"did we get everything?" into a SQL query instead of a guess.

SQLite rather than DuckDB: this is a high-frequency small-transaction write
workload, which is SQLite's home turf. DuckDB is used for analytics over the
resulting parquet in xcap.qa.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import LEDGER_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    endpoint     TEXT NOT NULL,
    key          TEXT NOT NULL,
    params_hash  TEXT NOT NULL,
    url          TEXT,
    status       TEXT NOT NULL,   -- ok | empty | not_found | http_error | transport_error
    http_status  INTEGER,
    attempts     INTEGER NOT NULL DEFAULT 0,
    call_cost    INTEGER NOT NULL DEFAULT 0,
    bytes        INTEGER,
    sha256       TEXT,
    raw_path     TEXT,
    error        TEXT,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (endpoint, key, params_hash)
);

CREATE INDEX IF NOT EXISTS idx_requests_status ON requests (endpoint, status);

-- Daily spend, so the budget survives process restarts. Keyed on the GMT day
-- because that is when EODHD resets subscription limits.
CREATE TABLE IF NOT EXISTS budget_day (
    gmt_day     TEXT PRIMARY KEY,
    calls_spent INTEGER NOT NULL DEFAULT 0
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def params_hash(params: dict[str, Any]) -> str:
    """Stable hash of request params, excluding the secret."""
    scrubbed = {k: v for k, v in sorted(params.items()) if k != "api_token"}
    blob = json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, path: Path = LEDGER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- request records -------------------------------------------------

    def lookup(self, endpoint: str, key: str, phash: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            "SELECT * FROM requests WHERE endpoint=? AND key=? AND params_hash=?",
            (endpoint, key, phash),
        )
        return cur.fetchone()

    def record(
        self,
        *,
        endpoint: str,
        key: str,
        phash: str,
        url: str,
        status: str,
        http_status: int | None,
        attempts: int,
        call_cost: int,
        nbytes: int | None = None,
        sha256: str | None = None,
        raw_path: str | None = None,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO requests (endpoint, key, params_hash, url, status, http_status,
                                  attempts, call_cost, bytes, sha256, raw_path, error, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (endpoint, key, params_hash) DO UPDATE SET
                url=excluded.url, status=excluded.status, http_status=excluded.http_status,
                attempts=requests.attempts + excluded.attempts,
                call_cost=requests.call_cost + excluded.call_cost,
                bytes=excluded.bytes, sha256=excluded.sha256, raw_path=excluded.raw_path,
                error=excluded.error, fetched_at=excluded.fetched_at
            """,
            (endpoint, key, phash, url, status, http_status, attempts, call_cost,
             nbytes, sha256, raw_path, error, utcnow()),
        )
        self.conn.commit()

    def rows(self, endpoint: str, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            cur = self.conn.execute(
                "SELECT * FROM requests WHERE endpoint=? AND status=? ORDER BY key",
                (endpoint, status),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM requests WHERE endpoint=? ORDER BY key", (endpoint,)
            )
        return cur.fetchall()

    def resolved(self, endpoint: str) -> set[str]:
        """Keys with a terminal answer -- ok, empty or 404 all mean 'do not re-spend'."""
        cur = self.conn.execute(
            "SELECT key FROM requests WHERE endpoint=? "
            "AND status IN ('ok','empty','not_found')",
            (endpoint,),
        )
        return {r["key"] for r in cur}

    def failures(self) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM requests WHERE status NOT IN ('ok','empty') ORDER BY endpoint, key"
        )
        return cur.fetchall()

    def summary(self) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            """
            SELECT endpoint, status, COUNT(*) AS n,
                   SUM(call_cost) AS calls, SUM(bytes) AS bytes
            FROM requests GROUP BY endpoint, status ORDER BY endpoint, status
            """
        )
        return cur.fetchall()

    # ---- budget ----------------------------------------------------------

    def spend(self, gmt_day: str, calls: int) -> int:
        self.conn.execute(
            """
            INSERT INTO budget_day (gmt_day, calls_spent) VALUES (?, ?)
            ON CONFLICT (gmt_day) DO UPDATE SET calls_spent = calls_spent + excluded.calls_spent
            """,
            (gmt_day, calls),
        )
        self.conn.commit()
        return self.spent_today(gmt_day)

    def spent_today(self, gmt_day: str) -> int:
        cur = self.conn.execute(
            "SELECT calls_spent FROM budget_day WHERE gmt_day=?", (gmt_day,)
        )
        row = cur.fetchone()
        return row["calls_spent"] if row else 0

    def budget_history(self) -> Iterable[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM budget_day ORDER BY gmt_day"
        ).fetchall()
