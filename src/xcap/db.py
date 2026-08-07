"""DuckDB access: one connection recipe, one query shorthand.

Every analytic read in this project is "open, run one statement, close". Spelling
that out per call site meant seventeen copies of the same three lines and two
private `_connect` helpers that had already drifted apart on temp-directory and
ordering settings.
"""

from __future__ import annotations

import duckdb

from .config import DATA_DIR


def connect() -> duckdb.DuckDBPyConnection:
    """A connection with the settings every caller wants: spill to a known temp
    directory (the default lands in /tmp and dies on large joins), and no
    insertion-order bookkeeping, which is pure overhead for aggregate reads."""
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DATA_DIR / '_duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def query(sql: str, params: list | None = None) -> list[tuple]:
    """Run one statement on a throwaway connection and return every row."""
    con = connect()
    try:
        return con.execute(sql, params or []).fetchall()
    finally:
        con.close()


def quoted(values) -> str:
    """SQL literal list for an IN clause. For identifiers and enum-ish constants
    defined in this codebase -- not for anything reaching it from the vendor."""
    return ",".join(f"'{v}'" for v in values)
