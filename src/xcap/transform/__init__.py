"""Raw archive -> parquet builders, and the few things all of them need."""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Rows per staging file. Big enough that the row-group layout stays useful,
#: small enough that a build never holds a whole dataset in memory.
ROWS_PER_STAGE_FILE = 4_000_000


def sha256_file(path: Path) -> str:
    """Checksum a parquet artifact without reading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
