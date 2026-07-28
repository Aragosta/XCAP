"""Verify and back up the raw archive.

`data/_raw/` is the purchase. The parquet is rebuildable from it; it is not
rebuildable from anything once the subscription ends. Two operations:

  verify  -- every successful ledger row has its raw file present on disk with
             a matching sha256. Catches silent corruption and accidental
             deletion, and is the precondition for trusting any backup.
  backup  -- mirror the archive plus the catalog to a destination, then verify
             the copy independently rather than trusting rsync's exit code.

The catalog is included because the ledger is what makes the archive
interpretable: without it you have 42,000 opaque .zst files and no record of
what they are, what they cost, or whether the set is complete.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import CATALOG_DIR, DATA_DIR, RAW_DIR
from .ledger import Ledger

log = logging.getLogger("xcap.backup")


@dataclass
class VerifyResult:
    checked: int = 0
    ok: int = 0
    missing: list[str] | None = None
    corrupt: list[str] | None = None

    def __post_init__(self) -> None:
        self.missing = self.missing if self.missing is not None else []
        self.corrupt = self.corrupt if self.corrupt is not None else []

    @property
    def clean(self) -> bool:
        return not self.missing and not self.corrupt


def verify_raw(ledger: Ledger, root: Path = RAW_DIR, *,
               full: bool = True, progress_every: int = 5000) -> VerifyResult:
    """Check every archived response against the checksum recorded at fetch time.

    `full=False` checks existence and size only, which is fast enough to run
    routinely; `full=True` rehashes and is what should gate a cancellation.
    """
    res = VerifyResult()
    rows = [r for r in ledger.conn.execute(
        "SELECT endpoint, key, raw_path, sha256, bytes FROM requests WHERE status='ok'"
    ).fetchall()]

    for i, row in enumerate(rows, 1):
        raw_path = row["raw_path"]
        if not raw_path:
            continue
        path = Path(raw_path)
        # Re-root so a backup copy can be verified with the same ledger.
        if root != RAW_DIR:
            try:
                path = root / path.relative_to(RAW_DIR)
            except ValueError:
                pass
        res.checked += 1
        if not path.exists():
            res.missing.append(f"{row['endpoint']}/{row['key']}")
            continue
        if full and row["sha256"]:
            # The stored digest is of the DECOMPRESSED response body, so the
            # archived file must be decompressed before hashing.
            try:
                import zstandard as zstd
                body = zstd.ZstdDecompressor().decompress(path.read_bytes())
            except Exception as exc:  # noqa: BLE001
                res.corrupt.append(f"{row['endpoint']}/{row['key']} (unreadable: {exc})")
                continue
            if hashlib.sha256(body).hexdigest() != row["sha256"]:
                res.corrupt.append(f"{row['endpoint']}/{row['key']} (checksum mismatch)")
                continue
        res.ok += 1
        if progress_every and i % progress_every == 0:
            log.info("  verified %d/%d", i, len(rows))
    return res


def backup(dest: Path, ledger: Ledger, *, verify: bool = True) -> dict:
    """Mirror data/_raw and data/catalog to `dest`, then verify the copy."""
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    src_size = sum(f.stat().st_size for f in RAW_DIR.rglob("*") if f.is_file())
    free = shutil.disk_usage(dest).free
    if free < src_size * 1.1:
        raise SystemExit(
            f"insufficient space at {dest}: need ~{src_size / 1e9:.1f} GB, "
            f"{free / 1e9:.1f} GB free"
        )

    started = datetime.now(timezone.utc)
    for sub in ("_raw", "catalog"):
        src = DATA_DIR / sub
        if not src.exists():
            continue
        log.info("mirroring %s -> %s", src, dest / sub)
        # --delete keeps the mirror a true reflection; without it a rename in
        # the source leaves an orphan in the backup that looks like real data.
        subprocess.run(
            ["rsync", "-a", "--delete", f"{src}/", str(dest / sub) + "/"],
            check=True,
        )

    result = {
        "destination": str(dest),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_bytes": src_size,
        "files": sum(1 for f in (dest / "_raw").rglob("*") if f.is_file()),
    }

    if verify:
        log.info("verifying the copy at %s (rehashing, not trusting rsync)", dest)
        v = verify_raw(ledger, root=dest / "_raw", full=True)
        result["verify"] = {
            "checked": v.checked, "ok": v.ok,
            "missing": len(v.missing), "corrupt": len(v.corrupt),
            "clean": v.clean,
            "missing_examples": v.missing[:10],
            "corrupt_examples": v.corrupt[:10],
        }

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "backup_manifest.json").write_text(json.dumps(result, indent=2))
    # Drop a copy alongside the backup so it is self-describing.
    (dest / "backup_manifest.json").write_text(json.dumps(result, indent=2))
    return result
