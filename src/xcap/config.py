"""Runtime configuration, loaded from the environment (and a .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "_raw"
PARQUET_DIR = DATA_DIR / "parquet"
CATALOG_DIR = DATA_DIR / "catalog"
LEDGER_PATH = CATALOG_DIR / "ledger.sqlite"


def _load_dotenv() -> None:
    """Minimal .env reader so we don't take a dependency for four variables."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Config:
    api_token: str
    base_url: str = "https://eodhd.com/api"
    rate_per_min: int = 800
    daily_call_budget: int = 100_000
    concurrency: int = 8
    timeout_s: float = 60.0
    max_attempts: int = 5

    @property
    def redacted_token(self) -> str:
        return f"{self.api_token[:6]}...{self.api_token[-4:]}"


def load_config() -> Config:
    _load_dotenv()
    token = os.environ.get("EODHD_API_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "EODHD_API_TOKEN is not set. Copy .env.example to .env and fill it in."
        )
    return Config(
        api_token=token,
        rate_per_min=int(os.environ.get("XCAP_RATE_PER_MIN", 800)),
        daily_call_budget=int(os.environ.get("XCAP_DAILY_CALL_BUDGET", 100_000)),
        concurrency=int(os.environ.get("XCAP_CONCURRENCY", 8)),
    )


def ensure_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, PARQUET_DIR, CATALOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
