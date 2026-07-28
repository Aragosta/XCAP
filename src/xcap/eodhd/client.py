"""Async EODHD client with raw-first archival.

Design rule for this project: the raw response bytes are the asset. Every
successful response is written to data/_raw compressed and checksummed *before*
anything tries to parse it. Parsers can be rewritten; the subscription cannot be
un-cancelled.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import zstandard as zstd

from ..config import Config, RAW_DIR
from ..ledger import Ledger, params_hash
from .budget import Budget, BudgetExceeded, ENDPOINT_COST, RateLimiter

log = logging.getLogger("xcap.client")

_UNSAFE = re.compile(r"[^A-Za-z0-9._=-]+")

# Retried with backoff. 429 additionally honours Retry-After.
RETRY_STATUS = {429, 500, 502, 503, 504}
# Terminal auth/entitlement problems: stop the whole run rather than burn budget.
FATAL_STATUS = {401, 403}


class FatalApiError(RuntimeError):
    """Auth or entitlement failure; retrying cannot help."""


def safe_key(key: str) -> str:
    return _UNSAFE.sub("_", key)[:180]


@dataclass(frozen=True)
class RequestSpec:
    endpoint: str          # logical name, drives cost + raw layout
    path: str              # URL path appended to base_url
    key: str               # unique identity within the endpoint (e.g. ticker)
    params: dict[str, Any] = field(default_factory=dict)
    ext: str = "json"

    @property
    def cost(self) -> int:
        return ENDPOINT_COST.get(self.endpoint, 1)

    @property
    def phash(self) -> str:
        return params_hash(self.params)

    def raw_path(self) -> Path:
        return RAW_DIR / self.endpoint / f"{safe_key(self.key)}.{self.phash}.{self.ext}.zst"


@dataclass
class FetchResult:
    spec: RequestSpec
    status: str                  # ok | empty | not_found | http_error | transport_error
    body: bytes | None = None
    http_status: int | None = None
    from_cache: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class EodhdClient:
    def __init__(self, cfg: Config, ledger: Ledger) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.budget = Budget(ledger, cfg.daily_call_budget)
        self.limiter = RateLimiter(cfg.rate_per_min)
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._client: httpx.AsyncClient | None = None
        self._compressor = zstd.ZstdCompressor(level=10)

    async def __aenter__(self) -> "EodhdClient":
        self._client = httpx.AsyncClient(
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout_s,
            headers={"User-Agent": "xcap-archiver/0.1"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()

    # ---- raw archival ----------------------------------------------------

    def _archive(self, spec: RequestSpec, body: bytes) -> tuple[Path, str]:
        path = spec.raw_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(body).hexdigest()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(self._compressor.compress(body))
        tmp.replace(path)  # atomic: a raw file either exists complete or not at all
        return path, digest

    @staticmethod
    def read_raw(path: Path) -> bytes:
        return zstd.ZstdDecompressor().decompress(path.read_bytes())

    # ---- fetching --------------------------------------------------------

    async def fetch(self, spec: RequestSpec, *, force: bool = False) -> FetchResult:
        """Fetch one request, resuming from the ledger unless `force`."""
        if not force:
            prior = self.ledger.lookup(spec.endpoint, spec.key, spec.phash)
            if prior and prior["status"] in ("ok", "empty", "not_found"):
                raw = Path(prior["raw_path"]) if prior["raw_path"] else None
                if prior["status"] != "ok" or (raw and raw.exists()):
                    return FetchResult(
                        spec=spec,
                        status=prior["status"],
                        body=self.read_raw(raw) if raw and raw.exists() else None,
                        http_status=prior["http_status"],
                        from_cache=True,
                    )

        params = dict(spec.params)
        params["api_token"] = self.cfg.api_token
        params.setdefault("fmt", "json")

        attempts = 0
        charged = 0
        last_status: int | None = None
        last_error: str | None = None

        async with self._sem:
            for attempt in range(1, self.cfg.max_attempts + 1):
                await self.limiter.acquire()
                try:
                    await self.budget.charge(spec.cost)
                    charged += spec.cost
                except BudgetExceeded:
                    self.ledger.record(
                        endpoint=spec.endpoint, key=spec.key, phash=spec.phash,
                        url=spec.path, status="transport_error", http_status=None,
                        attempts=attempts, call_cost=charged,
                        error="daily budget exhausted",
                    )
                    raise

                attempts += 1
                assert self._client is not None
                try:
                    resp = await self._client.get(spec.path, params=params)
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue

                last_status = resp.status_code

                if resp.status_code in FATAL_STATUS:
                    self.ledger.record(
                        endpoint=spec.endpoint, key=spec.key, phash=spec.phash,
                        url=str(resp.url.copy_remove_param("api_token")),
                        status="http_error", http_status=resp.status_code,
                        attempts=attempts, call_cost=charged, error=resp.text[:500],
                    )
                    raise FatalApiError(
                        f"{resp.status_code} on {spec.endpoint}/{spec.key} — "
                        "check token and plan entitlements"
                    )

                if resp.status_code == 404:
                    self.ledger.record(
                        endpoint=spec.endpoint, key=spec.key, phash=spec.phash,
                        url=str(resp.url.copy_remove_param("api_token")),
                        status="not_found", http_status=404,
                        attempts=attempts, call_cost=charged,
                    )
                    return FetchResult(spec, "not_found", http_status=404)

                if resp.status_code in RETRY_STATUS:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() \
                        else min(2 ** attempt, 60)
                    last_error = f"HTTP {resp.status_code}"
                    log.warning("%s/%s -> %s, retry in %.0fs",
                                spec.endpoint, spec.key, resp.status_code, delay)
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code != 200:
                    self.ledger.record(
                        endpoint=spec.endpoint, key=spec.key, phash=spec.phash,
                        url=str(resp.url.copy_remove_param("api_token")),
                        status="http_error", http_status=resp.status_code,
                        attempts=attempts, call_cost=charged, error=resp.text[:500],
                    )
                    return FetchResult(spec, "http_error", http_status=resp.status_code,
                                       error=resp.text[:500])

                body = resp.content
                # An empty list/object is a real answer, not a failure — record it
                # as such so coverage accounting can distinguish "no data" from "not fetched".
                if not body or body.strip() in (b"[]", b"{}", b'""'):
                    self.ledger.record(
                        endpoint=spec.endpoint, key=spec.key, phash=spec.phash,
                        url=str(resp.url.copy_remove_param("api_token")),
                        status="empty", http_status=200,
                        attempts=attempts, call_cost=charged, nbytes=len(body),
                    )
                    return FetchResult(spec, "empty", body=body, http_status=200)

                path, digest = self._archive(spec, body)
                self.ledger.record(
                    endpoint=spec.endpoint, key=spec.key, phash=spec.phash,
                    url=str(resp.url.copy_remove_param("api_token")),
                    status="ok", http_status=200, attempts=attempts,
                    call_cost=charged, nbytes=len(body), sha256=digest,
                    raw_path=str(path),
                )
                return FetchResult(spec, "ok", body=body, http_status=200)

        self.ledger.record(
            endpoint=spec.endpoint, key=spec.key, phash=spec.phash, url=spec.path,
            status="transport_error", http_status=last_status,
            attempts=attempts, call_cost=charged, error=last_error,
        )
        return FetchResult(spec, "transport_error", http_status=last_status, error=last_error)

    async def fetch_all(self, specs: list[RequestSpec], *, force: bool = False) -> list[FetchResult]:
        return list(await asyncio.gather(*(self.fetch(s, force=force) for s in specs)))
