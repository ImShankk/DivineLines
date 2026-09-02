"""Source adapter foundation.

Every external source in the platform goes through this class so that all of
them share: timeouts, bounded retries with exponential backoff, polite
per-host rate limiting, on-disk response caching, structured logging, and a
recorded last-success timestamp.

Two rules are enforced here rather than left to each adapter:

* a failed fetch raises or returns ``None`` — it **never** fabricates data;
* every payload carries its source name and retrieval timestamp.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

from ..config import settings
from ..db.repository import record_source_status
from ..logging_setup import get_logger

log = get_logger(__name__)

_LAST_REQUEST_AT: dict[str, float] = {}


class SourceError(RuntimeError):
    """Raised when a source cannot deliver usable data."""


class RateLimitError(SourceError):
    """Raised when a source signals quota exhaustion (HTTP 429 / 401 quota)."""


@dataclass
class FetchResult:
    """A payload plus the provenance needed to trust it."""

    data: Any
    source: str
    dataset: str
    retrieved_at: datetime
    from_cache: bool = False
    url: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.retrieved_at).total_seconds()


class HttpSource:
    """Base class for HTTP-backed adapters."""

    #: Short identifier used in provenance columns and the status page.
    name: str = "http"
    #: Directory the on-disk cache lives in.  Defaults to ``name``, but two
    #: adapters that read the *same* upstream document should share one, so a
    #: payload fetched by one is not re-fetched by the other.
    cache_namespace: str | None = None
    #: Seconds a cached response stays usable.  Override per adapter.
    cache_ttl: int = 3600
    #: Minimum seconds between requests to this source.
    min_interval: float = settings.sources.http_min_interval
    #: Some hosts reject non-browser agents outright; adapters may override.
    user_agent: str | None = None

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": self.user_agent or settings.sources.user_agent}
        )
        #: Headers from the most recent live response (quota tracking, etc.).
        self.last_response_headers: dict[str, str] = {}
        self.cache_dir = settings.paths.cache_dir / (self.cache_namespace or self.name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- cache

    def _cache_path(self, url: str, params: Mapping[str, Any] | None, suffix: str) -> Path:
        # Secrets must never end up in a cache key that is written to disk in
        # readable form; hashing the whole request handles that.
        blob = json.dumps({"url": url, "params": dict(params or {})}, sort_keys=True)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}{suffix}"

    def _read_cache(self, path: Path, ttl: int | None) -> tuple[bytes, datetime] | None:
        if not path.exists():
            return None
        ttl = self.cache_ttl if ttl is None else ttl
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if ttl >= 0 and datetime.now(timezone.utc) - modified > timedelta(seconds=ttl):
            return None
        return path.read_bytes(), modified

    # -------------------------------------------------------------- fetching

    def _throttle(self) -> None:
        last = _LAST_REQUEST_AT.get(self.name)
        if last is not None:
            wait = self.min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        _LAST_REQUEST_AT[self.name] = time.monotonic()

    def fetch(
        self,
        url: str,
        *,
        dataset: str,
        params: Mapping[str, Any] | None = None,
        ttl: int | None = None,
        suffix: str = ".json",
        force: bool = False,
        allow_cache_fallback: bool = True,
        status_dataset: str | None = None,
    ) -> FetchResult:
        """Fetch ``url`` with caching, retries and provenance.

        On failure, a still-cached (but stale) response is returned when
        ``allow_cache_fallback`` is set, flagged with ``stale=True`` in
        ``meta`` so callers can surface the staleness rather than hide it.
        """
        # Per-event adapters cache by event but should report health by feed:
        # one status row per lineup fetched would bury the status page.
        status_key = status_dataset or dataset
        cache_path = self._cache_path(url, params, suffix)

        if not force:
            cached = self._read_cache(cache_path, ttl)
            if cached is not None:
                payload, modified = cached
                return FetchResult(payload, self.name, dataset, modified,
                                   from_cache=True, url=url)

        if settings.offline:
            stale = self._read_cache(cache_path, ttl=-1)
            if stale is not None and allow_cache_fallback:
                payload, modified = stale
                record_source_status(self.name, status_key, status="degraded",
                                     message="offline mode: served stale cache")
                return FetchResult(payload, self.name, dataset, modified, from_cache=True,
                                   url=url, meta={"stale": True, "offline": True})
            record_source_status(self.name, status_key, status="error", message="offline mode")
            raise SourceError(f"{self.name}: offline mode and no cached {dataset}")

        attempts = max(1, settings.sources.http_retries)
        delay = settings.sources.http_backoff
        last_error: Exception | None = None
        started = time.monotonic()

        for attempt in range(1, attempts + 1):
            try:
                self._throttle()
                response = self.session.get(
                    url, params=dict(params or {}), timeout=settings.sources.http_timeout
                )
                if response.status_code == 429:
                    raise RateLimitError(f"{self.name}: rate limited on {dataset}")
                if response.status_code in (401, 403):
                    raise SourceError(
                        f"{self.name}: not authorised for {dataset} (HTTP {response.status_code})"
                    )
                response.raise_for_status()

                self.last_response_headers = dict(response.headers)
                cache_path.write_bytes(response.content)
                latency = int((time.monotonic() - started) * 1000)
                record_source_status(self.name, status_key, status="ok",
                                     rows=None, latency_ms=latency)
                return FetchResult(response.content, self.name, dataset,
                                   datetime.now(timezone.utc), url=response.url)
            except RateLimitError as exc:
                last_error = exc
                log.warning("rate limited", extra={"source": self.name, "dataset": dataset})
                break
            except Exception as exc:  # network, HTTP, parse
                last_error = exc
                log.warning(
                    "fetch failed",
                    extra={"source": self.name, "dataset": dataset,
                           "attempt": attempt, "error": str(exc)},
                )
                if attempt < attempts:
                    time.sleep(delay)
                    delay *= settings.sources.http_backoff

        if allow_cache_fallback:
            stale = self._read_cache(cache_path, ttl=-1)
            if stale is not None:
                payload, modified = stale
                record_source_status(self.name, status_key, status="degraded",
                                     message=f"served stale cache: {last_error}")
                log.warning("serving stale cache", extra={"source": self.name, "dataset": dataset})
                return FetchResult(payload, self.name, dataset, modified, from_cache=True,
                                   url=url, meta={"stale": True, "error": str(last_error)})

        record_source_status(self.name, status_key, status="error", message=str(last_error))
        raise SourceError(f"{self.name}: failed to fetch {dataset}: {last_error}")

    def fetch_json(self, url: str, *, dataset: str, **kwargs: Any) -> FetchResult:
        status_key = kwargs.get("status_dataset") or dataset
        result = self.fetch(url, dataset=dataset, **kwargs)
        try:
            result.data = json.loads(result.data)
        except (TypeError, ValueError) as exc:
            record_source_status(self.name, status_key, status="error",
                                 message=f"invalid JSON: {exc}")
            raise SourceError(f"{self.name}: invalid JSON for {dataset}: {exc}") from exc
        return result
