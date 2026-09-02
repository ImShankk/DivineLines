"""stats.nba.com adapter (via ``nba_api``).

Replaces the original ``syncData``/``data_refresh`` scripts.  Three problems
in those are fixed here:

* the season was hard-coded to ``"2025-26"`` — it is now derived from the
  clock, so a new season needs no code edit;
* ``GAME_ID`` was appended in whatever form the API returned, which is how the
  2025-26 season ended up duplicated under padded and unpadded ids — ids are
  now normalised to their canonical 10-character form before any comparison;
* games still in progress were stored as if final — only games with a decided
  result and a full complement of minutes are accepted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..config import nba_season_for_date, settings
from ..db.repository import record_source_status
from ..identity import resolve_nba_team
from ..logging_setup import get_logger
from .base import SourceError

log = get_logger(__name__)

SOURCE_NAME = "nba_stats"


@dataclass
class NbaGameLogs:
    frame: pd.DataFrame
    season: str
    retrieved_at: datetime
    rows: int


class NbaStatsSource:
    """Thin, defensive wrapper around ``nba_api``."""

    name = SOURCE_NAME

    def __init__(self, request_timeout: float | None = None) -> None:
        self.timeout = request_timeout or settings.sources.http_timeout

    def _endpoint(self):
        try:
            from nba_api.stats.endpoints import leaguegamelog  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SourceError(
                "nba_api is not installed; run `pip install -r requirements.txt`"
            ) from exc
        return leaguegamelog

    def fetch_season_logs(
        self,
        season: str | None = None,
        *,
        season_type: str = "Regular Season",
        date_from: str | None = None,
        retries: int = 2,
    ) -> NbaGameLogs:
        """Fetch team game logs for a season (defaults to the current one)."""
        season = season or nba_season_for_date()
        leaguegamelog = self._endpoint()
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(1, retries + 2):
            try:
                endpoint = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=season_type,
                    date_from_nullable=date_from or "",
                    timeout=self.timeout,
                )
                frame = endpoint.get_data_frames()[0]
                break
            except Exception as exc:  # network / API shape
                last_error = exc
                log.warning("nba_api call failed",
                            extra={"season": season, "attempt": attempt, "error": str(exc)})
                if attempt <= retries:
                    time.sleep(settings.sources.http_backoff * attempt)
        else:
            record_source_status(SOURCE_NAME, f"game_logs:{season}", status="error",
                                 message=str(last_error))
            raise SourceError(f"nba_stats: could not fetch {season} logs: {last_error}")

        cleaned = self.normalise(frame)
        latency = int((time.monotonic() - started) * 1000)
        record_source_status(SOURCE_NAME, f"game_logs:{season}", status="ok",
                             rows=len(cleaned), latency_ms=latency)
        return NbaGameLogs(cleaned, season, datetime.now(timezone.utc), len(cleaned))

    def fetch_player_advanced(self, season: str | None = None,
                              *, season_type: str = "Regular Season") -> pd.DataFrame:
        """Advanced per-player season stats (minutes, usage, PIE, net rating).

        These are the inputs to the player-impact model.  Only rows with a
        meaningful sample are returned — a player with two games of data would
        otherwise dominate an impact calculation through noise alone.
        """
        season = season or nba_season_for_date()
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SourceError("nba_api is not installed") from exc

        started = time.monotonic()
        try:
            frame = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star=season_type,
                measure_type_detailed_defense="Advanced", timeout=self.timeout,
            ).get_data_frames()[0]
        except Exception as exc:
            record_source_status(SOURCE_NAME, f"player_advanced:{season}", status="error",
                                 message=str(exc))
            raise SourceError(f"nba_stats: player stats for {season} unavailable: {exc}") from exc

        record_source_status(SOURCE_NAME, f"player_advanced:{season}", status="ok",
                             rows=len(frame), latency_ms=int((time.monotonic() - started) * 1000))
        return frame

    @staticmethod
    def normalise(frame: pd.DataFrame) -> pd.DataFrame:
        """Canonicalise ids, drop in-progress games, resolve team identity."""
        if frame is None or frame.empty:
            return pd.DataFrame()

        df = frame.copy()
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.strip().str.zfill(10)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df[df["GAME_DATE"].notna()]

        df["team_uid"] = df["TEAM_ID"].map(
            lambda v: (lambda abbr: f"nba:{abbr}" if abbr else None)(resolve_nba_team(v))
        )
        df = df[df["team_uid"].notna()]

        df["is_home"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False).astype(int)

        finished = (
            df["WL"].astype(str).str.upper().isin(["W", "L"])
            & (pd.to_numeric(df.get("MIN"), errors="coerce").fillna(0) >= 200)
        )
        complete = finished.groupby(df["GAME_ID"]).transform("sum") == 2
        df = df[complete]

        pairs = df.groupby("GAME_ID")["is_home"].agg(["size", "sum"])
        usable = pairs[(pairs["size"] == 2) & (pairs["sum"] == 1)].index
        df = df[df["GAME_ID"].isin(usable)]

        return df.sort_values(["GAME_DATE", "GAME_ID", "is_home"]).reset_index(drop=True)
