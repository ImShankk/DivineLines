"""ESPN historical odds adapter.

This closes the single biggest gap in V2: NBA betting performance was
unmeasurable because no historical NBA prices existed anywhere in the platform.
ESPN's core API publishes, per completed event, an **opening and a closing
price** from several bookmakers, with decimal odds included — enough to
reconstruct entry price, closing price and therefore CLV.

Two things this adapter is careful about:

* **In-play providers are excluded.** ESPN lists feeds such as
  "ESPN Bet - Live Odds" whose prices are quoted *during* the game. Ingesting
  those would put post-tip-off information into a pre-game backtest, which is
  exactly the class of leak the platform exists to avoid.
* **No capture timestamps are invented.** ESPN says "this was the open" and
  "this was the close" without saying when either was recorded. Rows are stored
  with the event start as a nominal ``captured_at`` and the truth carried in
  ``phase``; nothing downstream sorts these rows by time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from ..config import FRESHNESS_TTL
from ..identity import resolve_nba_team
from ..logging_setup import get_logger
from .base import HttpSource, SourceError

log = get_logger(__name__)

SITE_API = "https://site.api.espn.com/apis/site/v2/sports"
CORE_API = "https://sports.core.api.espn.com/v2/sports"

#: ESPN sport paths keyed by the platform's own sport/league identifiers.
SPORT_PATHS: dict[str, tuple[str, str]] = {
    "nba": ("basketball", "nba"),
    "ENG_PL": ("soccer", "eng.1"),
    "ENG_CH": ("soccer", "eng.2"),
    "ESP_LL": ("soccer", "esp.1"),
    "ITA_SA": ("soccer", "ita.1"),
    "GER_BL": ("soccer", "ger.1"),
    "FRA_L1": ("soccer", "fra.1"),
    "NED_ED": ("soccer", "ned.1"),
    "POR_PL": ("soccer", "por.1"),
}

#: Anything matching this is quoted in-play and must never enter a pre-game
#: dataset. ESPN's naming is inconsistent, hence a pattern rather than a list.
_LIVE_PROVIDER = re.compile(r"live\s*odds|in-?play|live\s*betting", re.IGNORECASE)


@dataclass
class HistoricalQuote:
    espn_event_id: str
    bookmaker: str
    market: str          # 'h2h'
    selection: str       # 'home' | 'away'
    phase: str           # 'open' | 'close'
    price_decimal: float


@dataclass
class EspnEvent:
    espn_event_id: str
    date_iso: str
    start_utc: str | None
    status: str
    home_abbr: str | None
    away_abbr: str | None
    home_name: str
    away_name: str


class EspnOddsSource(HttpSource):
    name = "espn_odds"
    cache_ttl = FRESHNESS_TTL["reference"]   # completed games never change
    #: ESPN rejects browser-spoofing agents; this one identifies us honestly.
    user_agent = (
        "Mozilla/5.0 (compatible; DivineLines/3.0; research; "
        "+https://github.com/ImShankk/DivineLines)"
    )
    min_interval = 0.4

    def _paths(self, sport: str) -> tuple[str, str]:
        if sport not in SPORT_PATHS:
            raise SourceError(f"espn_odds: no ESPN path configured for '{sport}'")
        return SPORT_PATHS[sport]

    # ------------------------------------------------------------- schedule

    def fetch_events(self, sport: str, day: str, *, force: bool = False) -> list[EspnEvent]:
        """Events on one calendar day. ``day`` is ``YYYYMMDD``."""
        group, league = self._paths(sport)
        result = self.fetch_json(
            f"{SITE_API}/{group}/{league}/scoreboard",
            dataset=f"scoreboard:{league}:{day}",
            status_dataset=f"scoreboard:{league}",
            params={"dates": day, "limit": 200},
            ttl=self.cache_ttl, force=force,
        )
        payload = result.data if isinstance(result.data, dict) else {}
        events: list[EspnEvent] = []
        for event in payload.get("events", []) or []:
            competitions = event.get("competitions") or []
            if not competitions:
                continue
            competition = competitions[0]
            home = away = None
            for competitor in competition.get("competitors", []) or []:
                if competitor.get("homeAway") == "home":
                    home = competitor
                else:
                    away = competitor
            if not home or not away:
                continue
            start = event.get("date")
            events.append(
                EspnEvent(
                    espn_event_id=str(event.get("id")),
                    date_iso=(start or "")[:10],
                    start_utc=start,
                    status=((competition.get("status") or event.get("status") or {})
                            .get("type", {}).get("name", "")),
                    home_abbr=resolve_nba_team((home.get("team") or {}).get("displayName"))
                    if sport == "nba" else None,
                    away_abbr=resolve_nba_team((away.get("team") or {}).get("displayName"))
                    if sport == "nba" else None,
                    home_name=(home.get("team") or {}).get("displayName", ""),
                    away_name=(away.get("team") or {}).get("displayName", ""),
                )
            )
        return events

    # ----------------------------------------------------------------- odds

    def fetch_event_odds(self, sport: str, espn_event_id: str, *,
                         force: bool = False) -> list[HistoricalQuote]:
        """Opening and closing moneyline prices from every non-live provider."""
        group, league = self._paths(sport)
        url = (f"{CORE_API}/{group}/leagues/{league}/events/{espn_event_id}"
               f"/competitions/{espn_event_id}/odds")
        result = self.fetch_json(url, dataset=f"odds:{league}:{espn_event_id}",
                                 status_dataset=f"odds:{league}",
                                 ttl=self.cache_ttl, force=force)
        payload = result.data if isinstance(result.data, dict) else {}

        quotes: list[HistoricalQuote] = []
        for item in payload.get("items", []) or []:
            provider = (item.get("provider") or {}).get("name") or "unknown"
            if _LIVE_PROVIDER.search(provider):
                continue
            for side, selection in (("homeTeamOdds", "home"), ("awayTeamOdds", "away")):
                block = item.get(side) or {}
                for phase in ("open", "close"):
                    price = self._moneyline_decimal(block.get(phase))
                    if price is None:
                        continue
                    quotes.append(
                        HistoricalQuote(
                            espn_event_id=str(espn_event_id), bookmaker=provider,
                            market="h2h", selection=selection, phase=phase,
                            price_decimal=price,
                        )
                    )
        return quotes

    @staticmethod
    def _moneyline_decimal(block: Any) -> float | None:
        if not isinstance(block, dict):
            return None
        moneyline = block.get("moneyLine")
        if not isinstance(moneyline, dict):
            return None
        for key in ("decimal", "value"):
            value = moneyline.get(key)
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            # A decimal price of 1.0 or below is a formatting artefact, not a
            # quote a bettor could take.
            if price > 1.0:
                return price
        return None

    # -------------------------------------------------------------- lineups

    def fetch_lineups(self, sport: str, espn_event_id: str, *,
                      force: bool = False) -> dict[str, Any]:
        """Team rosters for one event: formation, starters, bench, positions."""
        group, league = self._paths(sport)
        result = self.fetch_json(
            f"{SITE_API}/{group}/{league}/summary",
            dataset=f"summary:{league}:{espn_event_id}",
            status_dataset=f"summary:{league}",
            params={"event": espn_event_id},
            ttl=FRESHNESS_TTL["lineups"], force=force,
        )
        payload = result.data if isinstance(result.data, dict) else {}
        return {"rosters": payload.get("rosters") or [], "retrieved_at": result.retrieved_at,
                "from_cache": result.from_cache}
