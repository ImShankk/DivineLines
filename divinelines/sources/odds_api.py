"""The-Odds-API adapter — live market prices across many bookmakers.

Differences from the original single-book implementation:

* every bookmaker's price is kept, not just the first one that parses;
* prices are stored as **decimal** odds internally (American is a display
  format, and mixing the two is how sign errors get into EV maths);
* each poll is written as a timestamped *snapshot*, which is what makes line
  movement and closing-line value possible;
* the request budget is bounded and the remaining quota is tracked, because
  the free tier is a finite resource shared with the rest of the platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from ..config import FRESHNESS_TTL, settings
from ..logging_setup import get_logger
from .base import HttpSource, SourceError

log = get_logger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

#: Sport keys used by the platform.  Soccer competitions map onto the
#: platform's league ids so odds join to fixtures cleanly.
SPORT_KEYS: dict[str, str] = {
    "nba": "basketball_nba",
    "ENG_PL": "soccer_epl",
    "ESP_LL": "soccer_spain_la_liga",
    "ITA_SA": "soccer_italy_serie_a",
    "GER_BL": "soccer_germany_bundesliga",
    "FRA_L1": "soccer_france_ligue_one",
    "NED_ED": "soccer_netherlands_eredivisie",
    "POR_PL": "soccer_portugal_primeira_liga",
    "ENG_CH": "soccer_efl_champ",
}


@dataclass
class MarketQuote:
    event_id: str
    sport_key: str
    commence_time: str
    home_name: str
    away_name: str
    bookmaker: str
    market: str          # 'h2h' | 'totals'
    selection: str       # 'home' | 'draw' | 'away' | 'over_2.5' | 'under_2.5'
    price_decimal: float
    book_updated: str | None
    captured_at: str


class OddsApiSource(HttpSource):
    name = "odds_api"
    cache_ttl = FRESHNESS_TTL["odds"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._calls_this_run = 0
        self.quota_remaining: int | None = None
        self.quota_used: int | None = None

    # ------------------------------------------------------------- internals

    def _require_key(self) -> str:
        key = settings.sources.odds_api_key
        if not key:
            raise SourceError(
                "odds_api: ODDS_API_KEY is not set — add it to .env (see .env.example)"
            )
        return key

    def _budget_check(self) -> None:
        if self._calls_this_run >= settings.sources.odds_api_max_calls_per_run:
            raise SourceError(
                f"odds_api: request budget of {settings.sources.odds_api_max_calls_per_run} "
                "calls exhausted for this run (raise DL_ODDS_MAX_CALLS to allow more)"
            )

    def _track_quota(self) -> None:
        headers = self.last_response_headers
        remaining = headers.get("x-requests-remaining")
        used = headers.get("x-requests-used")
        if remaining is not None:
            try:
                self.quota_remaining = int(float(remaining))
            except ValueError:
                pass
        if used is not None:
            try:
                self.quota_used = int(float(used))
            except ValueError:
                pass
        if self.quota_remaining is not None and self.quota_remaining < 50:
            log.warning("odds API quota running low", extra={"remaining": self.quota_remaining})

    # ---------------------------------------------------------------- public

    def list_sports(self, *, force: bool = False) -> list[dict[str, Any]]:
        self._budget_check()
        result = self.fetch_json(
            f"{BASE_URL}/sports", dataset="sports",
            params={"apiKey": self._require_key()},
            ttl=FRESHNESS_TTL["reference"], force=force,
        )
        if not result.from_cache:
            self._calls_this_run += 1
            self._track_quota()
        return result.data if isinstance(result.data, list) else []

    def fetch_odds(
        self,
        sport: str,
        *,
        markets: Iterable[str] = ("h2h",),
        force: bool = False,
    ) -> tuple[list[MarketQuote], datetime]:
        """Current prices for every upcoming event in ``sport``.

        ``sport`` is a platform key (``'nba'``, ``'ENG_PL'``) or a raw
        The-Odds-API sport key.
        """
        sport_key = SPORT_KEYS.get(sport, sport)
        market_param = ",".join(markets)
        self._budget_check()

        result = self.fetch_json(
            f"{BASE_URL}/sports/{sport_key}/odds",
            dataset=f"odds:{sport_key}:{market_param}",
            params={
                "apiKey": self._require_key(),
                "regions": settings.sources.odds_regions,
                "markets": market_param,
                "oddsFormat": settings.sources.odds_format,
            },
            ttl=FRESHNESS_TTL["odds"],
            force=force,
        )
        if not result.from_cache:
            self._calls_this_run += 1
            self._track_quota()

        payload = result.data
        if isinstance(payload, dict) and "message" in payload:
            raise SourceError(f"odds_api: {payload['message']}")
        if not isinstance(payload, list):
            raise SourceError("odds_api: unexpected payload shape")

        captured_at = result.retrieved_at.astimezone(timezone.utc).isoformat(timespec="seconds")
        quotes: list[MarketQuote] = []
        for event in payload:
            quotes.extend(self._parse_event(event, sport_key, captured_at))

        log.info(
            "fetched odds",
            extra={"sport": sport_key, "events": len(payload), "quotes": len(quotes),
                   "cached": result.from_cache, "quota_remaining": self.quota_remaining},
        )
        return quotes, result.retrieved_at

    # ----------------------------------------------------------------- parse

    @staticmethod
    def _parse_event(event: dict[str, Any], sport_key: str, captured_at: str) -> list[MarketQuote]:
        home = event.get("home_team")
        away = event.get("away_team")
        if not home or not away:
            return []

        quotes: list[MarketQuote] = []
        for bookmaker in event.get("bookmakers", []) or []:
            book_title = bookmaker.get("title") or bookmaker.get("key") or "unknown"
            book_updated = bookmaker.get("last_update")
            for market in bookmaker.get("markets", []) or []:
                market_key = market.get("key")
                for outcome in market.get("outcomes", []) or []:
                    selection = OddsApiSource._selection_label(
                        market_key, outcome, home, away
                    )
                    price = outcome.get("price")
                    if selection is None or price is None:
                        continue
                    try:
                        price = float(price)
                    except (TypeError, ValueError):
                        continue
                    if price <= 1.0:
                        # Decimal odds must exceed 1.0; anything else means the
                        # response was not in the format we requested.
                        continue
                    quotes.append(
                        MarketQuote(
                            event_id=str(event.get("id")),
                            sport_key=sport_key,
                            commence_time=event.get("commence_time"),
                            home_name=home,
                            away_name=away,
                            bookmaker=book_title,
                            market="h2h" if market_key == "h2h" else market_key,
                            selection=selection,
                            price_decimal=price,
                            book_updated=book_updated,
                            captured_at=captured_at,
                        )
                    )
        return quotes

    @staticmethod
    def _selection_label(market_key: str | None, outcome: dict[str, Any],
                         home: str, away: str) -> str | None:
        name = outcome.get("name")
        if market_key == "h2h":
            if name == home:
                return "home"
            if name == away:
                return "away"
            if str(name).lower() == "draw":
                return "draw"
            return None
        if market_key == "totals":
            point = outcome.get("point")
            if point is None or name not in ("Over", "Under"):
                return None
            return f"{str(name).lower()}_{point}"
        return None
