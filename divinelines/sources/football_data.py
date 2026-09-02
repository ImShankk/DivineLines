"""football-data.co.uk adapter.

Provides historical soccer results, match statistics (shots, cards, corners,
referee) **and** bookmaker prices including closing odds, going back many
seasons.  Closing prices are what make an honest soccer backtest possible: we
can compare a decision-time price against the closing line rather than
assuming a flat vig.

The site publishes one CSV per division per season.  Files for a season in
progress grow over time, so cached copies expire quickly for the current
season and slowly for completed ones.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..config import FRESHNESS_TTL, SOCCER_LEAGUES, soccer_season_for_date
from ..logging_setup import get_logger
from .base import FetchResult, HttpSource, SourceError

log = get_logger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

#: (bookmaker label, prefix, is_closing).  ``Avg``/``Max`` are market
#: aggregates published by the site, not individual books; they are labelled
#: as such so nothing pretends to be a single sportsbook price.
ODDS_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("bet365", "B365", 0),
    ("pinnacle", "PS", 0),
    ("market_avg", "Avg", 0),
    ("market_best", "Max", 0),
    ("bet365", "B365C", 1),
    ("pinnacle", "PSC", 1),
    ("market_avg", "AvgC", 1),
    ("market_best", "MaxC", 1),
)

STAT_COLUMNS = {
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
}


@dataclass
class SoccerSeasonData:
    league_id: str
    season: str
    matches: pd.DataFrame
    retrieved_at: datetime
    from_cache: bool


class FootballDataSource(HttpSource):
    name = "football_data_uk"
    cache_ttl = FRESHNESS_TTL["soccer_results"]
    min_interval = 1.0

    def season_url(self, league_id: str, season: str) -> str:
        code = SOCCER_LEAGUES[league_id]["fd_code"]
        return f"{BASE_URL}/{season}/{code}.csv"

    def fetch_season(self, league_id: str, season: str, *, force: bool = False) -> SoccerSeasonData:
        if league_id not in SOCCER_LEAGUES:
            raise SourceError(f"unknown soccer league '{league_id}'")

        current = soccer_season_for_date()
        # A completed season's file never changes; cache it for a month.
        ttl = self.cache_ttl if season == current else FRESHNESS_TTL["reference"]

        result: FetchResult = self.fetch(
            self.season_url(league_id, season),
            dataset=f"{league_id}:{season}",
            ttl=ttl,
            suffix=".csv",
            force=force,
        )
        frame = self._parse_csv(result.data, league_id, season)
        return SoccerSeasonData(
            league_id=league_id,
            season=season,
            matches=frame,
            retrieved_at=result.retrieved_at,
            from_cache=result.from_cache,
        )

    # ------------------------------------------------------------------ parse

    def _parse_csv(self, payload: bytes, league_id: str, season: str) -> pd.DataFrame:
        text = payload.decode("utf-8-sig", errors="replace")
        raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        raw = raw.dropna(how="all")
        if raw.empty or "HomeTeam" not in raw.columns:
            raise SourceError(f"{self.name}: unexpected CSV layout for {league_id} {season}")

        # The file has to say it is the division we asked for. This is not
        # paranoia: at the start of the 2026-27 season the site had not yet
        # published E0.csv, and the request for it came back carrying EC (the
        # National League). Trusting the URL over the payload put 12 National
        # League fixtures and 9 Primeira Liga fixtures into the store labelled
        # as Premier League and La Liga, which is exactly the kind of silent
        # corruption everything else here is built to prevent.
        expected = SOCCER_LEAGUES[league_id]["fd_code"]
        if "Div" in raw.columns:
            divisions = {str(value).strip() for value in raw["Div"].dropna().unique()}
            if divisions and expected not in divisions:
                raise SourceError(
                    f"{self.name}: {league_id} {season} expected division '{expected}' "
                    f"but the file contains {sorted(divisions)}"
                )
            raw = raw[raw["Div"].astype(str).str.strip() == expected].copy()
            if raw.empty:
                raise SourceError(
                    f"{self.name}: no '{expected}' rows in the {league_id} {season} file"
                )

        # Rows without team names are trailing junk in several season files.
        raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()].copy()

        parsed = pd.DataFrame(
            {
                "league_id": league_id,
                "season": season,
                "home_name": raw["HomeTeam"].astype(str).str.strip(),
                "away_name": raw["AwayTeam"].astype(str).str.strip(),
            }
        )
        parsed["date"] = self._parse_dates(raw["Date"])
        parsed["time"] = raw["Time"].astype(str).str.strip() if "Time" in raw.columns else None

        for src, dest in (("FTHG", "home_score"), ("FTAG", "away_score"),
                          ("HTHG", "ht_home"), ("HTAG", "ht_away")):
            parsed[dest] = pd.to_numeric(raw.get(src), errors="coerce")
        parsed["result"] = raw.get("FTR")
        parsed["referee"] = raw["Referee"].astype(str).str.strip() if "Referee" in raw.columns else None

        for src, dest in STAT_COLUMNS.items():
            parsed[dest] = pd.to_numeric(raw.get(src), errors="coerce")

        for book, prefix, is_closing in ODDS_COLUMNS:
            suffix = "_close" if is_closing else "_open"
            for sel, letter in (("home", "H"), ("draw", "D"), ("away", "A")):
                column = f"{prefix}{letter}"
                parsed[f"odds_{book}_{sel}{suffix}"] = pd.to_numeric(
                    raw.get(column), errors="coerce"
                )

        for label, column in (("over_2.5", "Avg>2.5"), ("under_2.5", "Avg<2.5")):
            parsed[f"odds_market_avg_{label}_open"] = pd.to_numeric(raw.get(column), errors="coerce")

        parsed = parsed[parsed["date"].notna()]
        parsed = parsed.sort_values("date").reset_index(drop=True)
        return parsed

    @staticmethod
    def _parse_dates(series: pd.Series) -> pd.Series:
        """football-data uses dd/mm/yy in old files and dd/mm/yyyy in new ones."""
        text = series.astype(str).str.strip()
        parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
        fallback = pd.to_datetime(text, format="%d/%m/%y", errors="coerce")
        return parsed.fillna(fallback)


def available_seasons(seasons: list[str]) -> list[str]:
    """Filter out season codes that cannot exist yet."""
    current = soccer_season_for_date()
    return [s for s in seasons if s <= current]
