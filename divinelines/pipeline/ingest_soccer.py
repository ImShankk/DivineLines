"""Soccer ingestion: football-data.co.uk -> canonical schema.

Writes three things per season file: fixtures/results, match statistics, and
bookmaker price snapshots (pre-match and closing).

A note on timestamps, because honesty about provenance matters more than
convenience: football-data publishes two price columns per market — a
pre-match figure and a closing figure — without the exact moment either was
captured.  The platform therefore stores the pre-match snapshot at local
midnight of match day and the closing snapshot at kick-off, and marks the
latter with ``is_closing = 1``.  Every backtest keys off ``is_closing``, never
off the synthetic capture time, so no analysis depends on a timestamp the
source did not actually provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from ..config import SOCCER_LEAGUES, SOCCER_SEASONS, settings
from ..db.repository import (
    ensure_leagues,
    ensure_soccer_teams,
    game_uid_soccer,
    now_iso,
    record_source_status,
    soccer_team_uid,
    upsert_games,
    upsert_odds,
    upsert_soccer_stats,
)
from ..db.connection import init_db
from ..db.validation import validate_games, validate_odds
from ..logging_setup import get_logger
from ..sources.base import SourceError
from ..sources.football_data import FootballDataSource, available_seasons

log = get_logger(__name__)

SOURCE = "football_data_uk"
SELECTIONS = ("home", "draw", "away")


@dataclass
class IngestResult:
    league_id: str
    season: str
    games: int = 0
    stats: int = 0
    odds: int = 0
    errors: list[str] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "league_id": self.league_id, "season": self.season, "games": self.games,
            "stats": self.stats, "odds": self.odds, "errors": self.errors,
            "from_cache": self.from_cache,
        }


def _kickoff(row: pd.Series) -> str | None:
    time_value = row.get("time")
    if not time_value or str(time_value) in ("nan", "None", ""):
        return None
    try:
        return f"{row['date'].strftime('%Y-%m-%d')}T{str(time_value)[:5]}:00"
    except (AttributeError, ValueError):
        return None


def _odds_rows(row: pd.Series, game_uid: str, kickoff: str | None) -> list[dict[str, Any]]:
    date_iso = row["date"].strftime("%Y-%m-%d")
    pre_at = f"{date_iso}T00:00:00+00:00"
    close_at = (f"{kickoff}+00:00" if kickoff else f"{date_iso}T23:59:00+00:00")

    rows: list[dict[str, Any]] = []
    for column in row.index:
        if not column.startswith("odds_"):
            continue
        price = row[column]
        if pd.isna(price) or float(price) <= 1.0:
            continue
        body = column[len("odds_"):]
        if body.endswith("_close"):
            body, is_closing = body[: -len("_close")], 1
        elif body.endswith("_open"):
            body, is_closing = body[: -len("_open")], 0
        else:
            continue

        selection = next((s for s in SELECTIONS if body.endswith(f"_{s}")), None)
        if selection is not None:
            bookmaker = body[: -(len(selection) + 1)]
            market = "1x2"
        elif "_over_" in body or "_under_" in body:
            marker = "_over_" if "_over_" in body else "_under_"
            bookmaker = body.split(marker)[0]
            selection = body[len(bookmaker) + 1:]
            market = "totals"
        else:
            continue

        rows.append(
            {
                "game_uid": game_uid, "sport": "soccer", "market": market,
                "selection": selection, "bookmaker": bookmaker,
                "price_decimal": float(price),
                "captured_at": close_at if is_closing else pre_at,
                "book_updated": None, "is_closing": is_closing, "source": SOURCE,
            }
        )
    return rows


def ingest_league_season(league_id: str, season: str, *, force: bool = False,
                         strict: bool = False) -> IngestResult:
    """Ingest one division-season file."""
    result = IngestResult(league_id=league_id, season=season)
    source = FootballDataSource()

    try:
        data = source.fetch_season(league_id, season, force=force)
    except SourceError as exc:
        log.warning("soccer season unavailable",
                    extra={"league": league_id, "season": season, "error": str(exc)})
        result.errors.append(str(exc))
        return result

    result.from_cache = data.from_cache
    frame = data.matches
    if frame.empty:
        result.errors.append("empty season file")
        return result

    country = SOCCER_LEAGUES[league_id]["country"]
    ensure_soccer_teams(
        pd.concat([frame["home_name"], frame["away_name"]]).unique().tolist(), country
    )

    retrieved = now_iso()
    games: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    odds: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        home_uid = soccer_team_uid(row["home_name"])
        away_uid = soccer_team_uid(row["away_name"])
        if not home_uid or not away_uid or home_uid == away_uid:
            result.errors.append(f"unresolved teams: {row['home_name']} vs {row['away_name']}")
            continue

        date_iso = row["date"].strftime("%Y-%m-%d")
        game_uid = game_uid_soccer(league_id, date_iso, home_uid, away_uid)
        kickoff = _kickoff(row)
        played = pd.notna(row.get("home_score")) and pd.notna(row.get("away_score"))

        games.append(
            {
                "game_uid": game_uid, "sport": "soccer", "league_id": league_id,
                "season": season, "game_date": date_iso, "kickoff_utc": kickoff,
                "status": "final" if played else "scheduled",
                "home_team_uid": home_uid, "away_team_uid": away_uid,
                "home_score": float(row["home_score"]) if played else None,
                "away_score": float(row["away_score"]) if played else None,
                "neutral_site": 0, "venue": None,
                "source": SOURCE, "retrieved_at": retrieved,
            }
        )

        stat_row = {
            "game_uid": game_uid,
            "ht_home": None if pd.isna(row.get("ht_home")) else int(row["ht_home"]),
            "ht_away": None if pd.isna(row.get("ht_away")) else int(row["ht_away"]),
            "referee": row.get("referee") if isinstance(row.get("referee"), str) else None,
            "source": SOURCE, "retrieved_at": retrieved,
        }
        for column in ("home_shots", "away_shots", "home_sot", "away_sot",
                       "home_corners", "away_corners", "home_fouls", "away_fouls",
                       "home_yellow", "away_yellow", "home_red", "away_red"):
            value = row.get(column)
            stat_row[column] = None if pd.isna(value) else int(value)
        if any(v is not None for k, v in stat_row.items()
               if k not in ("game_uid", "source", "retrieved_at")):
            stats.append(stat_row)

        odds.extend(_odds_rows(row, game_uid, kickoff))

    games_df = pd.DataFrame(games)
    report = validate_games(games_df, dataset=f"soccer:{league_id}:{season}")
    report.log()
    report.persist()
    if strict:
        report.raise_if_critical()
    if not report.ok:
        result.errors.extend(f"{i.code}: {i.detail}" for i in report.critical)
        return result

    odds_report = validate_odds(odds)
    odds_report.persist()
    odds = [
        row for row in odds
        if 1.0 < row["price_decimal"] <= 1000
    ]

    ensure_leagues()
    result.games = upsert_games(games)
    result.stats = upsert_soccer_stats(stats)
    result.odds = upsert_odds(odds)

    record_source_status(SOURCE, f"{league_id}:{season}", status="ok", rows=result.games)
    log.info("ingested soccer season", extra=result.to_dict())
    return result


def ingest_soccer(leagues: Iterable[str] | None = None, seasons: Iterable[str] | None = None,
                  *, force: bool = False) -> list[IngestResult]:
    """Ingest every requested league-season, continuing past individual failures."""
    init_db()
    ensure_leagues()
    leagues = list(leagues or SOCCER_LEAGUES.keys())
    seasons = available_seasons(list(seasons or SOCCER_SEASONS))

    results: list[IngestResult] = []
    for league_id in leagues:
        for season in seasons:
            results.append(ingest_league_season(league_id, season, force=force))
    total_games = sum(r.games for r in results)
    total_odds = sum(r.odds for r in results)
    log.info("soccer ingest complete",
             extra={"league_seasons": len(results), "games": total_games, "odds": total_odds})
    return results


if __name__ == "__main__":  # pragma: no cover - operational entry point
    for item in ingest_soccer():
        print(item.to_dict())
