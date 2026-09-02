"""Refresh orchestration.

One place that runs the whole ingestion chain:

    FETCH -> VALIDATE -> NORMALISE -> STORE -> record freshness

Every stage records what it attempted and whether it worked, so a failure is
visible on the status page instead of silently leaving yesterday's data in
place.  Individual stages never abort the run: a dead odds feed must not stop
box scores from updating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from ..config import SOCCER_LEAGUES, nba_season_for_date, settings, soccer_season_for_date
from ..db.connection import init_db, query_df
from ..db.repository import (
    ensure_leagues,
    ensure_nba_teams,
    game_uid_nba,
    load_games,
    nba_team_uid,
    now_iso,
    record_source_status,
    soccer_team_uid,
    upsert_games,
    upsert_nba_box,
    upsert_odds,
    upsert_rows,
)
from ..db.validation import validate_games, validate_nba_box, validate_odds
from ..logging_setup import get_logger
from ..sources.base import SourceError
from ..sources.espn_nba import EspnNbaSource
from ..sources.nba_stats import NbaStatsSource
from ..sources.odds_api import SPORT_KEYS, OddsApiSource
from .ingest_soccer import ingest_soccer

log = get_logger(__name__)

BOX_FIELDS = {
    "MIN": "min", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a",
    "FTM": "ftm", "FTA": "fta", "OREB": "oreb", "DREB": "dreb", "REB": "reb",
    "AST": "ast", "STL": "stl", "BLK": "blk", "TOV": "tov", "PF": "pf",
    "PTS": "pts", "PLUS_MINUS": "plus_minus",
}


@dataclass
class StageResult:
    stage: str
    ok: bool
    rows: int = 0
    detail: str | None = None
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "ok": self.ok, "rows": self.rows,
                "detail": self.detail, "skipped": self.skipped}


@dataclass
class RefreshReport:
    started_at: str
    stages: list[StageResult] = field(default_factory=list)

    def add(self, result: StageResult) -> None:
        self.stages.append(result)
        level = log.info if result.ok else log.warning
        level("refresh stage", extra=result.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": now_iso(),
            "ok": all(s.ok or s.skipped for s in self.stages),
            "stages": [s.to_dict() for s in self.stages],
        }


# --------------------------------------------------------------------------
# NBA
# --------------------------------------------------------------------------

def refresh_nba_results(season: str | None = None, *, strict: bool = False) -> StageResult:
    """Pull completed NBA games for a season and store box scores."""
    season = season or nba_season_for_date()
    source = NbaStatsSource()
    try:
        logs = source.fetch_season_logs(season)
    except SourceError as exc:
        return StageResult("nba_results", False, detail=str(exc))

    frame = logs.frame
    if frame.empty:
        return StageResult("nba_results", True, 0,
                           detail=f"no completed games yet for {season}")

    retrieved = now_iso()
    games: list[dict[str, Any]] = []
    box: list[dict[str, Any]] = []

    for game_id, group in frame.groupby("GAME_ID"):
        home = group[group["is_home"] == 1]
        away = group[group["is_home"] == 0]
        if len(home) != 1 or len(away) != 1:
            continue
        home_row, away_row = home.iloc[0], away.iloc[0]
        game_uid = game_uid_nba(game_id)
        games.append(
            {
                "game_uid": game_uid, "sport": "nba", "league_id": "NBA", "season": season,
                "game_date": home_row["GAME_DATE"].strftime("%Y-%m-%d"),
                "kickoff_utc": None, "status": "final",
                "home_team_uid": home_row["team_uid"], "away_team_uid": away_row["team_uid"],
                "home_score": float(home_row["PTS"]), "away_score": float(away_row["PTS"]),
                "neutral_site": 0, "venue": None,
                "source": "nba_stats", "retrieved_at": retrieved,
            }
        )
        for row, opponent in ((home_row, away_row), (away_row, home_row)):
            record = {
                "game_uid": game_uid, "team_uid": row["team_uid"],
                "opp_uid": opponent["team_uid"], "is_home": int(row["is_home"]),
                "won": 1 if str(row["WL"]).upper() == "W" else 0,
                "source": "nba_stats", "retrieved_at": retrieved,
            }
            for source_col, column in BOX_FIELDS.items():
                value = row.get(source_col)
                record[column] = float(value) if pd.notna(value) else None
            box.append(record)

    if not games:
        return StageResult("nba_results", True, 0, detail="nothing new to store")

    games_report = validate_games(pd.DataFrame(games), dataset="nba_refresh")
    games_report.log()
    games_report.persist()
    box_report = validate_nba_box(pd.DataFrame(box))
    box_report.log()
    box_report.persist()
    if strict:
        games_report.raise_if_critical()
        box_report.raise_if_critical()
    if not games_report.ok or not box_report.ok:
        return StageResult("nba_results", False,
                           detail="validation failed; nothing written")

    upsert_games(games)
    upsert_nba_box(box)
    return StageResult("nba_results", True, len(games), detail=f"season {season}")


def refresh_nba_schedule(days_ahead: int = 7, days_back: int = 1) -> StageResult:
    """Store upcoming NBA fixtures so predictions have something to attach to."""
    source = EspnNbaSource()
    today = datetime.now(timezone.utc).date()
    days = [today + timedelta(days=offset) for offset in range(-days_back, days_ahead + 1)]

    try:
        events = source.fetch_schedule_range(days)
    except SourceError as exc:
        return StageResult("nba_schedule", False, detail=str(exc))
    if not events:
        return StageResult("nba_schedule", True, 0, detail="no games scheduled in window")

    retrieved = now_iso()
    season = nba_season_for_date()
    rows: list[dict[str, Any]] = []
    for event in events:
        kickoff = event.get("kickoff_utc")
        game_date = (pd.Timestamp(kickoff).tz_convert("UTC").strftime("%Y-%m-%d")
                     if kickoff else str(today))
        rows.append(
            {
                "game_uid": f"nba:espn:{event['espn_event_id']}",
                "sport": "nba", "league_id": "NBA", "season": season,
                "game_date": game_date, "kickoff_utc": kickoff,
                "status": event["status"],
                "home_team_uid": nba_team_uid(event["home_abbr"]),
                "away_team_uid": nba_team_uid(event["away_abbr"]),
                "home_score": event.get("home_score"), "away_score": event.get("away_score"),
                "neutral_site": event.get("neutral_site", 0), "venue": event.get("venue"),
                "source": "espn_nba", "retrieved_at": retrieved,
            }
        )

    rows = [r for r in rows if r["home_team_uid"] and r["away_team_uid"]]
    # Never overwrite an authoritative stats.nba.com result with an ESPN row.
    existing = query_df(
        "SELECT game_date, home_team_uid, away_team_uid FROM games "
        "WHERE sport='nba' AND source='nba_stats'"
    )
    known = {
        (r["game_date"], r["home_team_uid"], r["away_team_uid"])
        for _, r in existing.iterrows()
    } if not existing.empty else set()
    rows = [r for r in rows
            if (r["game_date"], r["home_team_uid"], r["away_team_uid"]) not in known]

    if not rows:
        return StageResult("nba_schedule", True, 0, detail="all fixtures already known")
    upsert_games(rows)
    return StageResult("nba_schedule", True, len(rows))


def refresh_nba_injuries() -> StageResult:
    """Store the current NBA availability picture with its as-of timestamp."""
    source = EspnNbaSource()
    try:
        records, retrieved = source.fetch_injuries(force=True)
    except SourceError as exc:
        return StageResult("nba_injuries", False, detail=str(exc))
    if not records:
        return StageResult("nba_injuries", True, 0, detail="no injuries reported")

    retrieved_iso = retrieved.isoformat(timespec="seconds")
    rows = [
        {
            "player_uid": f"nba:espn:{record.espn_athlete_id or record.player_name}",
            "team_uid": nba_team_uid(record.team_abbr) if record.team_abbr else None,
            "sport": "nba", "status": record.status, "detail": record.detail,
            "expected_return": record.expected_return, "as_of": record.as_of,
            "source": "espn_nba", "retrieved_at": retrieved_iso,
        }
        for record in records
    ]
    upsert_rows("player_status", rows)

    players = [
        {
            "player_uid": row["player_uid"], "sport": "nba",
            "full_name": record.player_name, "position": record.position,
            "team_uid": row["team_uid"], "external_ids": None, "active": 1,
            "source": "espn_nba", "retrieved_at": retrieved_iso,
        }
        for row, record in zip(rows, records)
    ]
    upsert_rows("players", players, conflict_columns=["player_uid"])
    return StageResult("nba_injuries", True, len(rows))


def refresh_nba_player_stats(season: str | None = None) -> StageResult:
    """Cache advanced player stats used by the impact model."""
    from ..config import previous_nba_season

    season = season or nba_season_for_date()
    # Before a season tips off there are no stats for it at all, and the API
    # answers with an empty frame rather than an error.  Walk back until a
    # season with real data is found so the impact model is never left blind.
    frame = pd.DataFrame()
    errors: list[str] = []
    candidate = season
    for _ in range(3):
        try:
            frame = NbaStatsSource().fetch_player_advanced(candidate)
        except SourceError as exc:
            errors.append(str(exc))
            frame = pd.DataFrame()
        if not frame.empty:
            season = candidate
            break
        candidate = previous_nba_season(candidate)

    if frame.empty:
        return StageResult("nba_player_stats", False,
                           detail="; ".join(errors) or "no player stats available")

    path = settings.paths.cache_dir / f"nba_player_advanced_{season}.parquet"
    try:
        frame.to_parquet(path)
    except Exception:  # parquet engine optional
        path = path.with_suffix(".csv")
        frame.to_csv(path, index=False)
    record_source_status("nba_stats", f"player_advanced:{season}", status="ok", rows=len(frame))
    return StageResult("nba_player_stats", True, len(frame), detail=f"season {season}")


def load_cached_player_stats(season: str | None = None) -> pd.DataFrame:
    """Most recent cached advanced player stats, newest season first."""
    season = season or nba_season_for_date()
    from ..config import previous_nba_season

    for candidate in (season, previous_nba_season(season),
                      previous_nba_season(previous_nba_season(season))):
        for suffix in (".parquet", ".csv"):
            path = settings.paths.cache_dir / f"nba_player_advanced_{candidate}{suffix}"
            if path.exists():
                frame = (pd.read_parquet(path) if suffix == ".parquet"
                         else pd.read_csv(path))
                frame.attrs["season"] = candidate
                return frame
    return pd.DataFrame()


# --------------------------------------------------------------------------
# Odds
# --------------------------------------------------------------------------

def refresh_odds(sports: Sequence[str] | None = None, *, markets: Sequence[str] = ("h2h",),
                 create_missing_games: bool = True) -> StageResult:
    """Poll live prices for each sport and store them as timestamped snapshots."""
    sports = list(sports or ["nba"])
    source = OddsApiSource()
    total = 0
    problems: list[str] = []

    for sport in sports:
        try:
            quotes, _ = source.fetch_odds(sport, markets=markets, force=True)
        except SourceError as exc:
            problems.append(f"{sport}: {exc}")
            continue

        rows, unmatched = _map_quotes_to_games(quotes, sport,
                                               create_missing_games=create_missing_games)
        report = validate_odds(rows)
        report.persist()
        rows = [r for r in rows if 1.0 < r["price_decimal"] <= 1000]
        total += upsert_odds(rows)
        if unmatched:
            problems.append(f"{sport}: {unmatched} events could not be matched to a fixture")

    return StageResult("odds", not problems or total > 0, total,
                       detail="; ".join(problems) or None)


def _map_quotes_to_games(quotes: Sequence[Any], sport: str, *,
                         create_missing_games: bool) -> tuple[list[dict[str, Any]], int]:
    """Attach market quotes to canonical fixtures, creating them when needed."""
    is_nba = sport == "nba"
    league_id = "NBA" if is_nba else sport
    resolve = nba_team_uid if is_nba else soccer_team_uid
    season = nba_season_for_date() if is_nba else soccer_season_for_date()

    candidates = load_games("nba" if is_nba else "soccer",
                            league_id=None if is_nba else league_id)
    lookup: dict[tuple[str, str], list[tuple[pd.Timestamp, str]]] = {}
    if not candidates.empty:
        for _, row in candidates.iterrows():
            key = (row["home_team_uid"], row["away_team_uid"])
            lookup.setdefault(key, []).append((pd.Timestamp(row["game_date"]), row["game_uid"]))

    retrieved = now_iso()
    rows: list[dict[str, Any]] = []
    new_games: dict[str, dict[str, Any]] = {}
    unmatched = 0

    for quote in quotes:
        home_uid = resolve(quote.home_name)
        away_uid = resolve(quote.away_name)
        if not home_uid or not away_uid:
            unmatched += 1
            continue
        commence = pd.Timestamp(quote.commence_time) if quote.commence_time else None
        commence_date = commence.tz_convert("UTC").normalize() if commence is not None else None

        game_uid = None
        for game_date, candidate_uid in lookup.get((home_uid, away_uid), []):
            if commence_date is None or abs((game_date.tz_localize("UTC") if game_date.tzinfo is None
                                             else game_date) - commence_date).days <= 1:
                game_uid = candidate_uid
                break

        if game_uid is None:
            if not create_missing_games or commence_date is None:
                unmatched += 1
                continue
            date_iso = commence_date.strftime("%Y-%m-%d")
            game_uid = (f"nba:odds:{quote.event_id}" if is_nba
                        else f"soccer:{league_id}:{date_iso}:"
                             f"{home_uid.split(':')[-1]}-vs-{away_uid.split(':')[-1]}")
            new_games[game_uid] = {
                "game_uid": game_uid, "sport": "nba" if is_nba else "soccer",
                "league_id": league_id, "season": season, "game_date": date_iso,
                "kickoff_utc": quote.commence_time, "status": "scheduled",
                "home_team_uid": home_uid, "away_team_uid": away_uid,
                "home_score": None, "away_score": None, "neutral_site": 0, "venue": None,
                "source": "odds_api", "retrieved_at": retrieved,
            }

        rows.append(
            {
                "game_uid": game_uid, "sport": "nba" if is_nba else "soccer",
                "market": "1x2" if (not is_nba and quote.market == "h2h") else quote.market,
                "selection": quote.selection, "bookmaker": quote.bookmaker,
                "price_decimal": quote.price_decimal, "captured_at": quote.captured_at,
                "book_updated": quote.book_updated, "is_closing": 0, "source": "odds_api",
            }
        )

    if new_games:
        from ..db.repository import ensure_soccer_teams

        if not is_nba:
            ensure_soccer_teams(
                [q.home_name for q in quotes] + [q.away_name for q in quotes]
            )
        upsert_games(list(new_games.values()))
    return rows, unmatched


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def refresh_all(*, sports: Sequence[str] | None = None, include_odds: bool = True,
                include_match_detail: bool = True,
                soccer_leagues: Sequence[str] | None = None) -> RefreshReport:
    """Run the full refresh chain, continuing past individual failures."""
    init_db()
    ensure_leagues()
    ensure_nba_teams()

    sports = list(sports or ["nba", "soccer"])
    report = RefreshReport(started_at=now_iso())

    if "nba" in sports:
        report.add(refresh_nba_results())
        report.add(refresh_nba_schedule())
        report.add(refresh_nba_injuries())
        report.add(refresh_nba_player_stats())

    if "soccer" in sports:
        leagues = list(soccer_leagues or SOCCER_LEAGUES.keys())
        season = soccer_season_for_date()
        try:
            results = ingest_soccer(leagues, [season], force=True)
            rows = sum(r.games for r in results)
            errors = [e for r in results for e in r.errors]
            report.add(StageResult("soccer_results", True, rows,
                                   detail="; ".join(errors[:3]) or None))
        except Exception as exc:  # pragma: no cover - defensive
            report.add(StageResult("soccer_results", False, detail=str(exc)))

        # Match detail is the only feed that changes *during* a fixture, so it
        # is re-read for matches plausibly in progress and left alone otherwise.
        # It is optional on purpose: a match-centre feed failing must not stop
        # the results and odds the betting side depends on.
        if include_match_detail:
            try:
                from .ingest_match_detail import refresh_live_matches

                detail = refresh_live_matches(league_ids=leagues)
                report.add(StageResult("match_detail", not detail.errors,
                                       detail.events_written,
                                       detail="; ".join(detail.errors[:3]) or None))
            except Exception as exc:  # pragma: no cover - defensive
                report.add(StageResult("match_detail", False, detail=str(exc)))

    if include_odds:
        odds_sports = ["nba"] if "nba" in sports else []
        if "soccer" in sports:
            odds_sports += [
                league for league in (soccer_leagues or SOCCER_LEAGUES.keys())
                if league in SPORT_KEYS
            ][:3]  # bounded: the odds API quota is a shared resource
        if odds_sports:
            report.add(refresh_odds(odds_sports))
        else:
            report.add(StageResult("odds", True, 0, skipped=True, detail="no sports selected"))

    return report
