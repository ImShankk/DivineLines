"""Historical NBA odds backfill.

V2 reported "NBA betting performance is unmeasured" because no historical NBA
prices existed. This walks the schedule day by day, maps ESPN events onto the
platform's canonical games, and stores each bookmaker's opening and closing
moneyline.

The mapping is by ``(date, home_team, away_team)`` rather than by id, because
the canonical games came from stats.nba.com and carry NBA game ids while ESPN
uses its own. Both sides go through the existing identity layer, so "LA
Clippers" and "Los Angeles Clippers" cannot become two different games.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd

from ..db.connection import query_df, write_connection
from ..db.repository import nba_team_uid, now_iso, record_source_status, upsert_odds
from ..db.validation import validate_odds
from ..logging_setup import get_logger
from ..sources.base import SourceError
from ..sources.espn_odds import EspnOddsSource

log = get_logger(__name__)

SOURCE = "espn_odds"


@dataclass
class BackfillReport:
    days_scanned: int = 0
    events_seen: int = 0
    games_matched: int = 0
    games_unmatched: int = 0
    games_with_odds: int = 0
    quotes_written: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "days_scanned": self.days_scanned, "events_seen": self.events_seen,
            "games_matched": self.games_matched, "games_unmatched": self.games_unmatched,
            "games_with_odds": self.games_with_odds, "quotes_written": self.quotes_written,
            "errors": self.errors[:5], "error_count": len(self.errors),
        }


def _canonical_index(sport: str = "nba") -> dict[tuple[str, str, str], str]:
    """``(date, home_uid, away_uid) -> game_uid`` for already-stored games."""
    frame = query_df(
        "SELECT game_uid, game_date, home_team_uid, away_team_uid FROM games WHERE sport = ?",
        (sport,),
    )
    return {
        (str(row["game_date"])[:10], row["home_team_uid"], row["away_team_uid"]): row["game_uid"]
        for _, row in frame.iterrows()
    }


def _game_dates(sport: str, seasons: Iterable[str] | None) -> list[str]:
    clause = ""
    params: list[Any] = [sport]
    if seasons:
        seasons = list(seasons)
        clause = f"AND season IN ({','.join('?' for _ in seasons)})"
        params.extend(seasons)
    frame = query_df(
        f"SELECT DISTINCT game_date FROM games WHERE sport = ? AND status = 'final' {clause} "
        "ORDER BY game_date",
        params,
    )
    return [str(value)[:10] for value in frame["game_date"].tolist()]


def _already_backfilled() -> set[str]:
    """Games that already carry ESPN prices, so a rerun is cheap."""
    frame = query_df(
        "SELECT DISTINCT game_uid FROM odds_snapshots WHERE source = ? AND phase IN ('open','close')",
        (SOURCE,),
    )
    return set(frame["game_uid"].tolist())


def backfill_nba_odds(seasons: Iterable[str] | None = None, *, limit_days: int | None = None,
                      force: bool = False, progress_every: int = 25) -> BackfillReport:
    """Fetch opening/closing NBA moneylines for stored final games."""
    report = BackfillReport()
    source = EspnOddsSource()
    index = _canonical_index("nba")
    if not index:
        raise RuntimeError("no NBA games stored — run `divinelines migrate` first")

    done = set() if force else _already_backfilled()
    dates = _game_dates("nba", seasons)
    if limit_days:
        dates = dates[:limit_days]

    espn_ids: dict[str, str] = {}
    pending: list[tuple[str, str, str | None]] = []   # (game_uid, espn_id, start_utc)

    for day in dates:
        report.days_scanned += 1
        try:
            events = source.fetch_events("nba", day.replace("-", ""))
        except SourceError as exc:
            report.errors.append(f"{day}: {exc}")
            continue

        for event in events:
            report.events_seen += 1
            home_uid = nba_team_uid(event.home_abbr) if event.home_abbr else None
            away_uid = nba_team_uid(event.away_abbr) if event.away_abbr else None
            if not home_uid or not away_uid:
                report.games_unmatched += 1
                continue

            # ESPN's event date is the UTC kick-off, which for a US evening game
            # is the following calendar day. Try both before giving up.
            game_uid = None
            for candidate_date in (day, event.date_iso,
                                   (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")):
                game_uid = index.get((candidate_date, home_uid, away_uid))
                if game_uid:
                    break
            if not game_uid:
                report.games_unmatched += 1
                continue

            report.games_matched += 1
            espn_ids[game_uid] = event.espn_event_id
            if game_uid not in done:
                pending.append((game_uid, event.espn_event_id, event.start_utc))

        if report.days_scanned % progress_every == 0:
            log.info("backfill progress",
                     extra={"days": report.days_scanned, "matched": report.games_matched,
                            "pending": len(pending)})

    _store_espn_ids(espn_ids)

    for position, (game_uid, espn_id, start_utc) in enumerate(pending, start=1):
        try:
            quotes = source.fetch_event_odds("nba", espn_id)
        except SourceError as exc:
            report.errors.append(f"{game_uid}: {exc}")
            continue
        if not quotes:
            continue

        nominal = start_utc or f"{game_uid.split(':')[-1]}"
        rows = [
            {
                "game_uid": game_uid, "sport": "nba", "market": "h2h",
                "selection": quote.selection, "bookmaker": quote.bookmaker,
                "price_decimal": quote.price_decimal,
                # Nominal timestamp: ESPN gives no capture time for open/close.
                # `phase` carries the meaning; nothing sorts these by time.
                "captured_at": nominal,
                "book_updated": None,
                "is_closing": 1 if quote.phase == "close" else 0,
                "phase": quote.phase, "source": SOURCE,
            }
            for quote in quotes
        ]
        report_validation = validate_odds(rows)
        rows = [r for r in rows if 1.0 < r["price_decimal"] <= 1000]
        if not rows:
            continue
        report.quotes_written += upsert_odds(rows)
        report.games_with_odds += 1

        if position % (progress_every * 4) == 0:
            log.info("odds backfill progress",
                     extra={"done": position, "of": len(pending),
                            "quotes": report.quotes_written})

    record_source_status(SOURCE, "nba_historical_odds",
                         status="ok" if report.quotes_written else "degraded",
                         rows=report.quotes_written,
                         message=f"{report.games_with_odds} games priced, "
                                 f"{len(report.errors)} errors")
    log.info("nba odds backfill complete", extra=report.to_dict())
    return report


def _store_espn_ids(mapping: dict[str, str]) -> None:
    """Keep the ESPN id on the game so lineups can reuse the mapping."""
    if not mapping:
        return
    with write_connection() as conn:
        conn.executemany(
            "UPDATE games SET espn_event_id = ? WHERE game_uid = ? AND espn_event_id IS NULL",
            [(espn_id, game_uid) for game_uid, espn_id in mapping.items()],
        )


def backfill_soccer_espn_ids(league_ids: Iterable[str], *, days_back: int = 400
                             ) -> dict[str, int]:
    """Map stored soccer fixtures onto ESPN event ids (needed for lineups)."""
    from ..db.repository import soccer_team_uid

    source = EspnOddsSource()
    counts: dict[str, int] = {}
    today = datetime.now(timezone.utc).date()

    for league_id in league_ids:
        frame = query_df(
            "SELECT game_uid, game_date, home_team_uid, away_team_uid FROM games "
            "WHERE sport='soccer' AND league_id = ? AND game_date >= ? AND espn_event_id IS NULL "
            "ORDER BY game_date DESC",
            (league_id, str(today - timedelta(days=days_back))),
        )
        if frame.empty:
            continue
        index = {
            (str(row["game_date"])[:10], row["home_team_uid"], row["away_team_uid"]): row["game_uid"]
            for _, row in frame.iterrows()
        }
        mapping: dict[str, str] = {}
        for day in sorted({key[0] for key in index}):
            try:
                events = source.fetch_events(league_id, day.replace("-", ""))
            except SourceError as exc:
                log.warning("espn soccer schedule failed",
                            extra={"league": league_id, "day": day, "error": str(exc)})
                continue
            for event in events:
                home_uid = soccer_team_uid(event.home_name)
                away_uid = soccer_team_uid(event.away_name)
                for candidate in (day, event.date_iso):
                    game_uid = index.get((candidate, home_uid, away_uid))
                    if game_uid:
                        mapping[game_uid] = event.espn_event_id
                        break
        _store_espn_ids(mapping)
        counts[league_id] = len(mapping)
        log.info("mapped soccer fixtures to ESPN",
                 extra={"league": league_id, "mapped": len(mapping), "candidates": len(index)})
    return counts
