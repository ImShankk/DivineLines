"""Ingest match detail: events, box statistics, player lines, match context.

This reads the same ESPN summary the lineup pipeline already fetches, so for
matches whose payload is on disk it costs nothing extra. What it adds is the
part of that document V3 never unpacked: the play-by-play.

## Timestamps, again

Two clocks are stored per event and they are not interchangeable.

``wallclock_utc`` and ``minute`` say when the event happened *in the match*.
``observed_at`` says when this platform saw it. Replay filters on the first;
leakage checks filter on the second. For a match ingested after full time,
every event shares one ``observed_at`` — which is exactly the truth, and it is
why a historical match cannot be used to claim we knew a 72nd-minute goal at
minute 32 during a live prediction.

## Player identity

Event participants arrive as bare display names; the roster arrives with ESPN
athlete ids. So the roster is parsed first and used to build a name-key →
player_uid map for the match, which is how "Bryan Mbeumo" in a shot event
becomes the same entity as athlete 244066 in the lineup. A name with no roster
match falls back to a name-derived uid, marked as such by the uid itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from ..db.connection import query_df, upsert_rows, write_connection
from ..db.repository import now_iso, record_source_status
from ..identity import player_name_key, soccer_player_uid
from ..logging_setup import get_logger
from ..sources.base import SourceError
from ..sources.espn_match import EspnMatchSource, MatchDetail, PlayerLine

log = get_logger(__name__)

SOURCE = "espn_match"


@dataclass
class MatchDetailReport:
    games_checked: int = 0
    events_written: int = 0
    team_stats_written: int = 0
    player_stats_written: int = 0
    contexts_written: int = 0
    players_registered: int = 0
    lineups_linked: int = 0
    skipped_no_espn_id: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "games_checked": self.games_checked,
            "events_written": self.events_written,
            "team_stats_written": self.team_stats_written,
            "player_stats_written": self.player_stats_written,
            "contexts_written": self.contexts_written,
            "players_registered": self.players_registered,
            "lineups_linked": self.lineups_linked,
            "skipped_no_espn_id": self.skipped_no_espn_id,
            "errors": self.errors[:5],
            "error_count": len(self.errors),
        }


def _team_uids(game_uid: str) -> dict[str, str]:
    frame = query_df(
        "SELECT home_team_uid, away_team_uid FROM games WHERE game_uid = ?", (game_uid,)
    )
    if frame.empty:
        return {}
    row = frame.iloc[0]
    return {"home": str(row["home_team_uid"]), "away": str(row["away_team_uid"])}


def _player_index(players: Sequence[PlayerLine]) -> dict[str, tuple[str, str]]:
    """name key -> (player_uid, team side), built from the roster.

    Both sides of a match are in one index because event participants do not
    say which team the *player* belongs to, only which team the event belongs
    to — and for an own goal those differ.
    """
    index: dict[str, tuple[str, str]] = {}
    for player in players:
        key = player_name_key(player.player_name)
        uid = player.player_uid
        if key and uid:
            index[key] = (uid, player.home_away)
    return index


def _register_players(detail: MatchDetail, team_uids: dict[str, str]) -> int:
    """Give every rostered player a row in ``players``.

    Soccer player identity did not exist before this; the table held NBA rows
    only. Registering here means event participants, lineup rows and future
    transfer records all point at one uid instead of three spellings.
    """
    timestamp = now_iso()
    rows = []
    for player in detail.players:
        uid = player.player_uid
        if not uid:
            continue
        rows.append({
            "player_uid": uid,
            "sport": "soccer",
            "full_name": player.player_name,
            "position": player.position,
            "team_uid": team_uids.get(player.home_away),
            "external_ids": json.dumps(
                {"espn_athlete_id": player.external_player_id}
                if player.external_player_id else {}
            ),
            "active": 1,
            "source": SOURCE,
            "retrieved_at": timestamp,
        })
    return upsert_rows("players", rows, conflict_columns=["player_uid"]) if rows else 0


def _link_lineups(game_uid: str, detail: MatchDetail, team_uids: dict[str, str]) -> int:
    """Backfill player_uid, jersey and substitution flags onto lineup rows.

    V3 wrote lineup observations with a NULL ``player_uid`` because there was
    nothing to point at. Now there is, and linking them is what lets the
    Match Center put a player's lineup slot and their events on the same card.
    """
    updates: list[tuple[Any, ...]] = []
    for player in detail.players:
        uid = player.player_uid
        team_uid = team_uids.get(player.home_away)
        if not uid or not team_uid:
            continue
        updates.append((uid, player.jersey, int(player.subbed_in), int(player.subbed_out),
                        game_uid, team_uid, player.player_name))
    if not updates:
        return 0

    with write_connection() as conn:
        cursor = conn.executemany(
            """
            UPDATE lineup_observations
               SET player_uid = ?, jersey = ?, subbed_in = ?, subbed_out = ?
             WHERE game_uid = ? AND team_uid = ? AND player_name = ?
            """,
            updates,
        )
        return int(cursor.rowcount or 0)


def _event_rows(game_uid: str, detail: MatchDetail, team_uids: dict[str, str],
                index: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    observed = detail.observed_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    retrieved = detail.retrieved_at.astimezone(timezone.utc).isoformat(timespec="seconds")

    side_by_name = {}
    if detail.context.home_name:
        side_by_name[detail.context.home_name] = "home"
    if detail.context.away_name:
        side_by_name[detail.context.away_name] = "away"

    rows: list[dict[str, Any]] = []
    home_score = away_score = 0
    for event in detail.events:
        side = side_by_name.get(event.team_name or "")
        player_uid = None
        if event.player_name:
            found = index.get(player_name_key(event.player_name))
            player_uid = found[0] if found else soccer_player_uid(event.player_name)
        assist_uid = None
        if event.assist_player_name:
            found = index.get(player_name_key(event.assist_player_name))
            assist_uid = found[0] if found else soccer_player_uid(event.assist_player_name)

        # Running score, so the timeline can show the state after each goal
        # without the frontend recomputing it from event types.
        if event.event_type in ("goal", "penalty_scored") and side:
            if side == "home":
                home_score += 1
            else:
                away_score += 1
        elif event.event_type == "own_goal" and side:
            # An own goal is credited to the scoring side's opponent.
            if side == "home":
                away_score += 1
            else:
                home_score += 1

        rows.append({
            "game_uid": game_uid,
            "external_id": event.external_id,
            "sequence": event.sequence,
            "event_type": event.event_type,
            "source_type": event.source_type,
            "period": event.period,
            "clock_seconds": event.clock_seconds,
            "clock_display": event.clock_display,
            "minute": event.minute,
            "wallclock_utc": event.wallclock_utc,
            "team_uid": team_uids.get(side) if side else None,
            "home_away": side,
            "player_uid": player_uid,
            "player_name": event.player_name,
            "assist_player_uid": assist_uid,
            "assist_player_name": event.assist_player_name,
            "scoring_play": int(event.scoring_play),
            "home_score": home_score,
            "away_score": away_score,
            "source_x": event.source_x,
            "source_y": event.source_y,
            "source_x2": event.source_x2,
            "source_y2": event.source_y2,
            "text": event.text,
            "short_text": event.short_text,
            "observed_at": observed,
            "retrieved_at": retrieved,
            "source": SOURCE,
        })
    return rows


def _stat_rows(game_uid: str, detail: MatchDetail, team_uids: dict[str, str]
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observed = detail.observed_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    retrieved = detail.retrieved_at.astimezone(timezone.utc).isoformat(timespec="seconds")

    team_rows = [
        {
            "game_uid": game_uid,
            "team_uid": team_uids.get(line.home_away) or line.team_name,
            "home_away": line.home_away,
            "stat_name": name,
            "stat_value": value,
            "display_value": display,
            "observed_at": observed, "retrieved_at": retrieved, "source": SOURCE,
        }
        for line in detail.team_stats
        for name, (value, display) in line.stats.items()
    ]

    player_rows = []
    for player in detail.players:
        uid = player.player_uid
        team_uid = team_uids.get(player.home_away)
        if not uid or not team_uid:
            continue
        for name, (value, display) in player.stats.items():
            player_rows.append({
                "game_uid": game_uid, "team_uid": team_uid,
                "player_uid": uid, "player_name": player.player_name,
                "stat_name": name, "stat_value": value, "display_value": display,
                "observed_at": observed, "retrieved_at": retrieved, "source": SOURCE,
            })
    return team_rows, player_rows


def _context_row(game_uid: str, detail: MatchDetail) -> dict[str, Any]:
    context = detail.context
    return {
        "game_uid": game_uid,
        "status_state": context.status_state,
        "status_name": context.status_name,
        "status_detail": context.status_detail,
        "period": context.period,
        "clock_display": context.clock_display,
        "venue": context.venue,
        "venue_city": context.venue_city,
        "venue_country": context.venue_country,
        "attendance": context.attendance,
        "officials": json.dumps(context.officials),
        "home_formation": context.home_formation,
        "away_formation": context.away_formation,
        "home_color": context.home_color,
        "away_color": context.away_color,
        "home_logo": context.home_logo,
        "away_logo": context.away_logo,
        "home_form": context.home_form,
        "away_form": context.away_form,
        "observed_at": detail.observed_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "retrieved_at": detail.retrieved_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE,
    }


def persist_match_detail(game_uid: str, detail: MatchDetail,
                         report: MatchDetailReport) -> None:
    """Write one parsed match into the store."""
    team_uids = _team_uids(game_uid)
    if not team_uids:
        report.errors.append(f"{game_uid}: no such game")
        return

    report.players_registered += _register_players(detail, team_uids)
    report.lineups_linked += _link_lineups(game_uid, detail, team_uids)

    index = _player_index(detail.players)
    events = _event_rows(game_uid, detail, team_uids, index)
    if events:
        # A re-parse can renumber sequences (it did, once, when I fixed the
        # ordering of period markers), so the old rows go before the new ones
        # land. Upserting alone would leave stale duplicates behind.
        with write_connection() as conn:
            conn.execute("DELETE FROM match_events WHERE game_uid = ? AND source = ?",
                         (game_uid, SOURCE))
        report.events_written += upsert_rows(
            "match_events", events,
            conflict_columns=["game_uid", "source", "external_id", "event_type", "sequence"],
        )

    team_rows, player_rows = _stat_rows(game_uid, detail, team_uids)
    if team_rows:
        report.team_stats_written += upsert_rows(
            "match_team_stats", team_rows,
            conflict_columns=["game_uid", "team_uid", "stat_name"],
        )
    if player_rows:
        report.player_stats_written += upsert_rows(
            "match_player_stats", player_rows,
            conflict_columns=["game_uid", "player_uid", "stat_name"],
        )

    report.contexts_written += upsert_rows(
        "match_context", [_context_row(game_uid, detail)], conflict_columns=["game_uid"]
    )


def ingest_match_detail(game_uids: Sequence[str], *, force: bool = False
                        ) -> MatchDetailReport:
    """Fetch and store match detail for specific fixtures."""
    report = MatchDetailReport()
    if not game_uids:
        return report

    source = EspnMatchSource()
    placeholders = ",".join("?" for _ in game_uids)
    games = query_df(
        f"SELECT game_uid, league_id, sport, espn_event_id, status, kickoff_utc, game_date "
        f"FROM games WHERE game_uid IN ({placeholders}) AND sport = 'soccer'",
        list(game_uids),
    )

    for _, game in games.iterrows():
        report.games_checked += 1
        if not game["espn_event_id"]:
            report.skipped_no_espn_id += 1
            continue

        # A finished match's record never changes, so it can be cached hard.
        # A live one must be re-read, which is what makes the live match centre
        # work at all.
        finished = bool(game["status"] == "final")

        try:
            detail = source.fetch_detail(
                str(game["league_id"]), str(game["espn_event_id"]),
                event_started=finished, force=force,
            )
        except SourceError as exc:
            report.errors.append(f"{game['game_uid']}: {exc}")
            continue

        persist_match_detail(str(game["game_uid"]), detail, report)

    record_source_status(
        SOURCE, "match_detail",
        status="ok" if report.events_written or report.contexts_written else "degraded",
        rows=report.events_written,
        message=f"{report.games_checked} games checked, {len(report.errors)} errors",
    )
    log.info("match detail ingest complete", extra=report.to_dict())
    return report


def backfill_match_detail(league_ids: Sequence[str], *, seasons: Sequence[str] | None = None,
                          limit: int = 400, refresh: bool = False) -> MatchDetailReport:
    """Ingest detail for completed matches that do not have it yet."""
    clauses = ["g.sport = 'soccer'", "g.espn_event_id IS NOT NULL",
               f"g.league_id IN ({','.join('?' for _ in league_ids)})"]
    params: list[Any] = list(league_ids)
    if seasons:
        clauses.append(f"g.season IN ({','.join('?' for _ in seasons)})")
        params.extend(seasons)
    missing = "" if refresh else "AND c.game_uid IS NULL"

    frame = query_df(
        f"""
        SELECT g.game_uid FROM games g
        LEFT JOIN match_context c ON c.game_uid = g.game_uid
        WHERE {' AND '.join(clauses)} {missing}
        ORDER BY g.game_date DESC LIMIT {int(limit)}
        """,
        params,
    )
    return ingest_match_detail(frame["game_uid"].tolist(), force=refresh)


def refresh_live_matches(*, window_hours: int = 3, league_ids: Sequence[str] | None = None
                         ) -> MatchDetailReport:
    """Re-read matches that are plausibly in progress.

    The window starts at kick-off and runs a few hours past it; anything
    outside that either has not begun or has a record that will not change.
    """
    clauses = ["g.sport = 'soccer'", "g.espn_event_id IS NOT NULL", "g.status != 'final'"]
    params: list[Any] = []
    if league_ids:
        clauses.append(f"g.league_id IN ({','.join('?' for _ in league_ids)})")
        params.extend(league_ids)

    now = datetime.now(timezone.utc)
    frame = query_df(
        f"""
        SELECT g.game_uid FROM games g
        WHERE {' AND '.join(clauses)}
          AND COALESCE(g.kickoff_utc, g.game_date) <= ?
          AND COALESCE(g.kickoff_utc, g.game_date) >= ?
        """,
        params + [now.isoformat(), (now - timedelta(hours=window_hours)).isoformat()],
    )
    return ingest_match_detail(frame["game_uid"].tolist(), force=True)
