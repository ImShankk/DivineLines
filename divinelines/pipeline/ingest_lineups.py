"""Lineup ingestion and the information-event stream.

Two jobs:

1. Store timestamped lineup observations so a prediction can be reconstructed
   from what was knowable when it was made.
2. Emit *information events* (``LINEUP_CONFIRMED``, ``PLAYER_OUT``, ...) by
   diffing consecutive observations, which is what later lets prediction
   movement be attributed to a cause rather than to "something changed".

The chronology rule is enforced on the read side, in
:func:`lineup_state_at`: a caller asks "what did we know at time T" and gets
only observations with ``observed_at <= T``. Nothing else may read the table
directly for feature purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from ..db.connection import query_df, upsert_rows
from ..db.repository import now_iso, record_source_status, soccer_team_uid
from ..identity import normalize_key
from ..logging_setup import get_logger
from ..sources.base import SourceError
from ..sources.espn_lineups import (
    STATE_CONFIRMED,
    STATE_FINAL,
    STATE_PROJECTED,
    EspnLineupSource,
    LineupObservation,
)

log = get_logger(__name__)

SOURCE = "espn_lineups"


@dataclass
class LineupIngestReport:
    events_checked: int = 0
    observations_written: int = 0
    players_written: int = 0
    information_events: int = 0
    skipped_no_espn_id: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_checked": self.events_checked,
            "observations_written": self.observations_written,
            "players_written": self.players_written,
            "information_events": self.information_events,
            "skipped_no_espn_id": self.skipped_no_espn_id,
            "errors": self.errors[:5], "error_count": len(self.errors),
        }


def _team_uid_for(game_uid: str, home_away: str) -> str | None:
    frame = query_df(
        "SELECT home_team_uid, away_team_uid FROM games WHERE game_uid = ?", (game_uid,)
    )
    if frame.empty:
        return None
    row = frame.iloc[0]
    return row["home_team_uid"] if home_away == "home" else row["away_team_uid"]


def _persist(game_uid: str, observation: LineupObservation) -> tuple[int, int]:
    rows: list[dict[str, Any]] = []
    observed = observation.observed_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    retrieved = observation.retrieved_at.astimezone(timezone.utc).isoformat(timespec="seconds")

    for team in observation.teams:
        team_uid = _team_uid_for(game_uid, team.home_away) or soccer_team_uid(team.team_name)
        if not team_uid:
            continue
        for entry in team.entries:
            rows.append({
                "game_uid": game_uid, "team_uid": team_uid, "sport": observation.sport,
                "player_uid": None, "player_name": entry.player_name,
                "external_player_id": entry.external_player_id,
                "status": entry.status, "role": entry.role,
                "position_group": entry.position_group,
                "formation_place": entry.formation_place, "formation": team.formation,
                "lineup_state": observation.lineup_state,
                "observed_at": observed, "source_timestamp": None,
                "retrieved_at": retrieved, "source": SOURCE,
            })

    if not rows:
        return 0, 0
    written = upsert_rows("lineup_observations", rows)
    return 1, written


def _diff_information_events(game_uid: str, observation: LineupObservation) -> int:
    """Emit events for what changed since the previous observation.

    A first observation emits ``LINEUP_CONFIRMED``; later ones emit the players
    who entered or left the XI. This is what makes "the probability moved 4.8
    points" answerable with "because the starting goalkeeper changed".
    """
    observed = observation.observed_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    previous = query_df(
        """
        SELECT team_uid, player_name, status, position_group, observed_at
        FROM lineup_observations
        WHERE game_uid = ? AND observed_at < ?
          AND observed_at = (SELECT MAX(observed_at) FROM lineup_observations
                             WHERE game_uid = ? AND observed_at < ?)
        """,
        (game_uid, observed, game_uid, observed),
    )

    events: list[dict[str, Any]] = []
    now = now_iso()

    if previous.empty:
        events.append({
            "game_uid": game_uid, "team_uid": None, "sport": observation.sport,
            "kind": "LINEUP_CONFIRMED" if observation.lineup_state == STATE_CONFIRMED
            else "LINEUP_OBSERVED",
            "detail": f"state={observation.lineup_state}; "
                      f"formations={'/'.join(t.formation or '?' for t in observation.teams)}",
            "magnitude": None, "observed_at": observed, "retrieved_at": now, "source": SOURCE,
        })
    else:
        before = {
            (row["team_uid"], normalize_key(row["player_name"])): row["status"]
            for _, row in previous.iterrows()
        }
        # Track what the new observation mentions, so a player who disappears
        # from the squad entirely still produces PLAYER_OUT. Only walking the
        # new entries would miss exactly the case that matters most: a starter
        # dropped from the sheet because he is injured.
        seen: set[tuple[str, str]] = set()
        previous_names = {
            (row["team_uid"], normalize_key(row["player_name"])): row["player_name"]
            for _, row in previous.iterrows()
        }

        for team in observation.teams:
            team_uid = _team_uid_for(game_uid, team.home_away)
            if not team_uid:
                continue
            for entry in team.entries:
                key = (team_uid, normalize_key(entry.player_name))
                seen.add(key)
                was = before.get(key)
                if was == entry.status:
                    continue
                if entry.status == "starter" and was != "starter":
                    kind = "PLAYER_IN"
                elif was == "starter" and entry.status != "starter":
                    kind = "PLAYER_OUT"
                else:
                    continue
                events.append({
                    "game_uid": game_uid, "team_uid": team_uid, "sport": observation.sport,
                    "kind": kind,
                    "detail": f"{entry.player_name} ({entry.position_group or entry.role or '?'})",
                    "magnitude": None, "observed_at": observed,
                    "retrieved_at": now, "source": SOURCE,
                })

        for key, status in before.items():
            if key in seen or status != "starter":
                continue
            team_uid, _ = key
            events.append({
                "game_uid": game_uid, "team_uid": team_uid, "sport": observation.sport,
                "kind": "PLAYER_OUT",
                "detail": f"{previous_names.get(key, 'unknown')} (dropped from squad)",
                "magnitude": None, "observed_at": observed,
                "retrieved_at": now, "source": SOURCE,
            })

    return upsert_rows("information_events", events) if events else 0


def ingest_lineups_for_games(game_uids: Sequence[str], *, force: bool = False
                             ) -> LineupIngestReport:
    """Fetch and store lineups for specific games."""
    report = LineupIngestReport()
    if not game_uids:
        return report

    source = EspnLineupSource()
    placeholders = ",".join("?" for _ in game_uids)
    games = query_df(
        f"SELECT game_uid, league_id, sport, espn_event_id, status, kickoff_utc, game_date "
        f"FROM games WHERE game_uid IN ({placeholders})",
        list(game_uids),
    )

    now = pd.Timestamp.now(tz="UTC")
    for _, game in games.iterrows():
        report.events_checked += 1
        if not game["espn_event_id"]:
            report.skipped_no_espn_id += 1
            continue

        start = pd.to_datetime(game["kickoff_utc"] or game["game_date"], utc=True,
                               format="mixed", errors="coerce")
        started = bool(game["status"] == "final" or (pd.notna(start) and start <= now))

        try:
            observation = source.fetch_lineup(
                str(game["league_id"]), str(game["espn_event_id"]),
                event_started=started, force=force,
            )
        except SourceError as exc:
            report.errors.append(f"{game['game_uid']}: {exc}")
            continue

        written_obs, written_players = _persist(str(game["game_uid"]), observation)
        report.observations_written += written_obs
        report.players_written += written_players
        report.information_events += _diff_information_events(str(game["game_uid"]), observation)

    record_source_status(
        SOURCE, "lineups",
        status="ok" if report.observations_written else "degraded",
        rows=report.players_written,
        message=f"{report.events_checked} events checked, {len(report.errors)} errors",
    )
    log.info("lineup ingest complete", extra=report.to_dict())
    return report


def ingest_upcoming_lineups(*, hours_ahead: int = 6, league_ids: Sequence[str] | None = None,
                            force: bool = False) -> LineupIngestReport:
    """Poll lineups for fixtures kicking off soon.

    Lineups are typically published about an hour before kick-off, so polling a
    fixture three days out is a wasted request; the window is deliberately tight.
    """
    clauses = ["g.status = 'scheduled'", "g.espn_event_id IS NOT NULL"]
    params: list[Any] = []
    if league_ids:
        clauses.append(f"g.league_id IN ({','.join('?' for _ in league_ids)})")
        params.extend(league_ids)

    horizon = (datetime.now(timezone.utc) + timedelta(hours=hours_ahead)).isoformat()
    frame = query_df(
        f"SELECT game_uid FROM games g WHERE {' AND '.join(clauses)} "
        f"AND COALESCE(g.kickoff_utc, g.game_date) <= ? ORDER BY g.kickoff_utc",
        params + [horizon],
    )
    return ingest_lineups_for_games(frame["game_uid"].tolist(), force=force)


def backfill_historical_lineups(league_ids: Sequence[str], *, limit: int = 400,
                                seasons: Sequence[str] | None = None) -> LineupIngestReport:
    """Fetch actual XIs for completed matches, for research only.

    These carry ``lineup_state='final'`` and are excluded from live prediction
    by :func:`lineup_state_at`. They exist so the question "would knowing the
    lineup have helped at all?" can be answered as an upper bound.
    """
    clauses = ["g.status = 'final'", "g.espn_event_id IS NOT NULL",
               f"g.league_id IN ({','.join('?' for _ in league_ids)})"]
    params: list[Any] = list(league_ids)
    if seasons:
        clauses.append(f"g.season IN ({','.join('?' for _ in seasons)})")
        params.extend(seasons)

    frame = query_df(
        f"""
        SELECT g.game_uid FROM games g
        LEFT JOIN (SELECT DISTINCT game_uid FROM lineup_observations) l ON l.game_uid = g.game_uid
        WHERE {' AND '.join(clauses)} AND l.game_uid IS NULL
        ORDER BY g.game_date DESC LIMIT {int(limit)}
        """,
        params,
    )
    return ingest_lineups_for_games(frame["game_uid"].tolist())


# --------------------------------------------------------------------------
# Chronology-safe reads
# --------------------------------------------------------------------------

def lineup_state_at(game_uid: str, as_of: datetime | str | None = None,
                    *, allow_final: bool = False) -> pd.DataFrame:
    """Lineup rows the platform had observed by ``as_of``.

    I deliberately filter on ``observed_at`` rather than on any source-supplied
    timestamp: ESPN does not say when a lineup was published, so the only thing
    that is verifiably true is when *we* saw it. ``allow_final`` is off by
    default so historical research rows can never reach a live prediction.
    """
    as_of = as_of or datetime.now(timezone.utc)
    stamp = as_of if isinstance(as_of, str) else as_of.astimezone(timezone.utc).isoformat()

    clause = "" if allow_final else f"AND lineup_state != '{STATE_FINAL}'"
    frame = query_df(
        f"""
        SELECT * FROM lineup_observations
        WHERE game_uid = ? AND observed_at <= ? {clause}
          AND observed_at = (
              SELECT MAX(observed_at) FROM lineup_observations
              WHERE game_uid = ? AND observed_at <= ? {clause}
          )
        """,
        (game_uid, stamp, game_uid, stamp),
    )
    return frame


def latest_lineup_state(game_uid: str) -> str:
    """``confirmed`` / ``projected`` / ``final`` / ``unknown`` for one game."""
    frame = query_df(
        "SELECT lineup_state FROM lineup_observations WHERE game_uid = ? "
        "ORDER BY observed_at DESC LIMIT 1",
        (game_uid,),
    )
    return str(frame["lineup_state"].iloc[0]) if not frame.empty else "unknown"


def information_events_for(game_uid: str, as_of: datetime | str | None = None) -> pd.DataFrame:
    params: list[Any] = [game_uid]
    clause = ""
    if as_of is not None:
        clause = "AND observed_at <= ?"
        params.append(as_of if isinstance(as_of, str)
                      else as_of.astimezone(timezone.utc).isoformat())
    return query_df(
        f"SELECT * FROM information_events WHERE game_uid = ? {clause} ORDER BY observed_at",
        params,
    )
