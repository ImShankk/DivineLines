"""Soccer Match Center routes.

Handlers do three things and no more: validate input, call a service, turn a
``LookupError`` into a 404. Everything else — assembly, chronology, the
decision about what can honestly be shown — lives in
:mod:`divinelines.matchcenter`.

The full Match Center is one request. The per-panel routes exist for the two
cases that genuinely need them: polling a single panel on a live match, and
scrubbing the replay slider without re-fetching the whole payload.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..db.connection import query_df
from ..logging_setup import get_logger
from ..matchcenter.report import match_report
from ..matchcenter.service import (
    MatchNotFound,
    load_game,
    match_center,
    match_events,
    match_momentum,
    match_passes,
    resolve_bounds,
    standings_for,
    first_event_observation,
)
from ..matchcenter.spatial import event_density, shot_map

log = get_logger(__name__)

router = APIRouter(prefix="/api/soccer", tags=["soccer"])

#: Replay positions beyond this are not a match minute any more. Extra time
#: plus a generous allowance for stoppages.
MAX_MINUTE = 150.0


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def _bounds(game_uid: str, as_of: str | None, minute: float | None):
    try:
        game = load_game(game_uid)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail=f"unknown match '{game_uid}'") from None
    return game, resolve_bounds(game, as_of, minute,
                                first_event_observed_at=first_event_observation(game_uid))


# ------------------------------------------------------------------ discovery

@router.get("/matches")
def matches(league_id: str | None = None,
            season: str | None = None,
            status: str | None = Query(None, pattern="^(scheduled|final)$"),
            with_events: bool = False,
            limit: int = Query(60, ge=1, le=500)) -> dict[str, Any]:
    """Soccer fixtures with a per-match count of what the Match Center can show.

    The counts are the point: a list that does not say which matches have an
    event feed sends the reader into an empty Match Center to find out.
    """
    clauses = ["g.sport = 'soccer'"]
    params: list[Any] = []
    for column, value in (("league_id", league_id), ("season", season), ("status", status)):
        if value:
            clauses.append(f"g.{column} = ?")
            params.append(value)
    # EXISTS rather than HAVING: SQLite rejects a HAVING clause on a query with
    # no GROUP BY, and the counts here are correlated subqueries, not aggregates.
    if with_events:
        clauses.append("EXISTS (SELECT 1 FROM match_events e WHERE e.game_uid = g.game_uid)")

    frame = query_df(
        f"""
        SELECT g.game_uid, g.league_id, g.season, g.game_date, g.kickoff_utc, g.status,
               g.home_team_uid, g.away_team_uid, g.home_score, g.away_score,
               th.canonical_name AS home_name, ta.canonical_name AS away_name,
               c.status_name, c.venue, c.attendance,
               (SELECT COUNT(*) FROM match_events e WHERE e.game_uid = g.game_uid) AS events,
               (SELECT COUNT(*) FROM odds_snapshots o WHERE o.game_uid = g.game_uid) AS prices,
               (SELECT COUNT(*) FROM lineup_observations l
                 WHERE l.game_uid = g.game_uid AND l.status = 'starter') AS starters,
               (SELECT COUNT(*) FROM predictions p WHERE p.game_uid = g.game_uid) AS predictions
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        LEFT JOIN match_context c ON c.game_uid = g.game_uid
        WHERE {' AND '.join(clauses)}
        ORDER BY g.game_date DESC, g.game_uid
        LIMIT ?
        """,
        params + [limit],
    )
    return {"matches": _records(frame), "count": int(len(frame))}


@router.get("/standings")
def standings(league_id: str, season: str,
              before: str | None = None) -> dict[str, Any]:
    """League table computed from stored results, optionally cut at a date."""
    game = {"league_id": league_id, "season": season,
            "game_date": before or "9999-12-31",
            "home_team_uid": None, "away_team_uid": None}
    return standings_for(game, as_of_date=before)


# --------------------------------------------------------------- per-match

@router.get("/match/{game_uid}/events")
def events(game_uid: str, as_of: str | None = None,
           minute: float | None = Query(None, ge=0, le=MAX_MINUTE),
           include_void: bool = False) -> dict[str, Any]:
    _game, bounds = _bounds(game_uid, as_of, minute)
    rows = match_events(game_uid, as_of=bounds.observation, minute=bounds.minute,
                        include_void=include_void)
    return {"game_uid": game_uid, "events": rows, "count": len(rows),
            "replay": bounds.to_dict()}


@router.get("/match/{game_uid}/momentum")
def momentum(game_uid: str, as_of: str | None = None,
             minute: float | None = Query(None, ge=0, le=MAX_MINUTE)) -> dict[str, Any]:
    _game, bounds = _bounds(game_uid, as_of, minute)
    payload = match_momentum(game_uid, as_of=bounds.observation, minute=bounds.minute)
    payload["replay"] = bounds.to_dict()
    return payload


@router.get("/match/{game_uid}/shots")
def shots(game_uid: str, as_of: str | None = None,
          minute: float | None = Query(None, ge=0, le=MAX_MINUTE)) -> dict[str, Any]:
    _game, bounds = _bounds(game_uid, as_of, minute)
    rows = match_events(game_uid, as_of=bounds.observation, minute=bounds.minute)
    points = shot_map(rows)
    return {
        "game_uid": game_uid,
        "available": bool(points),
        "points": [point.to_dict() for point in points],
        "located": len(points),
        "pitch": {"length": 105.0, "width": 68.0, "orientation": "home attacks right"},
        "reason": None if points else "no shot in this match carries a field position",
        "replay": bounds.to_dict(),
    }


@router.get("/match/{game_uid}/heatmap")
def heatmap(game_uid: str, side: str | None = Query(None, pattern="^(home|away)$"),
            as_of: str | None = None,
            minute: float | None = Query(None, ge=0, le=MAX_MINUTE),
            columns: int = Query(12, ge=4, le=32),
            rows: int = Query(8, ge=3, le=20)) -> dict[str, Any]:
    """Event-location density. Explicitly not a tracking heatmap."""
    _game, bounds = _bounds(game_uid, as_of, minute)
    events_rows = match_events(game_uid, as_of=bounds.observation, minute=bounds.minute)
    payload = event_density(events_rows, side=side, columns=columns, rows=rows)
    payload["game_uid"] = game_uid
    payload["side"] = side
    payload["replay"] = bounds.to_dict()
    return payload


@router.get("/match/{game_uid}/passes")
def passes(game_uid: str) -> dict[str, Any]:
    """Pass events — no configured source publishes them.

    Kept as a real endpoint rather than omitted so the frontend has one
    consistent way to ask, and gets a reason rather than a 404.
    """
    _game, _bounds_ = _bounds(game_uid, None, None)
    payload = match_passes(game_uid)
    payload["game_uid"] = game_uid
    return payload


@router.get("/match/{game_uid}/stats")
def stats(game_uid: str, as_of: str | None = None,
          minute: float | None = Query(None, ge=0, le=MAX_MINUTE)) -> dict[str, Any]:
    payload = _match_center(game_uid, as_of, minute)
    return {"game_uid": game_uid, "statistics": payload["statistics"],
            "replay": payload["replay"]}


@router.get("/match/{game_uid}/players")
def players(game_uid: str, as_of: str | None = None) -> dict[str, Any]:
    payload = _match_center(game_uid, as_of, None)
    return {"game_uid": game_uid, "players": payload["players"],
            "lineups": payload["lineups"], "contributions": payload["contributions"]}


@router.get("/match/{game_uid}/markets")
def markets(game_uid: str, market: str = "1x2", as_of: str | None = None,
            minute: float | None = Query(None, ge=0, le=MAX_MINUTE)) -> dict[str, Any]:
    payload = _match_center(game_uid, as_of, minute, market=market)
    return {"game_uid": game_uid, "market": payload["market"], "model": payload["model"],
            "model_vs_market": payload["model_vs_market"], "replay": payload["replay"]}


@router.get("/match/{game_uid}/report")
def report(game_uid: str, as_of: str | None = None,
           minute: float | None = Query(None, ge=0, le=MAX_MINUTE),
           market: str = "1x2") -> dict[str, Any]:
    try:
        return match_report(game_uid, as_of=as_of, minute=minute, market=market)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail=f"unknown match '{game_uid}'") from None


@router.get("/match/{game_uid}")
def match(game_uid: str, as_of: str | None = None,
          minute: float | None = Query(None, ge=0, le=MAX_MINUTE),
          market: str = "1x2") -> dict[str, Any]:
    """The whole Match Center in one request."""
    return _match_center(game_uid, as_of, minute, market=market)


def _match_center(game_uid: str, as_of: str | None, minute: float | None,
                  market: str = "1x2") -> dict[str, Any]:
    try:
        return match_center(game_uid, as_of=as_of, minute=minute, market=market)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail=f"unknown match '{game_uid}'") from None
