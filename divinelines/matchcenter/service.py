"""Assemble the Match Center for one fixture.

Routes call this; it calls the repository. No business logic lives in a
FastAPI handler.

## The two clocks, and why replay is enforced here

A Match Center view is bounded by two independent things:

``minute``
    where we are *in the match*. At minute 32 the 72nd-minute goal has not
    happened, so it must not be in the event list, the momentum curve, the
    shot map, the score or the report.

``as_of``
    what the *platform had observed* at a moment in wall-clock time. Prices,
    predictions and lineups are filtered on their observation timestamps, so a
    view reconstructed for 19:00 cannot contain the 19:15 price.

Passing ``minute`` without ``as_of`` derives one from kick-off, so a match
replay and an information replay stay consistent instead of showing minute 32
of the match next to the closing line.

The filtering happens in SQL and in these functions — never in React. A
frontend that merely hides future events is one refresh away from leaking
them, and the leakage tests here assert the server behaviour, not the markup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

from ..betting.odds_math import remove_vig
from ..db.connection import query_df
from ..logging_setup import get_logger
from ..sources.espn_match import SHOT_TYPES, VOID_TYPES
from .momentum import momentum_series, momentum_summary
from .quality import match_intelligence
from .spatial import event_density, has_position, shot_map
from .stats import contributions, player_lines, team_comparison

log = get_logger(__name__)

#: Live views of a match in progress; anything else is a settled record.
LIVE_STATES = {"LIVE_FIRST_HALF", "HALFTIME", "LIVE_SECOND_HALF",
               "EXTRA_TIME", "PENALTIES"}


class MatchNotFound(LookupError):
    """Raised when a fixture is not in the canonical store."""


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def load_game(game_uid: str) -> dict[str, Any]:
    frame = query_df(
        """
        SELECT g.*, th.canonical_name AS home_name, ta.canonical_name AS away_name,
               l.name AS league_name, l.country AS league_country
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        LEFT JOIN leagues l ON l.league_id = g.league_id
        WHERE g.game_uid = ?
        """,
        (game_uid,),
    )
    if frame.empty:
        raise MatchNotFound(game_uid)
    return _records(frame)[0]


def load_context(game_uid: str) -> dict[str, Any] | None:
    frame = query_df("SELECT * FROM match_context WHERE game_uid = ?", (game_uid,))
    if frame.empty:
        return None
    context = _records(frame)[0]
    try:
        context["officials"] = json.loads(context.get("officials") or "[]")
    except (TypeError, ValueError):
        context["officials"] = []
    return context


@dataclass(frozen=True)
class ReplayBounds:
    """The two cut-offs a bounded view is subject to.

    They are separate because they are different claims, and the first version
    of this code conflated them — which produced an empty match at every replay
    position, because a match backfilled in August has an ``observed_at`` in
    August no matter which minute you ask for.

    ``observation``
        applies to ``observed_at`` / ``captured_at`` / ``created_at``: what the
        platform had *seen* by this instant. Only set when a caller asks for it
        explicitly, because it is a statement about our own history.

    ``information``
        bounds the market, the model and the lineups. Derived from kick-off
        plus the replay minute when the caller replays by match clock.

    ``minute``
        bounds the event stream on the event's own match clock, which is the
        source's attested time for when it happened.

    ``retrospective_events`` records the thing a replay must not hide: for a
    match ingested after full time, the event stream was never observed live.
    Replaying it reconstructs *what happened by minute 32*, which is not the
    same claim as *what we knew at minute 32*.
    """

    observation: str | None
    information: str | None
    minute: float | None
    retrospective_events: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_as_of": self.observation,
            "information_as_of": self.information,
            "replay_minute": self.minute,
            "events_basis": ("match clock attested by the source"
                             if self.minute is not None else "all recorded events"),
            "retrospective_events": self.retrospective_events,
            "note": ("Events are bounded by their own match clock; prices, predictions "
                     "and lineups are bounded by when the platform observed them."
                     if self.minute is not None else
                     "Unbounded view: everything recorded for this fixture."),
        }


def resolve_bounds(game: dict[str, Any], as_of: str | None, minute: float | None,
                   *, first_event_observed_at: str | None = None) -> ReplayBounds:
    """Work out both cut-offs implied by a replay request."""
    information = as_of
    if information is None and minute is not None:
        kickoff = pd.to_datetime(game.get("kickoff_utc") or game.get("game_date"),
                                 utc=True, format="mixed", errors="coerce")
        if not pd.isna(kickoff):
            information = (kickoff + pd.Timedelta(minutes=float(minute))).isoformat()

    retrospective = False
    if first_event_observed_at:
        kickoff = pd.to_datetime(game.get("kickoff_utc") or game.get("game_date"),
                                 utc=True, format="mixed", errors="coerce")
        observed = pd.to_datetime(first_event_observed_at, utc=True,
                                  format="mixed", errors="coerce")
        if not pd.isna(kickoff) and not pd.isna(observed):
            retrospective = bool(observed > kickoff + pd.Timedelta(minutes=1))

    return ReplayBounds(observation=as_of, information=information, minute=minute,
                        retrospective_events=retrospective)


def first_event_observation(game_uid: str) -> str | None:
    frame = query_df(
        "SELECT MIN(observed_at) AS first FROM match_events WHERE game_uid = ?",
        (game_uid,),
    )
    if frame.empty:
        return None
    value = frame["first"].iloc[0]
    return None if pd.isna(value) else str(value)


# --------------------------------------------------------------------- events

def match_events(game_uid: str, *, as_of: str | None = None, minute: float | None = None,
                 include_void: bool = False) -> list[dict[str, Any]]:
    """Ordered events, bounded by both clocks.

    ``observed_at <= as_of`` is the leakage guard; ``minute <= minute`` is the
    replay guard. Events without a clock (a card the feed did not time) are
    kept out of a bounded replay rather than assumed to be early — guessing
    would put an untimed red card in the wrong half.
    """
    clauses = ["e.game_uid = ?"]
    params: list[Any] = [game_uid]
    if as_of:
        clauses.append("e.observed_at <= ?")
        params.append(as_of)
    if minute is not None:
        clauses.append("e.minute IS NOT NULL AND e.minute <= ?")
        params.append(float(minute))
    if not include_void:
        placeholders = ",".join("?" for _ in VOID_TYPES)
        clauses.append(f"e.event_type NOT IN ({placeholders})")
        params.extend(sorted(VOID_TYPES))

    frame = query_df(
        f"""
        SELECT e.*, t.canonical_name AS team_name
        FROM match_events e
        LEFT JOIN teams t ON t.team_uid = e.team_uid
        WHERE {' AND '.join(clauses)}
        ORDER BY e.sequence
        """,
        params,
    )
    return _records(frame)


def _score_from_events(events: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Running score at the last event seen.

    Recomputed rather than read from ``games``: the stored score is full time,
    which is the wrong answer for every replay position except the last one.
    """
    # The running score is monotonic, so taking the maximum is robust to an
    # event arriving out of order — which is not hypothetical: ESPN appends
    # period markers after the play stream, and reading the "last" value gave
    # a 10th-minute replay the full-time score.
    home = max((int(e["home_score"]) for e in events
                if e.get("home_score") is not None), default=0)
    away = max((int(e["away_score"]) for e in events
                if e.get("away_score") is not None), default=0)
    return home, away


# ---------------------------------------------------------------------- state

def match_state(game: dict[str, Any], context: dict[str, Any] | None,
                *, minute: float | None) -> dict[str, Any]:
    """The normalised state the frontend renders from.

    A replay is explicitly its own mode. A finished match being replayed is
    neither live nor finished from the reader's point of view, and labelling it
    "LIVE" because events are still arriving on screen would be a lie.
    """
    stored = (context or {}).get("status_name")
    from ..sources.espn_match import normalise_state

    state = normalise_state(stored, (context or {}).get("status_state"))
    if not context:
        state = "FINISHED" if game.get("status") == "final" else "SCHEDULED"

    mode = "POST_MATCH"
    if minute is not None:
        mode = "REPLAY"
    elif state in LIVE_STATES:
        mode = "LIVE"
    elif state == "SCHEDULED":
        mode = "PRE_MATCH"

    return {
        "state": state,
        "mode": mode,
        "is_live": state in LIVE_STATES and minute is None,
        "period": (context or {}).get("period"),
        "clock_display": (context or {}).get("clock_display"),
        "status_detail": (context or {}).get("status_detail"),
        "replay_minute": minute,
    }


# ----------------------------------------------------------------- statistics

def team_stats_for(game_uid: str, *, as_of: str | None = None) -> list[dict[str, Any]]:
    clause = "AND observed_at <= ?" if as_of else ""
    params: list[Any] = [game_uid] + ([as_of] if as_of else [])
    return _records(query_df(
        f"SELECT * FROM match_team_stats WHERE game_uid = ? {clause}", params
    ))


def _event_derived_stats(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counting statistics rebuilt from the events seen so far.

    The stored box score is a full-time figure. During a replay it would be
    wrong in the most misleading way possible — showing final possession next
    to a 32nd-minute score — so a bounded view recounts what it can from
    events and says plainly which metrics it cannot recover.
    """
    counters = {
        "totalShots": SHOT_TYPES,
        "shotsOnTarget": {"goal", "penalty_scored", "shot_on_target", "penalty_saved"},
        "blockedShots": {"shot_blocked"},
        "wonCorners": {"corner"},
        "foulsCommitted": {"foul", "handball"},
        "offsides": {"offside"},
        "yellowCards": {"yellow_card"},
        "redCards": {"red_card"},
    }
    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        side_events = [e for e in events if e.get("home_away") == side]
        for name, types in counters.items():
            rows.append({
                "home_away": side,
                "stat_name": name,
                "stat_value": float(sum(1 for e in side_events
                                        if str(e.get("event_type")) in types)),
                "display_value": None,
            })
    return {
        "rows": rows,
        "unavailable": ["possessionPct", "totalPasses", "accuratePasses", "passPct",
                        "totalTackles", "interceptions", "totalCrosses", "saves"],
    }


# --------------------------------------------------------------------- lineups

def lineups_for(game_uid: str, *, as_of: str | None = None) -> list[dict[str, Any]]:
    """The most recent lineup observation the platform had by ``as_of``."""
    clause = "AND observed_at <= ?" if as_of else ""
    params: list[Any] = [game_uid] + ([as_of] if as_of else [])
    params += [game_uid] + ([as_of] if as_of else [])
    return _records(query_df(
        f"""
        SELECT l.*, t.canonical_name AS team_name
        FROM lineup_observations l
        LEFT JOIN teams t ON t.team_uid = l.team_uid
        WHERE l.game_uid = ? {clause}
          AND l.observed_at = (
              SELECT MAX(observed_at) FROM lineup_observations
              WHERE game_uid = ? {clause}
          )
        ORDER BY l.team_uid, l.status DESC, CAST(l.formation_place AS INTEGER)
        """,
        params,
    ))


def lineup_board(rows: Sequence[dict[str, Any]], game: dict[str, Any]) -> dict[str, Any]:
    """Group lineup rows into a per-side board the pitch view can draw."""
    sides: dict[str, dict[str, Any]] = {}
    for side, team_uid in (("home", game.get("home_team_uid")),
                           ("away", game.get("away_team_uid"))):
        members = [row for row in rows if row.get("team_uid") == team_uid]
        sides[side] = {
            "team_uid": team_uid,
            "team_name": game.get(f"{side}_name"),
            "formation": next((m.get("formation") for m in members if m.get("formation")), None),
            "observed_at": next((m.get("observed_at") for m in members), None),
            "lineup_state": next((m.get("lineup_state") for m in members), None),
            "starters": [_lineup_entry(m) for m in members if m.get("status") == "starter"],
            "bench": [_lineup_entry(m) for m in members if m.get("status") == "bench"],
            "other": [_lineup_entry(m) for m in members
                      if m.get("status") not in ("starter", "bench")],
        }
    return sides


def _lineup_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_uid": row.get("player_uid"),
        "player_name": row.get("player_name"),
        "jersey": row.get("jersey"),
        "role": row.get("role"),
        "position_group": row.get("position_group"),
        "formation_place": row.get("formation_place"),
        "subbed_in": bool(row.get("subbed_in")),
        "subbed_out": bool(row.get("subbed_out")),
    }


# ---------------------------------------------------------------------- market

def market_for(game_uid: str, *, market: str = "1x2", as_of: str | None = None
               ) -> dict[str, Any]:
    """Price history and the derived no-vig view, bounded by observation time.

    Best price and consensus are reported as separate numbers on purpose. V3
    established how easily a best-of-N entry compared against a consensus close
    manufactures positive CLV, so the Match Center never collapses the two.
    """
    clause = "AND captured_at <= ?" if as_of else ""
    params: list[Any] = [game_uid, market] + ([as_of] if as_of else [])
    frame = query_df(
        f"""
        SELECT captured_at, phase, selection, bookmaker, price_decimal, source, is_closing
        FROM odds_snapshots
        WHERE game_uid = ? AND market = ? {clause}
        ORDER BY captured_at
        """,
        params,
    )
    if frame.empty:
        return {"available": False, "market": market,
                "reason": "no prices stored for this fixture", "snapshots": 0}

    selections = sorted(frame["selection"].unique().tolist())
    series: list[dict[str, Any]] = []
    for (captured, phase), group in frame.groupby(["captured_at", "phase"], sort=True):
        consensus = group.groupby("selection")["price_decimal"].median().to_dict()
        best = group.groupby("selection")["price_decimal"].max().to_dict()
        novig = None
        if len(consensus) >= 2:
            try:
                order = list(consensus)
                novig = dict(zip(order, remove_vig([consensus[s] for s in order])))
            except (ValueError, ZeroDivisionError):
                novig = None
        series.append({
            "captured_at": str(captured),
            "phase": phase,
            "consensus": {k: round(float(v), 3) for k, v in consensus.items()},
            "best": {k: round(float(v), 3) for k, v in best.items()},
            "novig": None if novig is None else {k: round(float(v), 5)
                                                 for k, v in novig.items()},
            "books": int(group["bookmaker"].nunique()),
        })

    opening = next((point for point in series if point["phase"] == "open"), series[0])
    closing = next((point for point in reversed(series) if point["phase"] == "close"), None)
    latest = series[-1]

    return {
        "available": True,
        "market": market,
        "selections": selections,
        "series": series,
        "opening": opening,
        "closing": closing,
        "latest": latest,
        "snapshots": int(len(frame)),
        "books": int(frame["bookmaker"].nunique()),
        "sources": sorted(frame["source"].unique().tolist()),
        "as_of": as_of,
        "note": ("Best price and consensus are shown separately. The best of N books "
                 "sits above the consensus by construction, and treating one as the "
                 "other is how a fake edge appears."),
    }


# ----------------------------------------------------------------------- model

def model_for(game_uid: str, *, as_of: str | None = None) -> dict[str, Any]:
    """Stored predictions for this fixture, newest last, plus model health."""
    clause = "AND created_at <= ?" if as_of else ""
    params: list[Any] = [game_uid] + ([as_of] if as_of else [])
    frame = query_df(
        f"""
        SELECT prediction_id, created_at, market, selection, model_prob, market_prob,
               price_decimal, bookmaker, edge, ev_per_unit, kelly_fraction, stake,
               confidence, edge_score, data_quality, model_id, model_version,
               prediction_stage, lineup_state, superseded_at, mode, flags, explanation
        FROM predictions WHERE game_uid = ? {clause}
        ORDER BY created_at, prediction_id
        """,
        params,
    )
    predictions = _records(frame)
    for prediction in predictions:
        for key in ("flags", "explanation"):
            raw = prediction.get(key)
            if isinstance(raw, str):
                try:
                    prediction[key] = json.loads(raw)
                except ValueError:
                    prediction[key] = None

    active = [p for p in predictions if not p.get("superseded_at")]
    latest: dict[str, dict[str, Any]] = {}
    for prediction in active:
        latest[str(prediction["selection"])] = prediction

    return {
        "available": bool(predictions),
        "predictions": predictions,
        "latest": latest,
        "count": len(predictions),
        "superseded": sum(1 for p in predictions if p.get("superseded_at")),
        "reason": None if predictions else "no prediction was recorded for this fixture",
    }


def model_vs_market(model: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    """Where the model disagrees with the market, in probability points.

    A disagreement is not an edge. Whether a price qualifies as a bet is the
    betting engine's decision and depends on model health and risk limits, not
    on the arithmetic here.
    """
    if not model.get("available") or not market.get("available"):
        return {"available": False,
                "reason": "needs both a stored prediction and a stored price"}

    reference = (market.get("latest") or {}).get("novig")
    if not reference:
        return {"available": False, "reason": "no no-vig market probability available"}

    rows = []
    for selection, prediction in (model.get("latest") or {}).items():
        market_probability = reference.get(selection)
        if market_probability is None:
            continue
        model_probability = float(prediction["model_prob"])
        rows.append({
            "selection": selection,
            "model_probability": round(model_probability, 5),
            "market_probability": round(float(market_probability), 5),
            "difference_points": round((model_probability - float(market_probability)) * 100, 2),
            "price": prediction.get("price_decimal"),
            "bookmaker": prediction.get("bookmaker"),
            "stake": prediction.get("stake"),
        })

    return {
        "available": bool(rows),
        "rows": sorted(rows, key=lambda row: -abs(row["difference_points"])),
        "note": ("A probability difference is a disagreement, not an established edge. "
                 "Staking is decided by the betting engine under model-health limits."),
    }


# ------------------------------------------------------------------ standings

def standings_for(game: dict[str, Any], *, as_of_date: str | None = None) -> dict[str, Any]:
    """League table built from this platform's own results.

    Deliberately not read from the source's embedded table: ours can be cut at
    a date, which is what a match page needs — the table as it stood before
    this fixture, not as it stands today.
    """
    league_id = game.get("league_id")
    season = game.get("season")
    if not league_id or not season:
        return {"available": False, "reason": "fixture has no league or season"}

    cutoff = as_of_date or game.get("game_date")
    frame = query_df(
        """
        SELECT g.home_team_uid, g.away_team_uid, g.home_score, g.away_score,
               th.canonical_name AS home_name, ta.canonical_name AS away_name
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        WHERE g.league_id = ? AND g.season = ? AND g.status = 'final'
          AND g.game_date < ?
        """,
        (league_id, season, cutoff),
    )
    if frame.empty:
        return {"available": False,
                "reason": f"no completed {league_id} {season} matches before {cutoff}"}

    table: dict[str, dict[str, Any]] = {}

    def entry(uid: str, name: str) -> dict[str, Any]:
        return table.setdefault(uid, {
            "team_uid": uid, "team_name": name, "played": 0, "won": 0,
            "drawn": 0, "lost": 0, "goals_for": 0, "goals_against": 0, "points": 0,
        })

    for _, row in frame.iterrows():
        home = entry(str(row["home_team_uid"]), str(row["home_name"]))
        away = entry(str(row["away_team_uid"]), str(row["away_name"]))
        home_goals, away_goals = int(row["home_score"] or 0), int(row["away_score"] or 0)
        for side, scored, conceded in ((home, home_goals, away_goals),
                                       (away, away_goals, home_goals)):
            side["played"] += 1
            side["goals_for"] += scored
            side["goals_against"] += conceded
        if home_goals > away_goals:
            home["won"] += 1; home["points"] += 3; away["lost"] += 1
        elif home_goals < away_goals:
            away["won"] += 1; away["points"] += 3; home["lost"] += 1
        else:
            home["drawn"] += 1; away["drawn"] += 1
            home["points"] += 1; away["points"] += 1

    rows = sorted(
        table.values(),
        key=lambda row: (-row["points"], -(row["goals_for"] - row["goals_against"]),
                         -row["goals_for"], row["team_name"]),
    )
    for position, row in enumerate(rows, start=1):
        row["position"] = position
        row["goal_difference"] = row["goals_for"] - row["goals_against"]

    return {
        "available": True,
        "league_id": league_id,
        "season": season,
        "as_of_date": cutoff,
        "table": rows,
        "highlight": [game.get("home_team_uid"), game.get("away_team_uid")],
        "note": "Computed from stored results before this fixture's date.",
    }


# ------------------------------------------------------------------- passing

def match_passes(game_uid: str) -> dict[str, Any]:
    """Pass events for a fixture — which no configured source publishes.

    The endpoint and the shape exist so a provider that does publish passes can
    be dropped in without redesigning anything. Until one is configured this
    returns ``NO_DATA`` with the reason, because an empty pitch with invented
    dots on it would be worse than no pitch at all.
    """
    aggregate = query_df(
        """
        SELECT home_away, stat_name, stat_value FROM match_team_stats
        WHERE game_uid = ? AND stat_name IN ('totalPasses','accuratePasses','passPct')
        """,
        (game_uid,),
    )
    totals: dict[str, dict[str, float]] = {}
    for row in _records(aggregate):
        totals.setdefault(str(row["home_away"]), {})[str(row["stat_name"])] = row["stat_value"]

    return {
        "available": False,
        "state": "NO_DATA",
        "reason": ("No source in the platform publishes pass events. ESPN's summary "
                   "feed carries aggregate pass counts but no passer, recipient or "
                   "coordinates, so a passing network cannot be derived from it."),
        "requires": ["pass event stream", "passer and recipient identity",
                     "start and end coordinates", "completion outcome"],
        "passes": [],
        "aggregate_totals": totals,
        "note": ("Aggregate pass counts are shown in the statistics panel. They are "
                 "the only pass information that exists for this competition."),
    }


# -------------------------------------------------------------------- momentum

def match_momentum(game_uid: str, *, as_of: str | None = None,
                   minute: float | None = None) -> dict[str, Any]:
    events = match_events(game_uid, as_of=as_of, minute=minute)
    result = momentum_series(events, until_minute=minute)
    result["summary"] = momentum_summary(result)
    result["game_uid"] = game_uid
    result["as_of"] = as_of
    return result


# ---------------------------------------------------------------- the assembly

def match_center(game_uid: str, *, as_of: str | None = None, minute: float | None = None,
                 market: str = "1x2") -> dict[str, Any]:
    """One coherent payload for the whole Match Center."""
    game = load_game(game_uid)
    context = load_context(game_uid)
    bounds = resolve_bounds(game, as_of, minute,
                            first_event_observed_at=first_event_observation(game_uid))
    minute = bounds.minute
    bounded = minute is not None

    # The event stream is bounded by the match clock (and by observation time
    # only when a caller explicitly asked for one); everything derived from our
    # own observation history is bounded by the information cut-off.
    events = match_events(game_uid, as_of=bounds.observation, minute=minute)
    lineup_rows = lineups_for(game_uid, as_of=bounds.information)
    odds = market_for(game_uid, market=market, as_of=bounds.information)
    model = model_for(game_uid, as_of=bounds.information)
    state = match_state(game, context, minute=minute)

    if bounded:
        derived = _event_derived_stats(events)
        statistics = team_comparison(derived["rows"])
        statistics["basis"] = "recounted from events up to the replay position"
        statistics["unavailable"] = derived["unavailable"]
        statistics["note"] = (
            "Possession, passing and defensive counters are only published as "
            "full-match totals, so they are withheld at a replay position rather "
            "than shown as if they were the state at that minute."
        )
        team_rows: list[dict[str, Any]] = []
        player_stat_rows: list[dict[str, Any]] = []
    else:
        team_rows = team_stats_for(game_uid, as_of=as_of)
        statistics = team_comparison(team_rows)
        statistics["basis"] = "source box score"
        statistics["unavailable"] = []
        player_stat_rows = _records(query_df(
            "SELECT * FROM match_player_stats WHERE game_uid = ?", (game_uid,)
        ))

    home_score, away_score = _score_from_events(events)
    if not bounded and game.get("status") == "final":
        # Outside replay the stored result is authoritative; the event stream
        # is a reconstruction and a missing goal event should not change a
        # settled score.
        home_score = game.get("home_score") if game.get("home_score") is not None else home_score
        away_score = game.get("away_score") if game.get("away_score") is not None else away_score

    located = sum(1 for e in events if has_position(e.get("source_x"), e.get("source_y")))
    shots = shot_map(events)
    momentum = momentum_series(events, until_minute=minute)
    momentum["summary"] = momentum_summary(momentum)

    roster = [
        {**row, "home_away": ("home" if row.get("team_uid") == game.get("home_team_uid")
                              else "away")}
        for row in lineup_rows
    ]

    return {
        "game_uid": game_uid,
        "generated_at": _iso(datetime.now(timezone.utc)),
        "as_of": bounds.information,
        "replay_minute": minute,
        "replay": bounds.to_dict(),
        "match": {
            "game_uid": game_uid,
            "sport": game.get("sport"),
            "league_id": game.get("league_id"),
            "league_name": game.get("league_name"),
            "league_country": game.get("league_country"),
            "season": game.get("season"),
            "game_date": game.get("game_date"),
            "kickoff_utc": game.get("kickoff_utc"),
            "status": game.get("status"),
            "home": {
                "team_uid": game.get("home_team_uid"),
                "name": game.get("home_name"),
                "score": home_score,
                "color": (context or {}).get("home_color"),
                "logo": (context or {}).get("home_logo"),
                "form": (context or {}).get("home_form"),
                "formation": (context or {}).get("home_formation"),
            },
            "away": {
                "team_uid": game.get("away_team_uid"),
                "name": game.get("away_name"),
                "score": away_score,
                "color": (context or {}).get("away_color"),
                "logo": (context or {}).get("away_logo"),
                "form": (context or {}).get("away_form"),
                "formation": (context or {}).get("away_formation"),
            },
            "venue": (context or {}).get("venue") or game.get("venue"),
            "venue_city": (context or {}).get("venue_city"),
            "attendance": (context or {}).get("attendance"),
            "officials": (context or {}).get("officials") or [],
        },
        "state": state,
        "events": events,
        "statistics": statistics,
        "lineups": lineup_board(lineup_rows, game),
        "players": player_lines(player_stat_rows, roster),
        "contributions": contributions(events),
        "momentum": momentum,
        "shots": {
            "available": bool(shots),
            "points": [shot.to_dict() for shot in shots],
            "total_shots": sum(1 for e in events if str(e.get("event_type")) in SHOT_TYPES),
            "located": len(shots),
            "reason": None if shots else "no shot in this match carries a field position",
            "pitch": {"length": 105.0, "width": 68.0,
                      "orientation": "home attacks right"},
        },
        "heatmap": event_density(events),
        "passing": match_passes(game_uid),
        "market": odds,
        "model": model,
        "model_vs_market": model_vs_market(model, odds),
        "standings": standings_for(game),
        "quality": match_intelligence(
            events=events, team_stats=team_rows,
            player_stats_count=len(player_stat_rows), lineup_rows=lineup_rows,
            odds_snapshots=odds.get("snapshots", 0) if odds.get("available") else 0,
            predictions=model.get("count", 0), located_events=located, context=context,
        ),
        "provenance": {
            "events": (context or {}).get("source"),
            "events_observed_at": (context or {}).get("observed_at"),
            "events_retrieved_at": (context or {}).get("retrieved_at"),
            "market_sources": odds.get("sources") if odds.get("available") else [],
            "lineup_observed_at": next((row.get("observed_at") for row in lineup_rows), None),
            "retrospective_events": bounds.retrospective_events,
        },
    }
