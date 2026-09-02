"""A structured match report: the same material, as something readable.

The report is assembled from the Match Center payload rather than by querying
again, so the prose and the panels cannot disagree. Every section states what
it is missing instead of omitting itself, because "no shot locations for this
match" is information and a silently absent section is not.

Nothing here narrates a cause. "Momentum swung to the away side around the
33rd minute, when a goal was recorded" is a description of two facts and their
order. "The goal shifted the game" is a claim this platform cannot support.
"""

from __future__ import annotations

from typing import Any, Sequence

from .service import match_center


def _score_line(match: dict[str, Any]) -> str:
    home, away = match["home"], match["away"]
    return f"{home['name']} {home['score']}–{away['score']} {away['name']}"


def _key_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"goal", "own_goal", "penalty_scored", "penalty_missed",
              "penalty_saved", "red_card", "var_decision"}
    return [
        {
            "minute": event.get("clock_display") or event.get("minute"),
            "event_type": event.get("event_type"),
            "team": event.get("team_name"),
            "player": event.get("player_name"),
            "assist": event.get("assist_player_name"),
            "score": (f"{event.get('home_score')}–{event.get('away_score')}"
                      if event.get("home_score") is not None else None),
            "text": event.get("text"),
        }
        for event in events
        if str(event.get("event_type")) in wanted
    ]


def _headline_stats(statistics: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = {"possessionPct", "totalShots", "shotsOnTarget", "wonCorners",
              "passPct", "foulsCommitted"}
    return [row for row in statistics.get("comparisons", []) if row["stat"] in wanted]


def _momentum_prose(momentum: dict[str, Any]) -> str:
    summary = momentum.get("summary") or {}
    if not summary.get("available"):
        return "No momentum series could be derived: " + str(
            momentum.get("reason") or "no clocked events."
        )
    share = summary.get("share_home")
    if share is None:
        return "Momentum was recorded but the series is empty."
    if share > 0.6:
        shape = "the home side held the positive side of the curve for most of the match"
    elif share < 0.4:
        shape = "the away side held the positive side of the curve for most of the match"
    else:
        shape = "the curve changed hands repeatedly"
    return (
        f"On the {summary['version']} definition, {shape} "
        f"({summary['minutes_home_ahead']} minutes home, "
        f"{summary['minutes_away_ahead']} minutes away). "
        f"The largest readings on the net curve were {summary['peak_home']:+.0f} "
        f"toward home and {summary['peak_away']:+.0f} toward away. "
        "The curve is a weighted, time-decayed summary of recorded events; it is "
        "descriptive and is not a model input."
    )


def _market_prose(market: dict[str, Any]) -> str:
    if not market.get("available"):
        return f"No market history: {market.get('reason')}."
    opening = (market.get("opening") or {}).get("consensus") or {}
    closing = ((market.get("closing") or market.get("latest")) or {}).get("consensus") or {}
    if not opening or not closing:
        return f"{market['snapshots']} price snapshots recorded across {market['books']} books."
    moves = []
    for selection in sorted(set(opening) & set(closing)):
        start, end = opening[selection], closing[selection]
        direction = "drifted" if end > start else ("shortened" if end < start else "held")
        moves.append(f"{selection} {direction} {start:.2f} → {end:.2f}")
    return (
        f"{market['snapshots']} snapshots across {market['books']} books. "
        + "; ".join(moves)
        + ". Consensus prices; best available prices are reported separately."
    )


def _model_prose(model: dict[str, Any], comparison: dict[str, Any]) -> str:
    if not model.get("available"):
        return ("No prediction was recorded for this fixture, so there is nothing "
                "to compare against the market.")
    if not comparison.get("available"):
        return (f"{model['count']} predictions stored, but "
                f"{comparison.get('reason')}.")
    parts = [
        f"{row['selection']}: model {row['model_probability']:.1%} vs market "
        f"{row['market_probability']:.1%} ({row['difference_points']:+.1f}pp)"
        for row in comparison["rows"]
    ]
    return ("; ".join(parts) +
            ". A disagreement is not an edge; staking is decided under model-health "
            "limits by the betting engine.")


def _limitations(payload: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for component in payload["quality"]["components"]:
        if component["state"] == "absent":
            notes.append(f"{component['label']}: {component['detail']}.")
        elif component["state"] == "partial":
            notes.append(f"{component['label']} is partial — {component['detail']}.")
    shots = payload["shots"]
    if shots.get("total_shots") and shots["located"] < shots["total_shots"]:
        notes.append(
            f"{shots['total_shots'] - shots['located']} of {shots['total_shots']} shots "
            "carry no field position and are absent from the shot map, though they "
            "are counted in the statistics."
        )
    if payload.get("replay", {}).get("retrospective_events"):
        notes.append(
            "The event stream for this match was ingested after full time, so a "
            "replay reconstructs what happened by a given minute — not what the "
            "platform knew at that minute."
        )
    return notes


def match_report(game_uid: str, *, as_of: str | None = None,
                 minute: float | None = None, market: str = "1x2") -> dict[str, Any]:
    """A section-by-section account of one match."""
    payload = match_center(game_uid, as_of=as_of, minute=minute, market=market)
    match = payload["match"]
    contributions = payload["contributions"]

    return {
        "game_uid": game_uid,
        "generated_at": payload["generated_at"],
        "replay": payload["replay"],
        "result": {
            "headline": _score_line(match),
            "state": payload["state"]["state"],
            "competition": match.get("league_name") or match.get("league_id"),
            "season": match.get("season"),
            "kickoff_utc": match.get("kickoff_utc"),
            "venue": match.get("venue"),
            "attendance": match.get("attendance"),
            "officials": match.get("officials"),
        },
        "key_events": _key_events(payload["events"]),
        "scorers": contributions["goals"],
        "assists": contributions["assists"],
        "momentum": {
            "prose": _momentum_prose(payload["momentum"]),
            "summary": payload["momentum"].get("summary"),
            "largest_swings": (payload["momentum"].get("swings") or [])[:5],
        },
        "statistics": {
            "headline": _headline_stats(payload["statistics"]),
            "basis": payload["statistics"].get("basis"),
            "unavailable": payload["statistics"].get("unavailable", []),
        },
        "shooting": {
            "total_shots": payload["shots"]["total_shots"],
            "located": payload["shots"]["located"],
            "by_outcome": _shot_breakdown(payload["shots"]["points"]),
            "note": ("Shot locations come from the event feed. No expected-goals "
                     "figure is published for this competition and none is estimated."),
        },
        "passing": {
            "state": payload["passing"]["state"],
            "reason": payload["passing"]["reason"],
            "aggregate_totals": payload["passing"]["aggregate_totals"],
        },
        "market": {"prose": _market_prose(payload["market"]),
                   "snapshots": payload["market"].get("snapshots", 0)},
        "model": {"prose": _model_prose(payload["model"], payload["model_vs_market"]),
                  "predictions": payload["model"].get("count", 0)},
        "data_quality": payload["quality"],
        "limitations": _limitations(payload),
    }


def _shot_breakdown(points: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    breakdown: dict[str, dict[str, int]] = {"home": {}, "away": {}}
    for point in points:
        side = point.get("home_away")
        if side not in breakdown:
            continue
        outcome = str(point.get("outcome"))
        breakdown[side][outcome] = breakdown[side].get(outcome, 0) + 1
    return breakdown
