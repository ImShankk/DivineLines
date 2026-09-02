"""Momentum: a transparent summary of observed events over time.

## What this is, and what it is not

Momentum here is a **descriptive** statistic. It compresses the event stream
into "who was pressing, when", so the timeline reads as a story rather than a
list. It is not a probability, it is not calibrated against anything, and it
has not been shown to predict a result. It is deliberately *not* a model
feature — a visually convincing curve is exactly the kind of thing that talks
its way into a champion model and quietly overfits it. If it is ever to become
one it goes through the same route as any other candidate: chronological
experiment first, promotion only on evidence.

## The definition

For a team, at match minute ``t``:

    momentum(t) = Σ  weight(event) · exp( -(t - t_event) / τ )
                 events with t_event ≤ t

so an event's influence decays with a half-life set by ``τ``. The reported
series is the home value minus the away value, scaled so the largest absolute
value in a typical match lands near 100; both raw sides are returned too.

Every parameter is in :data:`MOMENTUM_PARAMETERS` and every payload carries
them, so two momentum curves computed months apart can be compared or told
apart. Changing a weight means bumping :data:`MOMENTUM_VERSION`.

## Why these weights

They are a stated prior, not a fit — there is nothing to fit them against
without a target, and inventing one would be the dishonest move. The ordering
is the uncontroversial part: a goal outweighs a shot on target, which
outweighs a blocked shot, which outweighs a corner. The magnitudes are round
numbers chosen so that one goal is worth roughly two clear chances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

#: Bump on any change to weights, decay or the aggregation rule.
MOMENTUM_VERSION = "momentum/v1"

#: Event weight for the team that performed the event. A negative weight is a
#: team damaging its own position (conceding a card, being caught offside).
EVENT_WEIGHTS: dict[str, float] = {
    "goal": 10.0,
    "penalty_scored": 10.0,
    "own_goal": -6.0,
    "shot_on_target": 4.0,
    "penalty_saved": 4.0,
    "shot_woodwork": 3.5,
    "penalty_woodwork": 3.5,
    "shot_off_target": 1.6,
    "penalty_missed": 1.6,
    "shot_blocked": 1.5,
    "corner": 1.0,
    "free_kick": 0.3,
    "offside": -0.3,
    "handball": -0.3,
    "foul": -0.2,
    "yellow_card": -0.8,
    "red_card": -6.0,
}

#: Half-life of an event's influence, in match minutes. Eight minutes is a
#: judgement call: long enough that a spell of pressure reads as a spell,
#: short enough that a first-half goal does not still dominate at minute 80.
DECAY_HALF_LIFE_MINUTES = 8.0

#: Series resolution. One point per minute is plenty for a 90-minute match and
#: keeps the payload small enough to send whole.
STEP_MINUTES = 1.0

#: Events that are structural markers or retracted plays contribute nothing.
_IGNORED = {"kickoff", "halftime", "second_half_start", "full_time",
            "extra_time_start", "extra_time_end", "shootout", "deleted",
            "substitution", "var_decision", "other"}

MOMENTUM_PARAMETERS: dict[str, Any] = {
    "version": MOMENTUM_VERSION,
    "weights": EVENT_WEIGHTS,
    "decay_half_life_minutes": DECAY_HALF_LIFE_MINUTES,
    "step_minutes": STEP_MINUTES,
    "ignored_event_types": sorted(_IGNORED),
    "scale": "home minus away, multiplied by 6 for readability",
    "basis": "ESPN play-by-play events",
}


@dataclass
class MomentumPoint:
    minute: float
    home: float
    away: float
    net: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute": self.minute,
            "home": round(self.home, 3),
            "away": round(self.away, 3),
            "net": round(self.net, 3),
        }


def _decay(elapsed_minutes: float) -> float:
    if elapsed_minutes < 0:
        return 0.0
    return math.pow(0.5, elapsed_minutes / DECAY_HALF_LIFE_MINUTES)


def _contributing(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = []
    for event in events:
        event_type = str(event.get("event_type") or "")
        if event_type in _IGNORED:
            continue
        if EVENT_WEIGHTS.get(event_type) is None:
            continue
        if event.get("minute") is None:
            # Without a clock an event cannot be placed on a time axis. It
            # still exists in the timeline; it just cannot move this curve.
            continue
        if event.get("home_away") not in ("home", "away"):
            continue
        usable.append(event)
    return usable


def momentum_series(events: Sequence[dict[str, Any]], *,
                    until_minute: float | None = None) -> dict[str, Any]:
    """Momentum over the match, plus the events that drove each swing.

    ``until_minute`` truncates the series for replay. It is applied to the
    *events* as well as the axis, so a replay at minute 32 genuinely cannot
    feel a goal scored at minute 72 — the curve is computed from a filtered
    stream, not drawn short.
    """
    usable = _contributing(events)
    if until_minute is not None:
        usable = [e for e in usable if float(e["minute"]) <= until_minute]

    if not usable:
        return {
            "available": False,
            "reason": "no clocked events with a momentum weight for this match",
            "parameters": MOMENTUM_PARAMETERS,
            "series": [], "markers": [], "swings": [],
        }

    last_minute = max(float(e["minute"]) for e in usable)
    if until_minute is None:
        # Always run the axis to the end of a normal match so two matches are
        # visually comparable, unless the game itself ran longer.
        axis_end = max(last_minute, 90.0)
    else:
        # Run to the requested position, not to the last event before it. A
        # goal in the 32.9th minute has to be on the curve when the replay
        # sits at minute 33 — stopping the axis at the last event put it a
        # grid point short and the curve showed the score-before state.
        axis_end = float(until_minute)

    scale = 6.0
    series: list[MomentumPoint] = []
    minute = 0.0
    while minute <= axis_end + 1e-9:
        home = away = 0.0
        for event in usable:
            at = float(event["minute"])
            if at > minute:
                continue
            weight = EVENT_WEIGHTS[str(event["event_type"])] * _decay(minute - at)
            if event["home_away"] == "home":
                home += weight
            else:
                away += weight
        series.append(MomentumPoint(round(minute, 2), home * scale,
                                    away * scale, (home - away) * scale))
        minute += STEP_MINUTES

    return {
        "available": True,
        "parameters": MOMENTUM_PARAMETERS,
        "series": [point.to_dict() for point in series],
        "markers": _markers(usable),
        "swings": _swings(series, usable),
        "events_used": len(usable),
        "until_minute": until_minute,
        "note": ("A descriptive summary of recorded events, not a prediction and "
                 "not a model input. Weights and decay are a stated prior."),
    }


#: Events worth annotating directly on the chart.
_MARKER_TYPES = {"goal", "own_goal", "penalty_scored", "penalty_missed",
                 "penalty_saved", "red_card", "yellow_card", "substitution"}


def _markers(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "minute": float(event["minute"]),
            "clock_display": event.get("clock_display"),
            "event_type": event["event_type"],
            "home_away": event.get("home_away"),
            "team_uid": event.get("team_uid"),
            "player_name": event.get("player_name"),
            "event_row_id": event.get("event_row_id"),
            "text": event.get("short_text") or event.get("text"),
        }
        for event in events
        if str(event.get("event_type")) in _MARKER_TYPES
    ]


def _swings(series: Sequence[MomentumPoint], events: Sequence[dict[str, Any]],
            *, threshold: float = 4.0) -> list[dict[str, Any]]:
    """Minute-to-minute jumps large enough to be worth explaining.

    The event named is the one recorded in that minute — described as the
    associated event, never as the cause. The curve moves because that event
    entered the sum; whether the event *caused* the shift in the match is not
    something an event feed can establish.
    """
    by_minute: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_minute.setdefault(int(float(event["minute"])), []).append(event)

    swings: list[dict[str, Any]] = []
    for previous, current in zip(series, series[1:]):
        change = current.net - previous.net
        if abs(change) < threshold:
            continue
        associated = by_minute.get(int(current.minute), [])
        swings.append({
            "minute": current.minute,
            "net": round(current.net, 3),
            "change": round(change, 3),
            "direction": "home" if change > 0 else "away",
            "associated_events": [
                {
                    "event_type": event["event_type"],
                    "home_away": event.get("home_away"),
                    "player_name": event.get("player_name"),
                    "clock_display": event.get("clock_display"),
                    "text": event.get("short_text") or event.get("text"),
                }
                for event in associated
            ],
            "note": "associated event, not an established cause",
        })
    return swings


def momentum_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Headline figures for a momentum series, for cards and reports."""
    if not result.get("available"):
        return {"available": False, "reason": result.get("reason")}

    series = result["series"]
    net = [point["net"] for point in series]
    home_minutes = sum(1 for value in net if value > 0)
    return {
        "available": True,
        "peak_home": max(net),
        "peak_away": min(net),
        "final_net": net[-1] if net else 0.0,
        "minutes_home_ahead": home_minutes,
        "minutes_away_ahead": sum(1 for value in net if value < 0),
        "share_home": round(home_minutes / len(net), 4) if net else None,
        "version": MOMENTUM_VERSION,
    }
