"""Match statistics: team comparisons and position-aware player lines.

The feed's stat names are ESPN's. They are kept verbatim in the database and
given readable labels here, so a metric ESPN renames does not silently vanish
from a panel — it appears with its raw name until someone maps it.

Two things this module refuses to do:

* invent expected goals. The feed has no xG field. What *can* be computed from
  what is there is a shot count and a shot-location distribution, and those are
  presented as themselves.
* compare a goalkeeper to a striker on the same row. Player lines are grouped
  by position, and the metrics shown follow the position group.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

#: Team metrics worth showing, in display order, with the label the panel uses
#: and whether a higher number is better. ``None`` means neither direction is
#: obviously good — fouls are not a virtue, but nor is a low count a target.
TEAM_STAT_DISPLAY: tuple[tuple[str, str, str, bool | None], ...] = (
    ("possessionPct", "Possession", "percent", True),
    ("totalShots", "Shots", "count", True),
    ("shotsOnTarget", "Shots on target", "count", True),
    ("blockedShots", "Blocked shots", "count", None),
    ("wonCorners", "Corners", "count", True),
    ("offsides", "Offsides", "count", None),
    ("totalPasses", "Passes", "count", True),
    ("accuratePasses", "Accurate passes", "count", True),
    ("passPct", "Pass accuracy", "ratio", True),
    ("totalCrosses", "Crosses", "count", None),
    ("accurateCrosses", "Accurate crosses", "count", True),
    ("totalLongBalls", "Long balls", "count", None),
    ("accurateLongBalls", "Accurate long balls", "count", True),
    ("totalTackles", "Tackles", "count", None),
    ("effectiveTackles", "Tackles won", "count", True),
    ("interceptions", "Interceptions", "count", True),
    ("totalClearance", "Clearances", "count", None),
    ("saves", "Saves", "count", True),
    ("foulsCommitted", "Fouls", "count", None),
    ("yellowCards", "Yellow cards", "count", None),
    ("redCards", "Red cards", "count", None),
)

#: Per-player metrics by position group. A keeper's line is about shot
#: stopping; an outfielder's is about what they did with the ball.
PLAYER_STAT_DISPLAY: dict[str, tuple[tuple[str, str], ...]] = {
    "goalkeeper": (
        ("saves", "Saves"),
        ("shotsFaced", "Shots faced"),
        ("goalsConceded", "Conceded"),
        ("foulsCommitted", "Fouls"),
        ("yellowCards", "YC"),
        ("redCards", "RC"),
    ),
    "outfield": (
        ("totalGoals", "Goals"),
        ("goalAssists", "Assists"),
        ("totalShots", "Shots"),
        ("shotsOnTarget", "On target"),
        ("offsides", "Offsides"),
        ("foulsCommitted", "Fouls"),
        ("foulsSuffered", "Fouled"),
        ("yellowCards", "YC"),
        ("redCards", "RC"),
    ),
}

#: Ratios arrive from ESPN as 0–1 in some fields and 0–100 in others
#: (``passPct`` is 0.8, ``possessionPct`` is 61.1). Normalising on read means
#: the frontend never has to guess which convention a metric follows.
_FRACTION_STATS = {"passPct", "shotPct", "crossPct", "longballPct", "tacklePct"}


def _percent(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if name in _FRACTION_STATS else value


def team_comparison(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Home-vs-away table from long-form stat rows.

    Only metrics both sides reported are compared: a one-sided row would draw
    a bar against a blank, which reads as a zero rather than as a gap in the
    feed.
    """
    by_side: dict[str, dict[str, dict[str, Any]]] = {"home": {}, "away": {}}
    for row in rows:
        side = row.get("home_away")
        if side not in by_side:
            continue
        by_side[side][str(row.get("stat_name"))] = row

    comparisons: list[dict[str, Any]] = []
    for name, label, kind, higher_is_better in TEAM_STAT_DISPLAY:
        home_row = by_side["home"].get(name)
        away_row = by_side["away"].get(name)
        if home_row is None and away_row is None:
            continue
        home_value = _percent(name, _as_float(home_row))
        away_value = _percent(name, _as_float(away_row))
        if home_value is None and away_value is None:
            continue
        total = (home_value or 0.0) + (away_value or 0.0)
        comparisons.append({
            "stat": name,
            "label": label,
            "kind": kind,
            "higher_is_better": higher_is_better,
            "home": home_value,
            "away": away_value,
            "home_display": (home_row or {}).get("display_value"),
            "away_display": (away_row or {}).get("display_value"),
            # Share of the pair, for a two-sided bar. Undefined when both are
            # zero, which the UI renders as an even split rather than 100/0.
            "home_share": round((home_value or 0.0) / total, 4) if total else None,
        })

    unmapped = sorted(
        (set(by_side["home"]) | set(by_side["away"]))
        - {name for name, _, _, _ in TEAM_STAT_DISPLAY}
    )
    return {
        "available": bool(comparisons),
        "comparisons": comparisons,
        "unmapped_stats": unmapped,
        "note": ("Source-reported box statistics. No expected-goals figure is "
                 "published by this feed, so none is shown."),
    }


def _as_float(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    value = row.get("stat_value")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def player_lines(stat_rows: Iterable[dict[str, Any]],
                 roster: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One line per player, with the metrics their position calls for.

    The roster drives the list, not the stat table: a substitute who never
    touched the ball still played and should appear with zeros rather than
    disappear.
    """
    stats_by_player: dict[str, dict[str, Any]] = {}
    for row in stat_rows:
        key = str(row.get("player_uid"))
        stats_by_player.setdefault(key, {})[str(row.get("stat_name"))] = row

    lines: list[dict[str, Any]] = []
    for entry in roster:
        player_uid = str(entry.get("player_uid") or "")
        group = entry.get("position_group") or "outfield"
        display = PLAYER_STAT_DISPLAY.get(
            "goalkeeper" if group == "goalkeeper" else "outfield"
        )
        available = stats_by_player.get(player_uid, {})
        values = []
        for name, label in display:
            row = available.get(name)
            values.append({
                "stat": name,
                "label": label,
                "value": _as_float(row),
                "display": (row or {}).get("display_value"),
            })
        lines.append({
            "player_uid": player_uid or None,
            "player_name": entry.get("player_name"),
            "team_uid": entry.get("team_uid"),
            "home_away": entry.get("home_away"),
            "jersey": entry.get("jersey"),
            "position": entry.get("role"),
            "position_group": group,
            "formation_place": entry.get("formation_place"),
            "starter": bool(entry.get("status") == "starter"),
            "subbed_in": bool(entry.get("subbed_in")),
            "subbed_out": bool(entry.get("subbed_out")),
            "stats": values,
            "has_stats": bool(available),
        })

    lines.sort(key=lambda line: (
        line["home_away"] or "",
        0 if line["starter"] else 1,
        _place_order(line["formation_place"]),
        line["player_name"] or "",
    ))
    return {
        "available": bool(lines),
        "players": lines,
        "rated": False,
        "rating_note": ("No player rating is published by this feed and DivineLines "
                        "does not compute one, so no rating is shown."),
    }


def _place_order(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 99.0


def contributions(events: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Goals and assists per player, derived from the event stream.

    The box score carries these too, but deriving them here keeps the timeline
    and the player panel consistent: if an event is filtered out by replay, the
    goal it records disappears from both.
    """
    goals: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    assists: dict[tuple[str | None, str | None], dict[str, Any]] = {}

    for event in events:
        event_type = str(event.get("event_type"))
        if event_type not in ("goal", "penalty_scored", "own_goal"):
            continue
        key = (event.get("player_uid"), event.get("player_name"))
        entry = goals.setdefault(key, {
            "player_uid": key[0], "player_name": key[1],
            "team_uid": event.get("team_uid"), "home_away": event.get("home_away"),
            "goals": 0, "own_goals": 0, "minutes": [],
        })
        if event_type == "own_goal":
            entry["own_goals"] += 1
        else:
            entry["goals"] += 1
        entry["minutes"].append(event.get("clock_display") or event.get("minute"))

        assist_name = event.get("assist_player_name")
        if assist_name and event_type != "own_goal":
            assist_key = (event.get("assist_player_uid"), assist_name)
            assist = assists.setdefault(assist_key, {
                "player_uid": assist_key[0], "player_name": assist_name,
                "team_uid": event.get("team_uid"), "home_away": event.get("home_away"),
                "assists": 0,
            })
            assist["assists"] += 1

    return {
        "goals": sorted(goals.values(), key=lambda row: -row["goals"]),
        "assists": sorted(assists.values(), key=lambda row: -row["assists"]),
    }
