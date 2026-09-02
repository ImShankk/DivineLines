"""What do we actually have for this match?

Every panel in the Match Center depends on a different feed, and they fail
independently: a match can have a full event stream and no price history, or
prices and no lineup. A single "data quality: 87%" number would hide exactly
the thing a reader needs to know, so this reports one line per component with
the count behind it.

Nothing here is scored on a curve. A component is either present with a
measured count, partial with a stated shortfall, or absent with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: A full match has both sides' XI, so 22 starters. Lower than that means the
#: lineup is partial, not that the feed is broken.
EXPECTED_STARTERS = 22


@dataclass
class Component:
    name: str
    label: str
    state: str          # present | partial | absent
    detail: str
    count: int | None = None
    coverage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "label": self.label, "state": self.state,
            "detail": self.detail, "count": self.count,
            "coverage": None if self.coverage is None else round(self.coverage, 4),
        }


def _state(count: int, *, partial_below: int | None = None) -> str:
    if count <= 0:
        return "absent"
    if partial_below is not None and count < partial_below:
        return "partial"
    return "present"


def match_intelligence(*, events: Sequence[dict[str, Any]],
                       team_stats: Sequence[dict[str, Any]],
                       player_stats_count: int,
                       lineup_rows: Sequence[dict[str, Any]],
                       odds_snapshots: int,
                       predictions: int,
                       located_events: int,
                       context: dict[str, Any] | None) -> dict[str, Any]:
    """A component-by-component account of what this match can support."""
    starters = sum(1 for row in lineup_rows if row.get("status") == "starter")
    lineup_state = next((row.get("lineup_state") for row in lineup_rows), None)

    components = [
        Component(
            "events", "Match events", _state(len(events)),
            f"{len(events)} events from the play-by-play feed" if events
            else "no play-by-play recorded for this match",
            count=len(events),
        ),
        Component(
            "spatial", "Event coordinates",
            _state(located_events, partial_below=max(1, len(events) // 4)),
            (f"{located_events} of {len(events)} events carry a field position"
             if events else "no events, so no positions"),
            count=located_events,
            coverage=(located_events / len(events)) if events else None,
        ),
        Component(
            "team_stats", "Team statistics", _state(len(team_stats)),
            f"{len(team_stats)} box-score values" if team_stats
            else "no box score published",
            count=len(team_stats),
        ),
        Component(
            "player_stats", "Player statistics", _state(player_stats_count),
            f"{player_stats_count} per-player values" if player_stats_count
            else "no per-player statistics published",
            count=player_stats_count,
        ),
        Component(
            "lineups", "Lineups", _state(starters, partial_below=EXPECTED_STARTERS),
            (f"{starters} starters recorded, state '{lineup_state}'" if starters
             else "no lineup observed for this fixture"),
            count=starters,
            coverage=starters / EXPECTED_STARTERS if starters else None,
        ),
        Component(
            "odds", "Market prices", _state(odds_snapshots),
            f"{odds_snapshots} price snapshots" if odds_snapshots
            else "no prices stored for this fixture",
            count=odds_snapshots,
        ),
        Component(
            "model", "Model predictions", _state(predictions),
            f"{predictions} stored predictions" if predictions
            else "no prediction was recorded for this fixture",
            count=predictions,
        ),
        Component(
            "context", "Match context", "present" if context else "absent",
            "venue, officials, attendance and status" if context
            else "no match context ingested",
        ),
        # Stated flatly rather than left as an empty panel: the reason a pass
        # network is missing is a property of the feed, not of this match.
        Component(
            "passing_network", "Passing network", "absent",
            "no source in the platform publishes pass events with coordinates",
        ),
        Component(
            "tracking", "Player tracking", "absent",
            "no tracking provider is configured; heatmaps are event-derived",
        ),
        Component(
            "expected_goals", "Expected goals", "absent",
            "no source publishes xG for this competition and none is estimated",
        ),
    ]

    present = sum(1 for component in components if component.state == "present")
    partial = sum(1 for component in components if component.state == "partial")
    return {
        "components": [component.to_dict() for component in components],
        "present": present,
        "partial": partial,
        "absent": len(components) - present - partial,
        "grade": _grade(components),
        "note": ("Component states, not a single score. A match can have a complete "
                 "event feed and no market history, and the difference matters."),
    }


#: Components that decide the headline grade. The three permanently-absent
#: rows (passing, tracking, xG) are informational — grading a match down for a
#: gap that affects every match in the platform would make the badge useless.
_GRADED = ("events", "team_stats", "lineups", "odds")


def _grade(components: Sequence[Component]) -> str:
    graded = [c for c in components if c.name in _GRADED]
    if not graded:
        return "unknown"
    present = sum(1 for c in graded if c.state == "present")
    if present == len(graded):
        return "high"
    if present == 0:
        return "none"
    return "partial"
