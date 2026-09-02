"""Pitch coordinates: one transform, in one place.

## The source frame

ESPN gives ball events a ``fieldPositionX``/``fieldPositionY`` pair in the
range 0–1. I worked the semantics out from the data rather than from
documentation, because there is none:

* ``X`` is measured **from the goal the acting team is attacking**. Shots have
  a median X of 0.24 and corners 0.09 (both near the attacked goal); fouls sit
  at 0.66, which is where you would expect a defending side to commit them.
  So X = 0 is the opponent's goal line and X = 1 is the acting team's own.
* ``Y`` runs across the width, 0.5 being the centre.
* ``fieldPosition2X/Y`` is the ball's end point and is only populated for some
  shots and offsides.
* ``goalPositionX/Y`` exists in the schema but is 0.0 in every payload I have,
  so shot placement inside the goal mouth is *not* available.

Because the frame is relative to whoever acted, two shots at opposite ends of
the pitch have the same X. Drawing them raw would put both teams' attacks in
the same half — which is why normalisation is not optional.

## The platform frame

Everything downstream uses a single frame:

    (0, 0) ────────────────────────────── (105, 0)
      │                                       │
      │   home attacks →     ← away attacks   │
      │                                       │
    (0, 68) ───────────────────────────── (105, 68)

X runs 0–105 metres left to right, Y runs 0–68 top to bottom, the home team
always attacks to the right. The raw source values stay in the database
untouched; if this reading of the frame turns out to be wrong, every derived
number can be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

#: Metres. A standard pitch, used so distances read in familiar units.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

#: Goal centres in the platform frame.
HOME_ATTACK_GOAL = (PITCH_LENGTH, PITCH_WIDTH / 2)
AWAY_ATTACK_GOAL = (0.0, PITCH_WIDTH / 2)

#: ESPN sends 0.0/0.0 for events it has no position for (cards, substitutions)
#: rather than omitting the field. A shot from the exact corner of the goal
#: line is not a thing, so treating the origin as "absent" is safe — and the
#: alternative, a cluster of phantom events in one corner, is not.
_ORIGIN_EPSILON = 1e-9


def has_position(x: float | None, y: float | None) -> bool:
    """Whether a source coordinate pair carries real information."""
    if x is None or y is None:
        return False
    return abs(x) > _ORIGIN_EPSILON or abs(y) > _ORIGIN_EPSILON


def normalise_point(x: float | None, y: float | None, home_away: str | None
                    ) -> tuple[float, float] | None:
    """Source coordinates -> the platform pitch frame.

    ``home_away`` is which side acted, because the source frame is relative to
    the actor. Returns ``None`` when the source had no position, so callers
    cannot accidentally plot a placeholder.
    """
    if not has_position(x, y):
        return None
    assert x is not None and y is not None  # narrowed by has_position

    if home_away == "home":
        # Home attacks right: X = 0 at the away goal, so flip the axis.
        px = (1.0 - float(x)) * PITCH_LENGTH
        py = float(y) * PITCH_WIDTH
    else:
        px = float(x) * PITCH_LENGTH
        py = (1.0 - float(y)) * PITCH_WIDTH

    return (
        min(max(px, 0.0), PITCH_LENGTH),
        min(max(py, 0.0), PITCH_WIDTH),
    )


def distance_to_goal(point: tuple[float, float], home_away: str | None) -> float:
    """Metres from a point to the goal that side is attacking."""
    goal = HOME_ATTACK_GOAL if home_away == "home" else AWAY_ATTACK_GOAL
    return ((point[0] - goal[0]) ** 2 + (point[1] - goal[1]) ** 2) ** 0.5


@dataclass
class ShotPoint:
    event_row_id: int | None
    minute: float | None
    clock_display: str | None
    team_uid: str | None
    home_away: str | None
    player_name: str | None
    event_type: str
    outcome: str            # goal | on_target | off_target | blocked | woodwork
    x: float
    y: float
    end_x: float | None
    end_y: float | None
    distance_m: float
    text: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_row_id": self.event_row_id,
            "minute": self.minute,
            "clock_display": self.clock_display,
            "team_uid": self.team_uid,
            "home_away": self.home_away,
            "player_name": self.player_name,
            "event_type": self.event_type,
            "outcome": self.outcome,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "end_x": None if self.end_x is None else round(self.end_x, 2),
            "end_y": None if self.end_y is None else round(self.end_y, 2),
            "distance_m": round(self.distance_m, 1),
            "text": self.text,
        }


#: How a shot ended, collapsed to the four states a reader actually cares
#: about on a shot map.
SHOT_OUTCOMES: dict[str, str] = {
    "goal": "goal",
    "own_goal": "goal",
    "penalty_scored": "goal",
    "shot_on_target": "on_target",
    "penalty_saved": "on_target",
    "shot_off_target": "off_target",
    "penalty_missed": "off_target",
    "shot_blocked": "blocked",
    "shot_woodwork": "woodwork",
    "penalty_woodwork": "woodwork",
}


def shot_map(events: Iterable[dict[str, Any]]) -> list[ShotPoint]:
    """Shots with a real position, in the platform frame.

    A shot ESPN did not locate is dropped from the map rather than placed
    somewhere convenient; the count of dropped shots is reported alongside so
    the map never silently disagrees with the statistics panel.
    """
    points: list[ShotPoint] = []
    for event in events:
        outcome = SHOT_OUTCOMES.get(str(event.get("event_type")))
        if outcome is None:
            continue
        home_away = event.get("home_away")
        start = normalise_point(event.get("source_x"), event.get("source_y"), home_away)
        if start is None:
            continue
        end = normalise_point(event.get("source_x2"), event.get("source_y2"), home_away)
        points.append(
            ShotPoint(
                event_row_id=event.get("event_row_id"),
                minute=event.get("minute"),
                clock_display=event.get("clock_display"),
                team_uid=event.get("team_uid"),
                home_away=home_away,
                player_name=event.get("player_name"),
                event_type=str(event.get("event_type")),
                outcome=outcome,
                x=start[0], y=start[1],
                end_x=None if end is None else end[0],
                end_y=None if end is None else end[1],
                distance_m=distance_to_goal(start, home_away),
                text=event.get("text"),
            )
        )
    return points


def event_density(events: Iterable[dict[str, Any]], *, side: str | None = None,
                  columns: int = 12, rows: int = 8,
                  event_types: Sequence[str] | None = None) -> dict[str, Any]:
    """Binned counts of located events across the pitch.

    This is an **event-location density**, not a tracking heatmap. It shows
    where the ball was when something was recorded, which is a far smaller
    claim than where players spent the match, and the payload says so in a
    field the UI is required to render.
    """
    grid = [[0 for _ in range(columns)] for _ in range(rows)]
    located = 0
    considered = 0
    wanted = set(event_types) if event_types else None

    for event in events:
        if wanted is not None and str(event.get("event_type")) not in wanted:
            continue
        if side and event.get("home_away") != side:
            continue
        considered += 1
        point = normalise_point(event.get("source_x"), event.get("source_y"),
                                event.get("home_away"))
        if point is None:
            continue
        located += 1
        column = min(int(point[0] / PITCH_LENGTH * columns), columns - 1)
        row = min(int(point[1] / PITCH_WIDTH * rows), rows - 1)
        grid[row][column] += 1

    peak = max((cell for line in grid for cell in line), default=0)
    return {
        "kind": "event_location_density",
        "basis": "ball position at recorded events",
        "not_tracking": True,
        "note": ("Derived from event coordinates, not player tracking. It shows "
                 "where recorded events happened, not where players were."),
        "columns": columns,
        "rows": rows,
        "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
        "grid": grid,
        "peak": peak,
        "events_considered": considered,
        "events_located": located,
        "coverage": round(located / considered, 4) if considered else None,
    }


def average_position(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )
