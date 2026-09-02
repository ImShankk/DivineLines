"""ESPN match-detail adapter: events, box statistics, match context.

This reads the *same* summary document that :mod:`espn_lineups` already
fetches, which is why it shares that adapter's cache namespace — 760 matches
are already on disk and re-downloading them to read a different part of the
same JSON would be wasteful and rude to ESPN.

## What this feed actually contains

I audited every cached payload before writing a line of visualisation code,
because the difference between "we have match events" and "we have match data"
decides which panels can exist at all:

* ``commentary[].play`` — the full play-by-play. Shots (on target, off target,
  blocked, woodwork), goals, corners, fouls, offsides, cards, substitutions,
  VAR decisions. Each carries a period, a cumulative match clock, a real UTC
  wallclock, the team, the players involved, and — for ball events — a field
  position.
* ``boxscore.teams[].statistics`` — 28 team metrics including possession,
  total/accurate passes, tackles, crosses and long balls.
* ``rosters[].roster[].stats`` — 15 per-player metrics.
* ``gameInfo`` / ``header`` — venue, attendance, officials, status, score.

## What it does not contain, and therefore what cannot be drawn

* **No pass events.** There is no passer→receiver record anywhere in this
  feed, so a passing network cannot be derived from it. The aggregate
  ``totalPasses``/``accuratePasses`` counters are all that exists.
* **No player tracking.** Positions exist only for the ball at discrete
  events, so a true player heatmap is not derivable — only an event-location
  density, which is a different thing and is labelled as one.
* **No xG and no player ratings.** Neither field appears in any payload.

Coordinates are stored exactly as ESPN sends them. ``fieldPositionX`` is
measured from the goal the acting team is attacking (0.0 at that goal line)
and ``fieldPositionY`` across the width. Normalisation into a drawable pitch
frame happens in :mod:`divinelines.matchcenter.spatial`, once, so there is
only ever one place where the transform can be wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import FRESHNESS_TTL
from ..identity import soccer_player_uid
from ..logging_setup import get_logger
from .base import HttpSource, SourceError
from .espn_lineups import SITE_API, SPORT_PATHS, position_group_for

log = get_logger(__name__)

#: ESPN play types -> the platform's taxonomy. Anything unmapped keeps a
#: slugified version of the source label and is counted as ``other``: silently
#: dropping an unknown event type would make a match look emptier than it was.
EVENT_TYPES: dict[str, str] = {
    "goal": "goal",
    "goal---header": "goal",
    "goal---volley": "goal",
    "goal---free-kick": "goal",
    "goal---solo-run": "goal",
    "own-goal": "own_goal",
    "penalty---scored": "penalty_scored",
    "penalty---missed": "penalty_missed",
    "penalty---saved": "penalty_saved",
    "penalty---hit-woodwork": "penalty_woodwork",
    "shot-on-target": "shot_on_target",
    "shot-off-target": "shot_off_target",
    "shot-blocked": "shot_blocked",
    "shot-hit-woodwork": "shot_woodwork",
    "corner-awarded": "corner",
    "save": "save",
    "blocked-pass": "blocked_pass",
    "foul": "foul",
    "handball": "handball",
    "offside": "offside",
    "free-kick": "free_kick",
    "yellow-card": "yellow_card",
    "red-card": "red_card",
    "second-yellow-card---red": "red_card",
    "substitution": "substitution",
    "kickoff": "kickoff",
    "halftime": "halftime",
    "start-2nd-half": "second_half_start",
    "end-regular-time": "full_time",
    "end-extra-time": "extra_time_end",
    "start-extra-time": "extra_time_start",
    "penalty-shootout": "shootout",
    "var---referee-decision-cancelled": "var_decision",
    "var---referee-decision-confirmed": "var_decision",
    "deleted-after-review": "deleted",
}

#: Every event that represents an attempt on goal, in one place, so momentum,
#: the shot map and the statistics panel cannot drift apart.
SHOT_TYPES: frozenset[str] = frozenset({
    "goal", "own_goal", "penalty_scored", "penalty_missed", "penalty_saved",
    "penalty_woodwork", "shot_on_target", "shot_off_target", "shot_blocked",
    "shot_woodwork",
})

#: Structural markers, not things a team did.
PERIOD_TYPES: frozenset[str] = frozenset({
    "kickoff", "halftime", "second_half_start", "full_time",
    "extra_time_start", "extra_time_end", "shootout",
})

#: A play ESPN retracted after review. Kept for provenance, excluded from
#: every aggregate — counting a disallowed goal would be a lie with a source.
VOID_TYPES: frozenset[str] = frozenset({"deleted"})

#: ESPN's status names -> the platform's normalised match state. The frontend
#: renders from this, so status logic lives here rather than scattered across
#: React components.
MATCH_STATES: dict[str, str] = {
    "STATUS_SCHEDULED": "SCHEDULED",
    "STATUS_PRE_GAME": "SCHEDULED",
    "STATUS_FIRST_HALF": "LIVE_FIRST_HALF",
    "STATUS_IN_PROGRESS": "LIVE_FIRST_HALF",
    "STATUS_HALFTIME": "HALFTIME",
    "STATUS_SECOND_HALF": "LIVE_SECOND_HALF",
    "STATUS_END_OF_REGULATION": "LIVE_SECOND_HALF",
    "STATUS_FIRST_EXTRA_TIME": "EXTRA_TIME",
    "STATUS_SECOND_EXTRA_TIME": "EXTRA_TIME",
    "STATUS_HALFTIME_ET": "EXTRA_TIME",
    "STATUS_END_OF_EXTRATIME": "EXTRA_TIME",
    "STATUS_SHOOTOUT": "PENALTIES",
    "STATUS_FULL_TIME": "FINISHED",
    "STATUS_FINAL": "FINISHED",
    "STATUS_FINAL_AET": "FINISHED",
    "STATUS_FINAL_PEN": "FINISHED",
    "STATUS_POSTPONED": "POSTPONED",
    "STATUS_SUSPENDED": "POSTPONED",
    "STATUS_DELAYED": "POSTPONED",
    "STATUS_CANCELED": "CANCELLED",
    "STATUS_ABANDONED": "CANCELLED",
    "STATUS_FORFEIT": "CANCELLED",
}


def normalise_state(status_name: str | None, status_state: str | None) -> str:
    """Map a source status onto the platform's match state.

    Falling back on ``state`` ('pre'/'in'/'post') matters: ESPN adds status
    names for competitions I have never fetched, and an unknown name should
    still land somewhere sensible instead of rendering as "LIVE" by accident.
    """
    if status_name and status_name in MATCH_STATES:
        return MATCH_STATES[status_name]
    fallback = {"pre": "SCHEDULED", "in": "LIVE_SECOND_HALF", "post": "FINISHED"}
    return fallback.get((status_state or "").lower(), "SCHEDULED")


def _slug(value: str | None) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (value or "").lower()).strip("_")


def event_type_for(source_type: str | None) -> str:
    if not source_type:
        return "other"
    key = str(source_type).strip().lower()
    if key in EVENT_TYPES:
        return EVENT_TYPES[key]
    # Unmapped but recognisably a goal or a shot: better a coarse bucket than
    # a silent drop, and the raw label is stored next to it either way.
    if key.startswith("goal"):
        return "goal"
    if key.startswith("shot"):
        return "shot_off_target"
    return _slug(key) or "other"


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


@dataclass
class MatchEvent:
    external_id: str | None
    sequence: int
    event_type: str
    source_type: str | None
    period: int | None
    clock_seconds: float | None
    clock_display: str | None
    minute: float | None
    wallclock_utc: str | None
    team_name: str | None
    player_name: str | None
    assist_player_name: str | None
    scoring_play: bool
    source_x: float | None
    source_y: float | None
    source_x2: float | None
    source_y2: float | None
    text: str | None
    short_text: str | None

    @property
    def is_shot(self) -> bool:
        return self.event_type in SHOT_TYPES


@dataclass
class TeamStatLine:
    team_name: str
    home_away: str
    stats: dict[str, tuple[float | None, str | None]] = field(default_factory=dict)


@dataclass
class PlayerLine:
    player_name: str
    external_player_id: str | None
    team_name: str
    home_away: str
    jersey: str | None
    position: str | None
    position_group: str | None
    formation_place: str | None
    starter: bool
    subbed_in: bool
    subbed_out: bool
    stats: dict[str, tuple[float | None, str | None]] = field(default_factory=dict)

    @property
    def player_uid(self) -> str | None:
        return soccer_player_uid(self.player_name, self.external_player_id)


@dataclass
class MatchContext:
    status_state: str | None
    status_name: str | None
    status_detail: str | None
    match_state: str
    period: int | None
    clock_display: str | None
    venue: str | None
    venue_city: str | None
    venue_country: str | None
    attendance: int | None
    officials: list[str]
    home_formation: str | None
    away_formation: str | None
    home_color: str | None
    away_color: str | None
    home_logo: str | None
    away_logo: str | None
    home_form: str | None
    away_form: str | None
    home_score: int | None
    away_score: int | None
    home_name: str | None
    away_name: str | None
    kickoff_utc: str | None


@dataclass
class MatchDetail:
    espn_event_id: str
    league_id: str
    context: MatchContext
    events: list[MatchEvent]
    team_stats: list[TeamStatLine]
    players: list[PlayerLine]
    standings: list[dict[str, Any]]
    observed_at: datetime
    retrieved_at: datetime
    from_cache: bool
    source: str = "espn_match"

    def to_provenance(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "espn_event_id": self.espn_event_id,
            "observed_at": self.observed_at.isoformat(timespec="seconds"),
            "retrieved_at": self.retrieved_at.isoformat(timespec="seconds"),
            "from_cache": self.from_cache,
        }


class EspnMatchSource(HttpSource):
    """Match detail from ESPN's summary document."""

    name = "espn_match"
    #: Deliberately the same on-disk cache as the lineup adapter: it is one
    #: upstream document, and two caches would mean two downloads.
    cache_namespace = "espn_lineups"
    cache_ttl = FRESHNESS_TTL["lineups"]
    user_agent = (
        "Mozilla/5.0 (compatible; DivineLines/5.0; research; "
        "+https://github.com/ImShankk/DivineLines)"
    )
    min_interval = 0.5

    def fetch_detail(self, league_id: str, espn_event_id: str, *,
                     event_started: bool = False, force: bool = False) -> MatchDetail:
        if league_id not in SPORT_PATHS:
            raise SourceError(f"espn_match: no ESPN path for '{league_id}'")
        group, league = SPORT_PATHS[league_id]

        result = self.fetch_json(
            f"{SITE_API}/{group}/{league}/summary",
            dataset=f"match:{league}:{espn_event_id}",
            status_dataset=f"match_detail:{league}",
            params={"event": espn_event_id},
            ttl=FRESHNESS_TTL["reference"] if event_started else self.cache_ttl,
            force=force,
        )
        payload = result.data if isinstance(result.data, dict) else {}
        if not payload:
            raise SourceError(f"espn_match: empty summary for event {espn_event_id}")

        observed = datetime.now(timezone.utc) if not result.from_cache else result.retrieved_at
        return MatchDetail(
            espn_event_id=str(espn_event_id),
            league_id=league_id,
            context=parse_context(payload),
            events=parse_events(payload),
            team_stats=parse_team_stats(payload),
            players=parse_players(payload),
            standings=parse_standings(payload),
            observed_at=observed,
            retrieved_at=result.retrieved_at,
            from_cache=result.from_cache,
        )


# ---------------------------------------------------------------------------
# Parsers — pure functions over a payload, so the contract tests can run them
# against a stored fixture without touching the network.
# ---------------------------------------------------------------------------

def _competition(payload: dict[str, Any]) -> dict[str, Any]:
    competitions = ((payload.get("header") or {}).get("competitions")) or []
    return competitions[0] if competitions else {}


def parse_context(payload: dict[str, Any]) -> MatchContext:
    competition = _competition(payload)
    status = (competition.get("status") or {}).get("type") or {}
    game_info = payload.get("gameInfo") or {}
    venue = game_info.get("venue") or {}
    address = venue.get("address") or {}

    competitors = {c.get("homeAway"): c for c in (competition.get("competitors") or [])}
    rosters = {r.get("homeAway"): r for r in (payload.get("rosters") or [])}

    def score(side: str) -> int | None:
        raw = (competitors.get(side) or {}).get("score")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def team_field(side: str, key: str) -> Any:
        return ((competitors.get(side) or {}).get("team") or {}).get(key)

    def logo(side: str) -> str | None:
        logos = team_field(side, "logos") or []
        return logos[0].get("href") if logos else None

    return MatchContext(
        status_state=status.get("state"),
        status_name=status.get("name"),
        status_detail=status.get("detail") or status.get("description"),
        match_state=normalise_state(status.get("name"), status.get("state")),
        period=(competition.get("status") or {}).get("period"),
        clock_display=(competition.get("status") or {}).get("displayClock"),
        venue=venue.get("fullName"),
        venue_city=address.get("city"),
        venue_country=address.get("country"),
        attendance=game_info.get("attendance"),
        officials=[o.get("displayName") or o.get("fullName")
                   for o in (game_info.get("officials") or [])
                   if o.get("displayName") or o.get("fullName")],
        home_formation=(rosters.get("home") or {}).get("formation"),
        away_formation=(rosters.get("away") or {}).get("formation"),
        home_color=team_field("home", "color"),
        away_color=team_field("away", "color"),
        home_logo=logo("home"),
        away_logo=logo("away"),
        home_form=team_field("home", "form"),
        away_form=team_field("away", "form"),
        home_score=score("home"),
        away_score=score("away"),
        home_name=team_field("home", "displayName"),
        away_name=team_field("away", "displayName"),
        kickoff_utc=competition.get("date"),
    )


def _play_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Every distinct play, preferring the richer of two overlapping feeds.

    ``commentary`` repeats each play once per narrative line — a foul appears
    twice, from the fouler's and the victim's point of view — so plays are
    de-duplicated by id. ``keyEvents`` is a subset but carries athlete ids that
    the commentary copy omits, so it is merged in rather than ignored.
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def absorb(play: dict[str, Any]) -> None:
        # Narrative-only commentary lines ("Lineups are announced") carry no
        # play object. They are prose, not events, and must not become rows.
        if not isinstance(play, dict) or not play.get("type"):
            return
        key = str(play.get("id") or f"anon:{len(order)}")
        if key in merged:
            existing = merged[key]
            for name, value in play.items():
                # Fill gaps only; never overwrite a value the first copy had.
                if existing.get(name) in (None, "", [], {}) and value not in (None, "", [], {}):
                    existing[name] = value
            return
        merged[key] = dict(play)
        order.append(key)

    for entry in payload.get("commentary") or []:
        absorb(entry.get("play") or {})
    for entry in payload.get("keyEvents") or []:
        absorb(entry)

    # keyEvents is appended after the commentary stream, so period markers
    # would otherwise land at the end of the match with sequence numbers that
    # imply kick-off happened after full time. Sort into real match order.
    def when(key: str) -> tuple[float, float, int]:
        play = merged[key]
        period = (play.get("period") or {}).get("number")
        clock = (play.get("clock") or {}).get("value")
        return (
            float(period) if period is not None else 0.0,
            float(clock) if clock is not None else 0.0,
            order.index(key),
        )

    return [merged[key] for key in sorted(order, key=when)]


def parse_events(payload: dict[str, Any]) -> list[MatchEvent]:
    events: list[MatchEvent] = []
    for index, play in enumerate(_play_records(payload)):
        source_type = (play.get("type") or {}).get("type")
        clock = play.get("clock") or {}
        seconds = _float(clock.get("value"))
        participants = [
            (p.get("athlete") or {}).get("displayName")
            for p in (play.get("participants") or [])
        ]
        participants = [p for p in participants if p]

        events.append(
            MatchEvent(
                external_id=str(play["id"]) if play.get("id") else None,
                sequence=index,
                event_type=event_type_for(source_type),
                source_type=source_type,
                period=(play.get("period") or {}).get("number"),
                clock_seconds=seconds,
                clock_display=clock.get("displayValue") or None,
                # ESPN's clock counts from kick-off across both halves, so
                # added time is already folded in and needs no arithmetic.
                minute=None if seconds is None else round(seconds / 60.0, 2),
                wallclock_utc=play.get("wallclock"),
                team_name=(play.get("team") or {}).get("displayName"),
                player_name=participants[0] if participants else None,
                assist_player_name=participants[1] if len(participants) > 1 else None,
                scoring_play=bool(play.get("scoringPlay")),
                source_x=_float(play.get("fieldPositionX")),
                source_y=_float(play.get("fieldPositionY")),
                source_x2=_float(play.get("fieldPosition2X")),
                source_y2=_float(play.get("fieldPosition2Y")),
                text=play.get("text"),
                short_text=play.get("shortText"),
            )
        )
    return events


def parse_team_stats(payload: dict[str, Any]) -> list[TeamStatLine]:
    lines: list[TeamStatLine] = []
    for block in (payload.get("boxscore") or {}).get("teams") or []:
        stats: dict[str, tuple[float | None, str | None]] = {}
        for stat in block.get("statistics") or []:
            name = stat.get("name")
            if not name:
                continue
            display = stat.get("displayValue")
            value = _float(stat.get("value"))
            if value is None:
                value = _float(display)
            stats[str(name)] = (value, display)
        lines.append(
            TeamStatLine(
                team_name=(block.get("team") or {}).get("displayName", ""),
                home_away=block.get("homeAway") or "",
                stats=stats,
            )
        )
    return lines


def parse_players(payload: dict[str, Any]) -> list[PlayerLine]:
    players: list[PlayerLine] = []
    for block in payload.get("rosters") or []:
        team_name = (block.get("team") or {}).get("displayName", "")
        home_away = block.get("homeAway") or ""
        for entry in block.get("roster") or []:
            athlete = entry.get("athlete") or {}
            name = athlete.get("displayName")
            if not name:
                continue
            position = entry.get("position") or {}
            position_name = str(position.get("displayName") or position.get("name")
                                or position.get("abbreviation") or "").strip()
            stats: dict[str, tuple[float | None, str | None]] = {}
            for stat in entry.get("stats") or []:
                stat_name = stat.get("name")
                if not stat_name:
                    continue
                stats[str(stat_name)] = (_float(stat.get("value")), stat.get("displayValue"))
            players.append(
                PlayerLine(
                    player_name=name,
                    external_player_id=str(athlete.get("id")) if athlete.get("id") else None,
                    team_name=team_name,
                    home_away=home_away,
                    jersey=entry.get("jersey"),
                    position=position.get("abbreviation") or position_name or None,
                    position_group=position_group_for(position_name,
                                                      position.get("abbreviation")),
                    formation_place=(str(entry["formationPlace"])
                                     if entry.get("formationPlace") is not None else None),
                    starter=bool(entry.get("starter")),
                    subbed_in=bool(entry.get("subbedIn")),
                    subbed_out=bool(entry.get("subbedOut")),
                    stats=stats,
                )
            )
    return players


def parse_standings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """League table rows, flattened.

    ESPN nests the table under groups whose shape varies by competition, so
    anything that does not look like a standings entry is skipped rather than
    guessed at.
    """
    rows: list[dict[str, Any]] = []
    groups = (payload.get("standings") or {}).get("groups") or []
    for group in groups:
        header = group.get("header")
        for entry in (group.get("standings") or {}).get("entries") or []:
            # ESPN sends ``team`` as a plain name string on the summary
            # endpoint and as an object on the standings endpoint. Handle both
            # rather than assuming whichever one I happened to look at first.
            raw_team = entry.get("team")
            if isinstance(raw_team, dict):
                name = raw_team.get("displayName") or raw_team.get("shortDisplayName")
                external_id = raw_team.get("id")
            else:
                name = raw_team
                external_id = entry.get("id")
            if not name:
                continue
            values: dict[str, Any] = {}
            for stat in entry.get("stats") or []:
                key = stat.get("name") or stat.get("abbreviation")
                if not key:
                    continue
                value = stat.get("value")
                values[str(key)] = value if value is not None else stat.get("displayValue")
            rows.append({
                "team_name": name,
                "external_id": str(external_id) if external_id else None,
                "group": header,
                **values,
            })
    return rows


def load_fixture(path: str) -> dict[str, Any]:
    """Read a stored payload — used by the source contract tests."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
