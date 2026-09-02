"""Lineup and availability observations from ESPN.

Soccer first, because a confirmed XI moves a soccer price far more than an NBA
inactive list moves a basketball one — a missing goalkeeper or centre-forward
changes the goal expectation directly.

## The timestamp problem, stated plainly

ESPN publishes *what* the lineup is. It does not publish *when* that lineup
became public. So every observation is stamped with ``observed_at`` — the
moment this platform saw it — and that is the only timestamp anything is
allowed to filter on.

The consequence is deliberate and unavoidable: for a match played before this
platform started polling, the actual XI is retrievable but the publication time
is not. Such observations are recorded as ``lineup_state='final'`` and are
**barred from live prediction** — they exist for research, where they support
an explicit upper-bound experiment ("would knowing the XI have helped at all?")
rather than pretending to be information we had at kick-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from ..config import FRESHNESS_TTL
from ..logging_setup import get_logger
from .base import HttpSource, SourceError
from .espn_odds import SPORT_PATHS

log = get_logger(__name__)

SITE_API = "https://site.api.espn.com/apis/site/v2/sports"

#: ESPN position names -> the platform's coarse position groups. The impact
#: framework weights a goalkeeper differently from a substitute winger, so the
#: group matters more than the exact label.
POSITION_GROUPS: dict[str, str] = {
    "goalkeeper": "goalkeeper", "g": "goalkeeper",
    "defender": "defender", "d": "defender",
    "center back": "defender", "fullback": "defender",
    "midfielder": "midfielder", "m": "midfielder",
    "forward": "forward", "f": "forward", "striker": "forward",
    "guard": "guard", "center": "center", "point guard": "guard",
    "shooting guard": "guard", "small forward": "forward",
    "power forward": "forward",
}

#: ESPN's soccer summary reports the *formation slot* rather than a position
#: name — "CD-L", "AM-R", "LB". Mapping the slot prefix is what actually
#: classifies a starting XI; the display names above only cover substitutes and
#: the NBA feed. Without this, ~85% of starters landed in no position group at
#: all, which would have quietly disabled goalkeeper weighting.
SLOT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("G", "goalkeeper"),
    ("CD", "defender"), ("LB", "defender"), ("RB", "defender"),
    ("LWB", "defender"), ("RWB", "defender"), ("D", "defender"),
    ("DM", "midfielder"), ("CM", "midfielder"), ("AM", "midfielder"),
    ("LM", "midfielder"), ("RM", "midfielder"), ("M", "midfielder"),
    ("CF", "forward"), ("LF", "forward"), ("RF", "forward"),
    ("LW", "forward"), ("RW", "forward"), ("ST", "forward"), ("F", "forward"),
)


def position_group_for(position_name: str | None, slot: str | None) -> str | None:
    """Resolve a position group from ESPN's display name or formation slot."""
    if position_name:
        mapped = POSITION_GROUPS.get(position_name.strip().lower())
        if mapped:
            return mapped
    if slot:
        # Slots look like "CD-L" or "AM-R"; the side suffix is irrelevant here.
        base = slot.strip().upper().split("-")[0]
        for prefix, group in SLOT_PREFIXES:
            if base == prefix:
                return group
        for prefix, group in SLOT_PREFIXES:
            if base.startswith(prefix):
                return group
    return None

#: A lineup we saw before the event is live information; one we saw after it
#: started is a historical record of what happened.
STATE_PROJECTED = "projected"
STATE_CONFIRMED = "confirmed"
STATE_FINAL = "final"


@dataclass
class LineupEntry:
    player_name: str
    external_player_id: str | None
    status: str                 # starter | bench | unused | out
    role: str | None
    position_group: str | None
    formation_place: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name, "status": self.status,
            "role": self.role, "position_group": self.position_group,
        }


@dataclass
class TeamLineup:
    team_name: str
    home_away: str
    formation: str | None
    entries: list[LineupEntry] = field(default_factory=list)

    @property
    def starters(self) -> list[LineupEntry]:
        return [e for e in self.entries if e.status == "starter"]

    @property
    def has_goalkeeper(self) -> bool:
        return any(e.position_group == "goalkeeper" for e in self.starters)


@dataclass
class LineupObservation:
    espn_event_id: str
    sport: str
    league_id: str
    teams: list[TeamLineup]
    lineup_state: str
    observed_at: datetime
    retrieved_at: datetime
    from_cache: bool
    source: str = "espn_lineups"

    @property
    def is_usable_live(self) -> bool:
        """Only pre-event observations may inform a live prediction."""
        return self.lineup_state in (STATE_PROJECTED, STATE_CONFIRMED)


class EspnLineupSource(HttpSource):
    name = "espn_lineups"
    #: Lineups matter most in the hour before kick-off, so they expire fast.
    cache_ttl = FRESHNESS_TTL["lineups"]
    user_agent = (
        "Mozilla/5.0 (compatible; DivineLines/3.0; research; "
        "+https://github.com/ImShankk/DivineLines)"
    )
    min_interval = 0.5

    def fetch_lineup(self, league_id: str, espn_event_id: str, *,
                     event_started: bool = False, force: bool = False) -> LineupObservation:
        if league_id not in SPORT_PATHS:
            raise SourceError(f"espn_lineups: no ESPN path for '{league_id}'")
        group, league = SPORT_PATHS[league_id]

        result = self.fetch_json(
            f"{SITE_API}/{group}/{league}/summary",
            dataset=f"lineup:{league}:{espn_event_id}",
            status_dataset=f"lineups:{league}",
            params={"event": espn_event_id},
            # A finished match's lineup never changes, so it can be cached hard;
            # an upcoming one must be re-checked constantly.
            ttl=FRESHNESS_TTL["reference"] if event_started else self.cache_ttl,
            force=force,
        )
        payload = result.data if isinstance(result.data, dict) else {}
        rosters = payload.get("rosters") or []
        if not rosters:
            raise SourceError(f"espn_lineups: no rosters published for event {espn_event_id}")

        teams: list[TeamLineup] = []
        for block in rosters:
            entries: list[LineupEntry] = []
            for player in block.get("roster") or []:
                athlete = player.get("athlete") or {}
                name = athlete.get("displayName")
                if not name:
                    continue
                position = (player.get("position") or {})
                position_name = str(position.get("displayName")
                                    or position.get("name")
                                    or position.get("abbreviation") or "").strip()
                entries.append(
                    LineupEntry(
                        player_name=name,
                        external_player_id=str(athlete.get("id")) if athlete.get("id") else None,
                        status="starter" if player.get("starter") else (
                            "bench" if player.get("subbedIn") is not None else "unused"
                        ),
                        role=position.get("abbreviation") or position_name or None,
                        position_group=position_group_for(
                            position_name, position.get("abbreviation")
                        ),
                        formation_place=str(player.get("formationPlace"))
                        if player.get("formationPlace") is not None else None,
                    )
                )
            teams.append(
                TeamLineup(
                    team_name=(block.get("team") or {}).get("displayName", ""),
                    home_away=block.get("homeAway", ""),
                    formation=block.get("formation"),
                    entries=entries,
                )
            )

        # State follows from whether the event has begun, never from a guess
        # about how "official" the data looks.
        if event_started:
            state = STATE_FINAL
        else:
            state = (STATE_CONFIRMED
                     if all(len(team.starters) >= 5 for team in teams)
                     else STATE_PROJECTED)

        observed = datetime.now(timezone.utc) if not result.from_cache else result.retrieved_at
        return LineupObservation(
            espn_event_id=str(espn_event_id), sport="soccer" if group == "soccer" else "nba",
            league_id=league_id, teams=teams, lineup_state=state,
            observed_at=observed, retrieved_at=result.retrieved_at,
            from_cache=result.from_cache,
        )
