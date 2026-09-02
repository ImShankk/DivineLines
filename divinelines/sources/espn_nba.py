"""ESPN adapter for NBA availability and schedule data.

ESPN's public site API is the only free source that publishes NBA injury
*status* (Out / Doubtful / Questionable / Day-To-Day) per player.  It reports
current state only — there is no history endpoint — which has a hard
consequence the platform makes explicit everywhere: injury features cannot be
backtested, so they are applied as a **post-model adjustment at prediction
time**, never trained into the historical model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable

from ..config import FRESHNESS_TTL
from ..identity import resolve_nba_team
from ..logging_setup import get_logger
from .base import HttpSource, SourceError

log = get_logger(__name__)

SITE_API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

#: ESPN status strings -> the platform's availability vocabulary, plus the
#: probability that the player suits up.  The probabilities are the standard
#: reading of NBA injury-report terminology (a "questionable" tag has
#: historically resolved to playing roughly half the time); they are priors,
#: exposed in config-like form here so they can be revised, not hidden magic.
STATUS_MAP: dict[str, tuple[str, float]] = {
    "out": ("out", 0.0),
    "injured reserve": ("out", 0.0),
    "suspension": ("suspended", 0.0),
    "suspended": ("suspended", 0.0),
    "doubtful": ("doubtful", 0.25),
    "questionable": ("questionable", 0.50),
    "day-to-day": ("questionable", 0.65),
    "game-time decision": ("questionable", 0.50),
    "probable": ("probable", 0.85),
    "active": ("available", 1.0),
    "available": ("available", 1.0),
}


@dataclass
class PlayerAvailability:
    player_name: str
    espn_athlete_id: str | None
    team_abbr: str | None
    position: str | None
    status: str
    play_probability: float
    detail: str | None
    expected_return: str | None
    as_of: str
    source: str = "espn"

    @property
    def is_uncertain(self) -> bool:
        return 0.0 < self.play_probability < 1.0


class EspnNbaSource(HttpSource):
    name = "espn_nba"
    cache_ttl = FRESHNESS_TTL["injuries"]
    #: ESPN's site API returns 403 to unfamiliar agents.  This identifies the
    #: client honestly while remaining acceptable to the host; requests stay
    #: rate-limited and cached so the endpoint is not hammered.
    user_agent = (
        "Mozilla/5.0 (compatible; DivineLines/2.0; research; "
        "+https://github.com/ImShankk/DivineLines)"
    )

    # ------------------------------------------------------------- injuries

    def fetch_injuries(self, *, force: bool = False) -> tuple[list[PlayerAvailability], datetime]:
        result = self.fetch_json(
            f"{SITE_API}/injuries", dataset="injuries",
            ttl=FRESHNESS_TTL["injuries"], force=force,
        )
        payload = result.data
        if not isinstance(payload, dict) or "injuries" not in payload:
            raise SourceError("espn: unexpected injuries payload")

        records: list[PlayerAvailability] = []
        for team_block in payload.get("injuries", []):
            team_abbr = resolve_nba_team(team_block.get("displayName"))
            for entry in team_block.get("injuries", []):
                athlete = entry.get("athlete") or {}
                raw_status = str(entry.get("status") or "").strip().lower()
                status, play_probability = STATUS_MAP.get(raw_status, ("unknown", 0.5))
                if status == "unknown":
                    log.warning("unmapped injury status", extra={"status": raw_status})
                details = entry.get("details") or {}
                detail_bits = [
                    details.get("type"), details.get("detail"), details.get("side"),
                ]
                records.append(
                    PlayerAvailability(
                        player_name=athlete.get("displayName") or "Unknown",
                        espn_athlete_id=str(athlete.get("id")) if athlete.get("id") else None,
                        team_abbr=team_abbr or resolve_nba_team(
                            (athlete.get("team") or {}).get("displayName")
                        ),
                        position=((athlete.get("position") or {}).get("abbreviation")),
                        status=status,
                        play_probability=play_probability,
                        detail=" ".join(b for b in detail_bits if b) or None,
                        expected_return=details.get("returnDate"),
                        as_of=entry.get("date") or result.retrieved_at.isoformat(),
                    )
                )
        log.info("fetched NBA injuries", extra={"players": len(records),
                                                "cached": result.from_cache})
        return records, result.retrieved_at

    # ------------------------------------------------------------- schedule

    def fetch_scoreboard(self, day: date | str, *, force: bool = False) -> list[dict[str, Any]]:
        """Games for one calendar day, scheduled or final."""
        stamp = day.strftime("%Y%m%d") if isinstance(day, date) else str(day).replace("-", "")
        result = self.fetch_json(
            f"{SITE_API}/scoreboard", dataset=f"scoreboard:{stamp}",
            params={"dates": stamp, "limit": 100},
            ttl=FRESHNESS_TTL["schedule"], force=force,
        )
        events = result.data.get("events", []) if isinstance(result.data, dict) else []
        return [g for g in (self._parse_event(e) for e in events) if g]

    def fetch_schedule_range(self, days: Iterable[date | str]) -> list[dict[str, Any]]:
        games: list[dict[str, Any]] = []
        for day in days:
            try:
                games.extend(self.fetch_scoreboard(day))
            except SourceError as exc:
                log.warning("scoreboard fetch failed", extra={"day": str(day), "error": str(exc)})
        return games

    @staticmethod
    def _parse_event(event: dict[str, Any]) -> dict[str, Any] | None:
        competitions = event.get("competitions") or []
        if not competitions:
            return None
        comp = competitions[0]
        home = away = None
        for competitor in comp.get("competitors", []):
            team_name = (competitor.get("team") or {}).get("displayName")
            abbr = resolve_nba_team(team_name) or resolve_nba_team(
                (competitor.get("team") or {}).get("abbreviation")
            )
            score = competitor.get("score")
            entry = {
                "abbr": abbr,
                "score": float(score) if score not in (None, "") else None,
            }
            if competitor.get("homeAway") == "home":
                home = entry
            else:
                away = entry
        if not home or not away or not home["abbr"] or not away["abbr"]:
            return None

        status_name = ((comp.get("status") or event.get("status") or {})
                       .get("type", {}).get("name", ""))
        final = status_name == "STATUS_FINAL"
        return {
            "espn_event_id": str(event.get("id")),
            "kickoff_utc": event.get("date"),
            "status": "final" if final else "scheduled",
            "home_abbr": home["abbr"],
            "away_abbr": away["abbr"],
            "home_score": home["score"] if final else None,
            "away_score": away["score"] if final else None,
            "neutral_site": 1 if comp.get("neutralSite") else 0,
            "venue": (comp.get("venue") or {}).get("fullName"),
        }

    # --------------------------------------------------------------- rosters

    def fetch_roster(self, team_abbr: str, *, force: bool = False) -> list[dict[str, Any]]:
        """Current roster for a team — used for roster-change detection."""
        result = self.fetch_json(
            f"{SITE_API}/teams/{team_abbr.lower()}/roster",
            dataset=f"roster:{team_abbr}", ttl=FRESHNESS_TTL["rosters"], force=force,
        )
        athletes = result.data.get("athletes", []) if isinstance(result.data, dict) else []
        players: list[dict[str, Any]] = []
        for athlete in athletes:
            entries = athlete.get("items", [athlete]) if "items" in athlete else [athlete]
            for item in entries:
                if not item.get("displayName"):
                    continue
                players.append(
                    {
                        "espn_athlete_id": str(item.get("id")),
                        "full_name": item.get("displayName"),
                        "position": ((item.get("position") or {}).get("abbreviation")),
                        "jersey": item.get("jersey"),
                        "age": item.get("age"),
                        "experience": ((item.get("experience") or {}).get("years")),
                        "team_abbr": team_abbr,
                    }
                )
        return players
