"""Canonical entity identity.

Inconsistent naming is the most dangerous failure mode in sports data:
``LA Lakers`` / ``Los Angeles Lakers`` / ``L.A. Lakers`` must resolve to one
entity, and ``Man City`` (football-data) must resolve to the same club as
``Manchester City`` (The-Odds-API).

This module owns *all* name resolution.  Nothing else in the platform should
compare team names with string equality.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

_PUNCT = re.compile(r"[^a-z0-9]+")

_NOISE_WORDS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "us", "ud", "cd", "rc", "sv",
    "vfl", "vfb", "tsg", "fsv", "bsc", "if", "ff", "sk", "aik", "the",
}


def normalize_key(text: str) -> str:
    """Aggressively normalise a name into a comparison key."""
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = _PUNCT.sub(" ", folded.lower()).strip()
    tokens = [t for t in folded.split() if t not in _NOISE_WORDS]
    return " ".join(tokens) if tokens else folded


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NBATeam:
    team_id: int          # official NBA team id (stable across renames)
    abbr: str
    full_name: str
    city: str
    tz: str               # IANA-ish offset bucket used for travel features
    lat: float
    lon: float


#: Static reference data (official team ids, arena coordinates).  Coordinates
#: are the home arena location and are used only for travel-distance features.
NBA_TEAMS: tuple[NBATeam, ...] = (
    NBATeam(1610612737, "ATL", "Atlanta Hawks", "Atlanta", "ET", 33.7573, -84.3963),
    NBATeam(1610612738, "BOS", "Boston Celtics", "Boston", "ET", 42.3662, -71.0621),
    NBATeam(1610612751, "BKN", "Brooklyn Nets", "Brooklyn", "ET", 40.6826, -73.9754),
    NBATeam(1610612766, "CHA", "Charlotte Hornets", "Charlotte", "ET", 35.2251, -80.8392),
    NBATeam(1610612741, "CHI", "Chicago Bulls", "Chicago", "CT", 41.8807, -87.6742),
    NBATeam(1610612739, "CLE", "Cleveland Cavaliers", "Cleveland", "ET", 41.4965, -81.6882),
    NBATeam(1610612742, "DAL", "Dallas Mavericks", "Dallas", "CT", 32.7905, -96.8104),
    NBATeam(1610612743, "DEN", "Denver Nuggets", "Denver", "MT", 39.7487, -105.0077),
    NBATeam(1610612765, "DET", "Detroit Pistons", "Detroit", "ET", 42.3410, -83.0552),
    NBATeam(1610612744, "GSW", "Golden State Warriors", "San Francisco", "PT", 37.7680, -122.3877),
    NBATeam(1610612745, "HOU", "Houston Rockets", "Houston", "CT", 29.7508, -95.3621),
    NBATeam(1610612754, "IND", "Indiana Pacers", "Indianapolis", "ET", 39.7640, -86.1555),
    NBATeam(1610612746, "LAC", "LA Clippers", "Los Angeles", "PT", 33.9450, -118.3410),
    NBATeam(1610612747, "LAL", "Los Angeles Lakers", "Los Angeles", "PT", 34.0430, -118.2673),
    NBATeam(1610612763, "MEM", "Memphis Grizzlies", "Memphis", "CT", 35.1382, -90.0505),
    NBATeam(1610612748, "MIA", "Miami Heat", "Miami", "ET", 25.7814, -80.1870),
    NBATeam(1610612749, "MIL", "Milwaukee Bucks", "Milwaukee", "CT", 43.0451, -87.9172),
    NBATeam(1610612750, "MIN", "Minnesota Timberwolves", "Minneapolis", "CT", 44.9795, -93.2760),
    NBATeam(1610612740, "NOP", "New Orleans Pelicans", "New Orleans", "CT", 29.9490, -90.0821),
    NBATeam(1610612752, "NYK", "New York Knicks", "New York", "ET", 40.7505, -73.9934),
    NBATeam(1610612760, "OKC", "Oklahoma City Thunder", "Oklahoma City", "CT", 35.4634, -97.5151),
    NBATeam(1610612753, "ORL", "Orlando Magic", "Orlando", "ET", 28.5392, -81.3839),
    NBATeam(1610612755, "PHI", "Philadelphia 76ers", "Philadelphia", "ET", 39.9012, -75.1720),
    NBATeam(1610612756, "PHX", "Phoenix Suns", "Phoenix", "MT", 33.4457, -112.0712),
    NBATeam(1610612757, "POR", "Portland Trail Blazers", "Portland", "PT", 45.5316, -122.6668),
    NBATeam(1610612758, "SAC", "Sacramento Kings", "Sacramento", "PT", 38.5802, -121.4997),
    NBATeam(1610612759, "SAS", "San Antonio Spurs", "San Antonio", "CT", 29.4271, -98.4375),
    NBATeam(1610612761, "TOR", "Toronto Raptors", "Toronto", "ET", 43.6435, -79.3791),
    NBATeam(1610612762, "UTA", "Utah Jazz", "Salt Lake City", "MT", 40.7683, -111.9011),
    NBATeam(1610612764, "WAS", "Washington Wizards", "Washington", "ET", 38.8981, -77.0209),
)

NBA_BY_ABBR: dict[str, NBATeam] = {t.abbr: t for t in NBA_TEAMS}
NBA_BY_ID: dict[int, NBATeam] = {t.team_id: t for t in NBA_TEAMS}

#: Historical/alternate abbreviations and nicknames that must not create a
#: second entity.  Franchise relocations map onto the current franchise id.
_NBA_EXTRA_ALIASES: dict[str, str] = {
    "nor": "NOP", "noh": "NOP", "nok": "NOP", "new orleans hornets": "NOP",
    "njn": "BKN", "new jersey nets": "BKN",
    "sea": "OKC", "seattle supersonics": "OKC",
    "van": "MEM", "vancouver grizzlies": "MEM",
    "cha bobcats": "CHA", "charlotte bobcats": "CHA",
    "gs": "GSW", "sa": "SAS", "ny": "NYK", "no": "NOP", "phx suns": "PHX",
    "pho": "PHX", "brk": "BKN", "cho": "CHA", "utah": "UTA",
    "la lakers": "LAL", "l a lakers": "LAL", "los angeles lakers": "LAL",
    "la clippers": "LAC", "l a clippers": "LAC", "los angeles clippers": "LAC",
    "clippers": "LAC", "lakers": "LAL",
    "golden state": "GSW", "portland": "POR", "oklahoma city": "OKC",
}


@lru_cache(maxsize=4096)
def resolve_nba_team(value: str | int | None) -> str | None:
    """Resolve any NBA team spelling/id to its canonical abbreviation."""
    if value is None:
        return None
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit() and len(value) > 5):
        team = NBA_BY_ID.get(int(value))
        return team.abbr if team else None

    raw = str(value).strip()
    if raw.upper() in NBA_BY_ABBR:
        return raw.upper()

    key = normalize_key(raw)
    if not key:
        return None
    for team in NBA_TEAMS:
        if key == normalize_key(team.full_name):
            return team.abbr
    if key in _NBA_EXTRA_ALIASES:
        return _NBA_EXTRA_ALIASES[key]
    # Nickname match ("Timberwolves", "Trail Blazers")
    for team in NBA_TEAMS:
        nickname = normalize_key(team.full_name.replace(team.city, "").strip())
        if nickname and key == nickname:
            return team.abbr
    # City match, only when unambiguous
    city_hits = [t for t in NBA_TEAMS if normalize_key(t.city) == key]
    if len(city_hits) == 1:
        return city_hits[0].abbr
    return None


def nba_team(abbr: str) -> NBATeam:
    return NBA_BY_ABBR[abbr]


# ---------------------------------------------------------------------------
# Soccer
# ---------------------------------------------------------------------------

#: Maps the many spellings seen across sources onto one canonical club name.
#: Left-hand keys are ``normalize_key`` outputs.  Extend freely — resolution is
#: explicit rather than fuzzy so a wrong merge can never happen silently.
_SOCCER_ALIASES: dict[str, str] = {
    # England
    "man united": "Manchester United", "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "man city": "Manchester City",
    "newcastle": "Newcastle United", "newcastle utd": "Newcastle United",
    "nottm forest": "Nottingham Forest", "nott m forest": "Nottingham Forest",
    "notts forest": "Nottingham Forest",
    "wolves": "Wolverhampton Wanderers", "wolverhampton": "Wolverhampton Wanderers",
    "tottenham": "Tottenham Hotspur", "spurs": "Tottenham Hotspur",
    "west ham": "West Ham United", "west brom": "West Bromwich Albion",
    "brighton": "Brighton and Hove Albion", "brighton hove albion": "Brighton and Hove Albion",
    "leicester": "Leicester City", "leeds": "Leeds United",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "sheffield weds": "Sheffield Wednesday",
    "qpr": "Queens Park Rangers", "hull": "Hull City", "stoke": "Stoke City",
    "swansea": "Swansea City", "cardiff": "Cardiff City", "norwich": "Norwich City",
    "birmingham": "Birmingham City", "coventry": "Coventry City",
    "derby": "Derby County", "ipswich": "Ipswich Town", "luton": "Luton Town",
    "preston": "Preston North End", "bristol city": "Bristol City",
    "huddersfield": "Huddersfield Town", "blackburn": "Blackburn Rovers",
    "bolton": "Bolton Wanderers", "wigan": "Wigan Athletic",
    "afc bournemouth": "Bournemouth", "crystal palace": "Crystal Palace",
    # Spain
    "ath madrid": "Atletico Madrid", "atletico": "Atletico Madrid",
    "atl madrid": "Atletico Madrid", "atletico de madrid": "Atletico Madrid",
    "ath bilbao": "Athletic Club", "athletic bilbao": "Athletic Club",
    "athletic club bilbao": "Athletic Club",
    "espanol": "Espanyol", "sociedad": "Real Sociedad",
    "betis": "Real Betis", "real betis balompie": "Real Betis",
    "vallecano": "Rayo Vallecano", "celta": "Celta Vigo", "celta de vigo": "Celta Vigo",
    "la coruna": "Deportivo La Coruna", "alaves": "Deportivo Alaves",
    "cadiz": "Cadiz", "valladolid": "Real Valladolid",
    # Italy
    "inter": "Inter Milan", "internazionale": "Inter Milan", "inter milano": "Inter Milan",
    "milan": "AC Milan", "ac milan": "AC Milan",
    "roma": "AS Roma", "as roma": "AS Roma", "lazio": "Lazio",
    "juventus": "Juventus", "napoli": "Napoli", "hellas verona": "Verona",
    # Germany
    "bayern munich": "Bayern Munich", "bayern": "Bayern Munich",
    "dortmund": "Borussia Dortmund", "borussia dortmund": "Borussia Dortmund",
    "m gladbach": "Borussia Monchengladbach", "monchengladbach": "Borussia Monchengladbach",
    "borussia monchengladbach": "Borussia Monchengladbach",
    "leverkusen": "Bayer Leverkusen", "bayer 04 leverkusen": "Bayer Leverkusen",
    "ein frankfurt": "Eintracht Frankfurt", "eintracht frankfurt": "Eintracht Frankfurt",
    "rb leipzig": "RB Leipzig", "leipzig": "RB Leipzig",
    "hoffenheim": "Hoffenheim", "tsg hoffenheim": "Hoffenheim",
    "schalke 04": "Schalke 04", "werder bremen": "Werder Bremen",
    "fc koln": "FC Koln", "cologne": "FC Koln", "koln": "FC Koln",
    "union berlin": "Union Berlin", "st pauli": "St Pauli",
    "mainz": "Mainz 05", "mainz 05": "Mainz 05", "wolfsburg": "Wolfsburg",
    "freiburg": "Freiburg", "augsburg": "Augsburg", "stuttgart": "Stuttgart",
    # France
    "paris sg": "Paris Saint Germain", "psg": "Paris Saint Germain",
    "paris saint germain": "Paris Saint Germain",
    "marseille": "Marseille", "olympique marseille": "Marseille",
    "lyon": "Lyon", "olympique lyonnais": "Lyon", "monaco": "Monaco",
    "st etienne": "Saint Etienne", "saint etienne": "Saint Etienne",
    "paris fc": "Paris FC",
    # Netherlands / Portugal
    "ajax": "Ajax", "psv eindhoven": "PSV", "psv": "PSV",
    "feyenoord": "Feyenoord", "az alkmaar": "AZ Alkmaar", "az": "AZ Alkmaar",
    "sp lisbon": "Sporting CP", "sporting lisbon": "Sporting CP",
    "sporting cp": "Sporting CP", "porto": "Porto", "fc porto": "Porto",
    "benfica": "Benfica", "sl benfica": "Benfica",
    "sp braga": "Braga", "sporting braga": "Braga",
    "guimaraes": "Vitoria Guimaraes", "vit guimaraes": "Vitoria Guimaraes",
}


@lru_cache(maxsize=8192)
def canonical_club_name(value: str | None) -> str | None:
    """Resolve a club spelling to the platform's canonical club name."""
    if not value:
        return None
    key = normalize_key(value)
    if not key:
        return None
    if key in _SOCCER_ALIASES:
        return _SOCCER_ALIASES[key]
    # Title-case the normalised form as the canonical fallback.  Because the
    # key is deterministic, two spellings that normalise identically merge.
    return " ".join(word.capitalize() for word in key.split())


@lru_cache(maxsize=8192)
def club_id(value: str | None) -> str | None:
    """Stable, league-independent club identifier (clubs change division)."""
    canonical = canonical_club_name(value)
    if canonical is None:
        return None
    return normalize_key(canonical).replace(" ", "-")


def same_club(a: str | None, b: str | None) -> bool:
    left, right = club_id(a), club_id(b)
    return left is not None and left == right


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

#: A player key built from a name is fragile — "Kevin De Bruyne", "Kevin
#: DeBruyne" and "De Bruyne, Kevin" all have to collapse to one entity, and
#: two different players can legitimately share a name. So a source-supplied
#: athlete id always wins, and the name key is only the fallback.
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def player_name_key(value: str | None) -> str:
    """Comparison key for a player name.

    ``normalize_key`` strips accents and punctuation, which already merges
    "Gabriel Magalhães" with "Gabriel Magalhaes". On top of that I fold the
    "Surname, Firstname" form, because ESPN's roster and its commentary
    disagree about which one they use.
    """
    if not value:
        return ""
    text = str(value)
    if "," in text:
        surname, _, rest = text.partition(",")
        text = f"{rest.strip()} {surname.strip()}"
    key = normalize_key(text)
    tokens = [t for t in key.split() if t not in _NAME_SUFFIXES]
    return " ".join(tokens) if tokens else key


def soccer_player_uid(name: str | None, external_id: str | int | None = None) -> str | None:
    """Canonical player identifier.

    ``soccer:espn:274272`` when the source gives an athlete id, otherwise
    ``soccer:name:altay-bayindir``. Name-derived ids are explicitly marked so
    that a later merge can find them: they are a weaker claim to identity and
    the uid says so rather than hiding it.
    """
    if external_id not in (None, "", 0):
        return f"soccer:espn:{external_id}"
    key = player_name_key(name)
    if not key:
        return None
    return f"soccer:name:{key.replace(' ', '-')}"


def same_player(a: str | None, b: str | None) -> bool:
    left, right = player_name_key(a), player_name_key(b)
    return bool(left) and left == right
