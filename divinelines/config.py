"""Central configuration for DivineLines.

Every tunable value in the platform lives here (or in the environment) rather
than being scattered through modules.  Import ``settings`` and read attributes;
never hard-code a threshold, a path, a season or a Kelly fraction elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Paths:
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    reference_dir: Path = REPO_ROOT / "data" / "reference"
    artifacts_dir: Path = REPO_ROOT / "data" / "artifacts"
    logs_dir: Path = REPO_ROOT / "data" / "logs"
    #: Canonical, normalised platform database (created by ``migrate``).
    db_path: Path = REPO_ROOT / "data" / "divinelines.db"
    #: The original v1 database.  Read-only; kept so nothing is destroyed.
    legacy_nba_db: Path = REPO_ROOT / "data" / "processed" / "nba_data.db"

    def ensure(self) -> None:
        for directory in (
            self.data_dir,
            self.cache_dir,
            self.reference_dir,
            self.artifacts_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BettingConfig:
    """Risk and staking policy.  All of it is configurable — nothing hard-coded."""

    bankroll: float = _env_float("DL_BANKROLL", 1000.0)
    kelly_fraction: float = _env_float("DL_KELLY_FRACTION", 0.25)
    #: Never stake more than this share of bankroll on a single selection.
    max_stake_pct: float = _env_float("DL_MAX_STAKE_PCT", 0.02)
    #: Total stake across a single slate/day.
    max_slate_exposure_pct: float = _env_float("DL_MAX_SLATE_EXPOSURE_PCT", 0.10)
    #: Total stake on selections from one game (correlation control).
    max_game_exposure_pct: float = _env_float("DL_MAX_GAME_EXPOSURE_PCT", 0.03)
    #: Total stake on selections involving one team.
    max_team_exposure_pct: float = _env_float("DL_MAX_TEAM_EXPOSURE_PCT", 0.04)
    #: Minimum modelled edge (model prob - no-vig market prob) to recommend.
    min_edge: float = _env_float("DL_MIN_EDGE", 0.02)
    #: Minimum edge-quality score (0-10) to recommend.
    min_edge_score: float = _env_float("DL_MIN_EDGE_SCORE", 4.0)
    #: Reject prices outside this decimal-odds band (liquidity / longshot bias).
    min_decimal_odds: float = _env_float("DL_MIN_DECIMAL_ODDS", 1.20)
    max_decimal_odds: float = _env_float("DL_MAX_DECIMAL_ODDS", 15.0)
    #: Flag a model as an outlier when it disagrees with the market by more.
    model_outlier_threshold: float = _env_float("DL_MODEL_OUTLIER_THRESHOLD", 0.25)


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = _env_int("DL_RANDOM_SEED", 42)
    #: Fraction of the chronological history held out as the final test block.
    test_fraction: float = _env_float("DL_TEST_FRACTION", 0.20)
    #: Games at the start of a season before current-season form is trusted
    #: alone; used by the Bayesian shrinkage blender.
    shrinkage_prior_games: float = _env_float("DL_SHRINKAGE_PRIOR_GAMES", 12.0)
    #: Weight retained from the previous season's rating at a season rollover.
    elo_season_regression: float = _env_float("DL_ELO_SEASON_REGRESSION", 0.75)
    elo_k_nba: float = _env_float("DL_ELO_K_NBA", 20.0)
    elo_k_soccer: float = _env_float("DL_ELO_K_SOCCER", 20.0)
    calibration_method: str = _env_str("DL_CALIBRATION", "isotonic")  # isotonic|platt|none


@dataclass(frozen=True)
class SourceConfig:
    odds_api_key: str | None = os.getenv("ODDS_API_KEY") or None
    odds_regions: str = _env_str("DL_ODDS_REGIONS", "us,uk,eu")
    odds_format: str = "decimal"
    http_timeout: float = _env_float("DL_HTTP_TIMEOUT", 20.0)
    http_retries: int = _env_int("DL_HTTP_RETRIES", 3)
    http_backoff: float = _env_float("DL_HTTP_BACKOFF", 1.5)
    #: Polite delay between successive requests to the same host.
    http_min_interval: float = _env_float("DL_HTTP_MIN_INTERVAL", 0.75)
    user_agent: str = _env_str(
        "DL_USER_AGENT",
        "DivineLines/2.0 (research; contact via repository)",
    )
    #: Guard rail so a runaway loop cannot burn the odds-API quota.
    odds_api_max_calls_per_run: int = _env_int("DL_ODDS_MAX_CALLS", 8)


#: Time-to-live, in seconds, after which a dataset is considered stale.
#: Consumed by :mod:`divinelines.data.freshness`.
FRESHNESS_TTL: dict[str, int] = {
    "odds": _env_int("DL_TTL_ODDS", 10 * 60),
    "injuries": _env_int("DL_TTL_INJURIES", 6 * 60 * 60),
    "lineups": _env_int("DL_TTL_LINEUPS", 45 * 60),
    "schedule": _env_int("DL_TTL_SCHEDULE", 12 * 60 * 60),
    "box_scores": _env_int("DL_TTL_BOXSCORES", 12 * 60 * 60),
    "rosters": _env_int("DL_TTL_ROSTERS", 24 * 60 * 60),
    "transactions": _env_int("DL_TTL_TRANSACTIONS", 24 * 60 * 60),
    "standings": _env_int("DL_TTL_STANDINGS", 12 * 60 * 60),
    "soccer_results": _env_int("DL_TTL_SOCCER_RESULTS", 24 * 60 * 60),
    "reference": _env_int("DL_TTL_REFERENCE", 30 * 24 * 60 * 60),
}


#: Soccer competitions the platform can ingest.  ``fd_code`` is the
#: football-data.co.uk division code.  ``strength`` is a coarse league-quality
#: multiplier used only for cross-league priors on promoted/new clubs; it is a
#: prior, not a fitted parameter, and is documented as such.
SOCCER_LEAGUES: dict[str, dict[str, Any]] = {
    "ENG_PL": {"name": "Premier League", "country": "England", "fd_code": "E0", "tier": 1, "strength": 1.00},
    "ENG_CH": {"name": "Championship", "country": "England", "fd_code": "E1", "tier": 2, "strength": 0.80},
    "ESP_LL": {"name": "La Liga", "country": "Spain", "fd_code": "SP1", "tier": 1, "strength": 0.97},
    "ITA_SA": {"name": "Serie A", "country": "Italy", "fd_code": "I1", "tier": 1, "strength": 0.96},
    "GER_BL": {"name": "Bundesliga", "country": "Germany", "fd_code": "D1", "tier": 1, "strength": 0.95},
    "FRA_L1": {"name": "Ligue 1", "country": "France", "fd_code": "F1", "tier": 1, "strength": 0.92},
    "NED_ED": {"name": "Eredivisie", "country": "Netherlands", "fd_code": "N1", "tier": 1, "strength": 0.86},
    "POR_PL": {"name": "Primeira Liga", "country": "Portugal", "fd_code": "P1", "tier": 1, "strength": 0.86},
}

def _default_soccer_seasons(history: int = 9) -> list[str]:
    """Season codes ending with the current one.

    Derived from the clock rather than hard-coded, so a new season needs no
    edit: on 1 July the list rolls forward on its own.
    """
    current_start = (datetime.now(timezone.utc).year
                     if datetime.now(timezone.utc).month >= 7
                     else datetime.now(timezone.utc).year - 1)
    return [
        f"{str(year)[-2:]}{str(year + 1)[-2:]}"
        for year in range(current_start - history + 1, current_start + 1)
    ]


#: Seasons to ingest for soccer, as football-data.co.uk season codes.
SOCCER_SEASONS: list[str] = (
    _env_str("DL_SOCCER_SEASONS", ",".join(_default_soccer_seasons())).split(",")
)


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    betting: BettingConfig = field(default_factory=BettingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)
    #: ``live`` uses current data and writes predictions to the ledger;
    #: ``research`` allows experiments and never touches live prediction state.
    mode: str = _env_str("DL_MODE", "research")
    log_level: str = _env_str("DL_LOG_LEVEL", "INFO")
    log_json: bool = _env_bool("DL_LOG_JSON", False)
    offline: bool = _env_bool("DL_OFFLINE", False)

    def with_mode(self, mode: str) -> "Settings":
        return replace(self, mode=mode)


settings = Settings()
settings.paths.ensure()


# --------------------------------------------------------------------------
# Season helpers — the platform must never need a hard-coded SEASON constant.
# --------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def nba_season_for_date(when: date | datetime | None = None) -> str:
    """Return the NBA season label (e.g. ``2025-26``) containing ``when``.

    The NBA season rolls over on 1 August: anything from August onward belongs
    to the season that starts that calendar year.
    """
    when = when or utcnow()
    if isinstance(when, datetime):
        when = when.date()
    start_year = when.year if when.month >= 8 else when.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def nba_season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def soccer_season_for_date(when: date | datetime | None = None) -> str:
    """Return the football-data season code (e.g. ``2526``) containing ``when``.

    European seasons roll over on 1 July.
    """
    when = when or utcnow()
    if isinstance(when, datetime):
        when = when.date()
    start_year = when.year if when.month >= 7 else when.year - 1
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def previous_nba_season(season: str) -> str:
    start = nba_season_start_year(season) - 1
    return f"{start}-{str(start + 1)[-2:]}"
