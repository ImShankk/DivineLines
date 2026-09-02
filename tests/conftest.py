"""Shared fixtures.

Every test runs against a temporary database seeded with deterministic
fixtures.  Nothing here touches the network, a live scoreboard or the odds
API — a test that depends on today's ESPN response is not a test.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from divinelines.config import settings  # noqa: E402
from divinelines.db import connection  # noqa: E402
from divinelines.db import migrations as db_migrations  # noqa: E402


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """A fresh, schema-initialised database for one test."""
    db_path = tmp_path / "test.db"
    # ``Paths`` is a frozen dataclass by design (config should not drift at
    # runtime), so the test overrides it explicitly and restores it after.
    originals = {
        "db_path": settings.paths.db_path,
        "artifacts_dir": settings.paths.artifacts_dir,
        "cache_dir": settings.paths.cache_dir,
    }
    artifacts = tmp_path / "artifacts"
    cache = tmp_path / "cache"
    artifacts.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    object.__setattr__(settings.paths, "db_path", db_path)
    object.__setattr__(settings.paths, "artifacts_dir", artifacts)
    object.__setattr__(settings.paths, "cache_dir", cache)
    original_connect = connection.connect

    from contextlib import contextmanager

    @contextmanager
    def patched(path=None, *, readonly=False):
        with original_connect(path or db_path, readonly=readonly) as conn:
            yield conn

    monkeypatch.setattr(connection, "connect", patched)
    for module_name in (
        "divinelines.db.repository",
        "divinelines.db.validation",
        "divinelines.betting.ledger",
        "divinelines.models.registry",
        "divinelines.data.freshness",
    ):
        module = sys.modules.get(module_name)
        if module and hasattr(module, "connect"):
            monkeypatch.setattr(module, "connect", patched, raising=False)

    connection.init_db(db_path)
    # Tests must run against the same schema as production, migrations included;
    # otherwise a V3 column missing in the test database looks like a code bug.
    monkeypatch.setattr(db_migrations, "write_connection", connection.write_connection,
                        raising=False)
    db_migrations.migrate(db_path)
    try:
        yield db_path
    finally:
        for name, value in originals.items():
            object.__setattr__(settings.paths, name, value)


@pytest.fixture()
def seeded_db(temp_db):
    """Two NBA teams, one soccer league and a handful of finished games."""
    from divinelines.db.repository import (
        ensure_leagues,
        ensure_nba_teams,
        ensure_soccer_teams,
        upsert_games,
        upsert_nba_box,
        upsert_odds,
    )

    ensure_leagues()
    ensure_nba_teams()
    ensure_soccer_teams(["Arsenal", "Chelsea"], "England")

    now = datetime.now(timezone.utc)
    games = []
    box = []
    for index in range(6):
        game_date = (now - timedelta(days=30 - index * 3)).date().isoformat()
        game_uid = f"nba:00225000{index:02d}"
        home, away = ("nba:BOS", "nba:LAL") if index % 2 == 0 else ("nba:LAL", "nba:BOS")
        home_score, away_score = (110 + index, 104 + index)
        games.append(
            {
                "game_uid": game_uid, "sport": "nba", "league_id": "NBA", "season": "2025-26",
                "game_date": game_date, "kickoff_utc": None, "status": "final",
                "home_team_uid": home, "away_team_uid": away,
                "home_score": home_score, "away_score": away_score,
                "neutral_site": 0, "venue": None,
                "source": "test", "retrieved_at": now.isoformat(),
            }
        )
        for team, opponent, is_home, points in (
            (home, away, 1, home_score), (away, home, 0, away_score)
        ):
            box.append(
                {
                    "game_uid": game_uid, "team_uid": team, "opp_uid": opponent,
                    "is_home": is_home, "won": int(points == max(home_score, away_score)),
                    "min": 240.0, "fgm": 40.0, "fga": 88.0, "fg3m": 12.0, "fg3a": 34.0,
                    "ftm": 18.0, "fta": 22.0, "oreb": 10.0, "dreb": 33.0, "reb": 43.0,
                    "ast": 25.0, "stl": 7.0, "blk": 5.0, "tov": 13.0, "pf": 19.0,
                    "pts": float(points), "plus_minus": float(home_score - away_score),
                    "source": "test", "retrieved_at": now.isoformat(),
                }
            )
    upsert_games(games)
    upsert_nba_box(box)

    upcoming = (now + timedelta(days=1)).date().isoformat()
    upsert_games([
        {
            "game_uid": "nba:upcoming", "sport": "nba", "league_id": "NBA", "season": "2025-26",
            "game_date": upcoming, "kickoff_utc": None, "status": "scheduled",
            "home_team_uid": "nba:BOS", "away_team_uid": "nba:LAL",
            "home_score": None, "away_score": None, "neutral_site": 0, "venue": None,
            "source": "test", "retrieved_at": now.isoformat(),
        }
    ])
    upsert_odds([
        {
            "game_uid": "nba:upcoming", "sport": "nba", "market": "h2h", "selection": selection,
            "bookmaker": book, "price_decimal": price,
            "captured_at": now.isoformat(timespec="seconds"), "book_updated": None,
            "is_closing": 0, "source": "test",
        }
        for book, prices in {"BookA": {"home": 1.80, "away": 2.10},
                             "BookB": {"home": 1.85, "away": 2.05}}.items()
        for selection, price in prices.items()
    ])
    return temp_db


@pytest.fixture()
def nba_team_games():
    """Deterministic team-game frame for feature-builder tests."""
    import numpy as np

    rng = np.random.default_rng(11)
    rows = []
    base = pd.Timestamp("2024-10-20")
    for index in range(24):
        game_date = base + pd.Timedelta(days=index * 2)
        season = "2024-25" if index < 16 else "2025-26"
        game_uid = f"nba:g{index:03d}"
        home, away = ("nba:BOS", "nba:LAL") if index % 2 == 0 else ("nba:LAL", "nba:BOS")
        # Mixed outcomes: a fixture where the home side always wins makes the
        # target constant and silently disables any test that correlates
        # against it.
        home_points = 104 + (index % 7) * 2
        away_points = 103 + ((index * 3) % 11) * 2
        for team, opponent, is_home, points in (
            (home, away, 1, home_points), (away, home, 0, away_points)
        ):
            # Vary the box score per team-game: identical rows would make every
            # differential feature constant and untestable.
            rows.append(
                {
                    "game_uid": game_uid, "game_date": game_date, "season": season,
                    "status": "final", "home_score": home_points, "away_score": away_points,
                    "neutral_site": 0, "team_uid": team, "opp_uid": opponent,
                    "is_home": is_home, "won": int(points == max(home_points, away_points)),
                    "min": 240.0,
                    "fgm": float(rng.integers(36, 46)), "fga": float(rng.integers(82, 95)),
                    "fg3m": float(rng.integers(8, 18)), "fg3a": float(rng.integers(28, 42)),
                    "ftm": float(rng.integers(12, 24)), "fta": float(rng.integers(16, 30)),
                    "oreb": float(rng.integers(6, 14)), "dreb": float(rng.integers(28, 38)),
                    "reb": float(rng.integers(38, 50)), "ast": float(rng.integers(20, 30)),
                    "stl": float(rng.integers(4, 11)), "blk": float(rng.integers(2, 8)),
                    "tov": float(rng.integers(9, 18)), "pf": float(rng.integers(14, 24)),
                    "pts": float(points), "plus_minus": float(home_points - away_points),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def soccer_matches():
    """Synthetic two-season league with known strengths and a real fixture list.

    A proper double round-robin matters: a rotation where each club always
    meets the same opponent leaves attack, defence and home advantage only
    weakly identifiable, which would make the model look broken when it is not.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    teams = [f"soccer:club{i}" for i in range(8)]
    strength = {team: 0.30 * (len(teams) / 2 - index) / len(teams)
                for index, team in enumerate(teams)}

    rows = []
    date = pd.Timestamp("2024-08-10")
    game_index = 0
    for season in ("2425", "2526"):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lambda_home = np.exp(0.25 + strength[home] - strength[away] + 0.25)
                lambda_away = np.exp(0.25 + strength[away] - strength[home])
                rows.append(
                    {
                        "game_uid": f"soccer:m{game_index:04d}",
                        "league_id": "ENG_PL", "season": season,
                        "game_date": date, "kickoff_utc": None, "status": "final",
                        "home_team_uid": home, "away_team_uid": away,
                        "home_score": float(rng.poisson(lambda_home)),
                        "away_score": float(rng.poisson(lambda_away)),
                        "home_name": home, "away_name": away,
                        "home_shots": 12, "away_shots": 10, "home_sot": 5, "away_sot": 4,
                        "home_corners": 5, "away_corners": 4, "home_fouls": 11,
                        "away_fouls": 12, "home_yellow": 2, "away_yellow": 2,
                        "home_red": 0, "away_red": 0, "referee": "Test Ref",
                    }
                )
                game_index += 1
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture()
def dixon_coles_ground_truth():
    """A league generated from known parameters, for recovery testing."""
    import numpy as np

    rng = np.random.default_rng(5)
    teams = [f"t{i}" for i in range(12)]
    truth = {
        "attack": {team: float(rng.normal(0, 0.25)) for team in teams},
        "defence": {team: float(rng.normal(0, 0.25)) for team in teams},
        "home_advantage": 0.30,
        "base": 0.20,
    }

    rows = []
    date = pd.Timestamp("2024-08-01")
    for _ in range(20):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lambda_home = np.exp(truth["base"] + truth["attack"][home]
                                     - truth["defence"][away] + truth["home_advantage"])
                lambda_away = np.exp(truth["base"] + truth["attack"][away]
                                     - truth["defence"][home])
                rows.append({
                    "home_team_uid": home, "away_team_uid": away, "league_id": "L",
                    "home_score": float(rng.poisson(lambda_home)),
                    "away_score": float(rng.poisson(lambda_away)),
                    "game_date": date, "season": "2425",
                })
                date += pd.Timedelta(hours=6)
    return pd.DataFrame(rows), truth


# ---------------------------------------------------------------------------
# Soccer match centre
# ---------------------------------------------------------------------------

SOCCER_MATCH_UID = "soccer:ENG_PL:2026-05-24:arsenal-vs-chelsea"
KICKOFF = "2026-05-24T15:00:00+00:00"
#: Every backfilled match in the platform is read after full time, so the
#: fixture stamps one ingest time on the whole stream — which is the truth,
#: and is what the leakage tests need to be able to assert against.
INGESTED_AT = "2026-08-17T09:00:00+00:00"


@pytest.fixture()
def soccer_match_with_events(temp_db):
    """One finished soccer match with events, box scores, lineups and prices.

    Deliberately hand-built rather than loaded from the ESPN fixture: the
    replay tests need a known goal at a known minute, and a real payload would
    make every assertion depend on what happened in one Premier League match.
    """
    from divinelines.db.connection import upsert_rows
    from divinelines.db.repository import (
        ensure_leagues,
        ensure_soccer_teams,
        upsert_games,
        upsert_odds,
    )

    ensure_leagues()
    ensure_soccer_teams(["Arsenal", "Chelsea"], "England")
    home, away = "soccer:arsenal", "soccer:chelsea"

    upsert_games([{
        "game_uid": SOCCER_MATCH_UID, "sport": "soccer", "league_id": "ENG_PL",
        "season": "2526", "game_date": "2026-05-24", "kickoff_utc": KICKOFF,
        "status": "final", "home_team_uid": home, "away_team_uid": away,
        "home_score": 0, "away_score": 1, "neutral_site": 0, "venue": "Emirates Stadium",
        "source": "test", "retrieved_at": INGESTED_AT,
    }])
    # An earlier fixture, so the standings panel has something to compute from.
    upsert_games([{
        "game_uid": "soccer:ENG_PL:2026-05-01:chelsea-vs-arsenal", "sport": "soccer",
        "league_id": "ENG_PL", "season": "2526", "game_date": "2026-05-01",
        "kickoff_utc": "2026-05-01T15:00:00+00:00", "status": "final",
        "home_team_uid": away, "away_team_uid": home, "home_score": 2, "away_score": 1,
        "neutral_site": 0, "venue": "Stamford Bridge",
        "source": "test", "retrieved_at": INGESTED_AT,
    }])

    events = [
        (0, "kickoff", None, 0.0, None, None, 0, 0),
        (1, "shot_off_target", "home", 8.0, 0.30, 0.55, 0, 0),
        (2, "foul", "away", 12.0, 0.66, 0.40, 0, 0),
        (3, "corner", "home", 18.0, 0.09, 0.95, 0, 0),
        (4, "shot_on_target", "home", 25.0, 0.20, 0.50, 0, 0),
        (5, "yellow_card", "away", 31.0, None, None, 0, 0),
        (6, "shot_blocked", "away", 40.0, 0.28, 0.44, 0, 0),
        (7, "halftime", None, 45.0, None, None, 0, 0),
        (8, "second_half_start", None, 45.5, None, None, 0, 0),
        (9, "goal", "away", 58.0, 0.04, 0.52, 0, 1),
        (10, "substitution", "home", 63.0, None, None, 0, 1),
        (11, "shot_on_target", "home", 77.0, 0.15, 0.48, 0, 1),
        (12, "full_time", None, 92.0, None, None, 0, 1),
    ]
    upsert_rows("match_events", [
        {
            "game_uid": SOCCER_MATCH_UID, "external_id": "e%d" % sequence,
            "sequence": sequence, "event_type": event_type, "source_type": event_type,
            "period": 1 if minute <= 45 else 2, "clock_seconds": minute * 60,
            "clock_display": "%d'" % int(minute), "minute": minute,
            "wallclock_utc": None,
            "team_uid": (home if side == "home" else away) if side else None,
            "home_away": side,
            "player_uid": "soccer:espn:1" if side else None,
            "player_name": "Test Player" if side else None,
            "assist_player_uid": None, "assist_player_name": None,
            "scoring_play": int(event_type == "goal"),
            "home_score": home_score, "away_score": away_score,
            "source_x": x, "source_y": y, "source_x2": None, "source_y2": None,
            "text": "%s at %s" % (event_type, minute), "short_text": event_type,
            "observed_at": INGESTED_AT, "retrieved_at": INGESTED_AT, "source": "espn_match",
        }
        for sequence, event_type, side, minute, x, y, home_score, away_score in events
    ])

    upsert_rows("match_team_stats", [
        {
            "game_uid": SOCCER_MATCH_UID, "team_uid": team_uid, "home_away": side,
            "stat_name": name, "stat_value": value, "display_value": str(value),
            "observed_at": INGESTED_AT, "retrieved_at": INGESTED_AT, "source": "espn_match",
        }
        for team_uid, side, values in (
            (home, "home", {"possessionPct": 58.2, "totalShots": 14.0,
                            "shotsOnTarget": 5.0, "totalPasses": 512.0, "passPct": 0.86}),
            (away, "away", {"possessionPct": 41.8, "totalShots": 9.0,
                            "shotsOnTarget": 3.0, "totalPasses": 388.0, "passPct": 0.79}),
        )
        for name, value in values.items()
    ])

    upsert_rows("match_context", [{
        "game_uid": SOCCER_MATCH_UID, "status_state": "post",
        "status_name": "STATUS_FULL_TIME", "status_detail": "FT", "period": 2,
        "clock_display": None, "venue": "Emirates Stadium", "venue_city": "London",
        "venue_country": "England", "attendance": 60214,
        "officials": '["Test Official"]',
        "home_formation": "4-3-3", "away_formation": "3-4-3",
        "home_color": "ff0000", "away_color": "0000ff",
        "home_logo": None, "away_logo": None, "home_form": "WWDLW", "away_form": "LDWWL",
        "observed_at": INGESTED_AT, "retrieved_at": INGESTED_AT, "source": "espn_match",
    }])

    upsert_rows("lineup_observations", [
        {
            "game_uid": SOCCER_MATCH_UID, "team_uid": team_uid, "sport": "soccer",
            "player_uid": "soccer:espn:%s%d" % (team_uid[-3:], index),
            "player_name": "%s %d" % (label, index), "external_player_id": None,
            "status": "starter" if index <= 11 else "bench",
            "role": "G" if index == 1 else "M",
            "position_group": "goalkeeper" if index == 1 else "midfielder",
            "formation_place": str(index), "formation": formation,
            "lineup_state": "final",
            "observed_at": INGESTED_AT, "source_timestamp": None,
            "retrieved_at": INGESTED_AT, "source": "espn_lineups",
            "jersey": str(index), "subbed_in": 0, "subbed_out": 0,
        }
        for team_uid, label, formation in ((home, "Home Player", "4-3-3"),
                                           (away, "Away Player", "3-4-3"))
        for index in range(1, 14)
    ])

    upsert_odds([
        {
            "game_uid": SOCCER_MATCH_UID, "sport": "soccer", "market": "1x2",
            "selection": selection, "bookmaker": book, "price_decimal": price,
            "captured_at": captured, "book_updated": None,
            "is_closing": int(phase == "close"), "phase": phase, "source": "test",
        }
        for book in ("BookA", "BookB")
        for captured, phase, prices in (
            ("2026-05-23T12:00:00+00:00", "open",
             {"home": 2.10, "draw": 3.40, "away": 3.60}),
            ("2026-05-24T14:45:00+00:00", "close",
             {"home": 2.05, "draw": 3.45, "away": 3.70}),
        )
        for selection, price in prices.items()
    ])
    return SOCCER_MATCH_UID


@pytest.fixture()
def insert_event(soccer_match_with_events):
    """Push an extra event into a match — used by the leakage tests."""
    from divinelines.db.connection import upsert_rows

    def _insert(*, minute, event_type, side, sequence, observed_at=INGESTED_AT):
        upsert_rows("match_events", [{
            "game_uid": soccer_match_with_events,
            "external_id": "injected%d" % sequence,
            "sequence": sequence, "event_type": event_type, "source_type": event_type,
            "period": 2, "clock_seconds": minute * 60,
            "clock_display": "%d'" % int(minute), "minute": minute, "wallclock_utc": None,
            "team_uid": "soccer:arsenal" if side == "home" else "soccer:chelsea",
            "home_away": side, "player_uid": None, "player_name": "Injected",
            "assist_player_uid": None, "assist_player_name": None,
            "scoring_play": int(event_type == "goal"),
            "home_score": 1 if (event_type == "goal" and side == "home") else 0,
            "away_score": 1,
            "source_x": 0.06, "source_y": 0.5, "source_x2": None, "source_y2": None,
            "text": "injected", "short_text": "injected",
            "observed_at": observed_at, "retrieved_at": observed_at,
            "source": "espn_match",
        }])

    return _insert


@pytest.fixture()
def insert_price(soccer_match_with_events):
    from divinelines.db.repository import upsert_odds

    def _insert(*, captured_at, price):
        upsert_odds([{
            "game_uid": soccer_match_with_events, "sport": "soccer", "market": "1x2",
            "selection": selection, "bookmaker": "BookLate", "price_decimal": price,
            "captured_at": captured_at, "book_updated": None, "is_closing": 0,
            "phase": "snapshot", "source": "test",
        } for selection in ("home", "draw", "away")])

    return _insert


@pytest.fixture()
def insert_prediction(soccer_match_with_events):
    from divinelines.db.connection import upsert_rows

    def _insert(*, created_at, probability):
        upsert_rows("predictions", [{
            "created_at": created_at, "sport": "soccer", "league_id": "ENG_PL",
            "game_uid": soccer_match_with_events, "market": "1x2", "selection": "home",
            "model_prob": probability, "market_prob": 0.48, "price_decimal": 2.10,
            "bookmaker": "BookA", "edge": 0.02, "ev_per_unit": 0.03,
            "kelly_fraction": 0.01, "stake": 0.0, "confidence": 0.5, "edge_score": 4.0,
            "data_quality": 80.0, "model_id": None, "model_version": "test-1",
            "data_version": None, "features": None, "explanation": None, "flags": None,
            "mode": "paper",
        }])

    return _insert


@pytest.fixture()
def insert_lineup_observation(soccer_match_with_events):
    from divinelines.db.connection import upsert_rows

    def _insert(*, observed_at, player_name):
        upsert_rows("lineup_observations", [{
            "game_uid": soccer_match_with_events, "team_uid": "soccer:arsenal",
            "sport": "soccer", "player_uid": None, "player_name": player_name,
            "external_player_id": None, "status": "starter", "role": "F",
            "position_group": "forward", "formation_place": "9", "formation": "4-3-3",
            "lineup_state": "final", "observed_at": observed_at, "source_timestamp": None,
            "retrieved_at": observed_at, "source": "espn_lineups",
            "jersey": "9", "subbed_in": 0, "subbed_out": 0,
        }])

    return _insert
