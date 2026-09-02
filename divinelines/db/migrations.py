"""Versioned schema migrations.

V2 created its schema with a single idempotent ``schema.sql`` script. That works
until a constraint needs to change — SQLite cannot alter one in place — so V3
adds a proper migration ledger. Each migration runs once, in order, inside a
transaction, and records itself.

Migrations must be safe to run against a populated database. The V3 odds
rebuild moves ~583k rows, so it is written as a table rebuild rather than a
drop-and-recreate: losing historical price snapshots would destroy the only
market history the platform has.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..logging_setup import get_logger
from .connection import init_db, write_connection

log = get_logger(__name__)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def _add_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if not _has_column(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


# ---------------------------------------------------------------------------
# 003 — market phases
# ---------------------------------------------------------------------------

def _migration_003(conn: sqlite3.Connection) -> None:
    """Give price snapshots an explicit market phase.

    ESPN publishes an opening and a closing price for a historical game with no
    capture timestamp for either. The V2 unique key was
    ``(game_uid, market, selection, bookmaker, captured_at)``, so an open and a
    close carrying the same nominal timestamp would collide and one would be
    silently dropped. Phase becomes part of the identity of a price.
    """
    if _has_column(conn, "odds_snapshots", "phase"):
        return

    conn.execute("ALTER TABLE odds_snapshots RENAME TO odds_snapshots_v2")
    conn.execute(
        """
        CREATE TABLE odds_snapshots (
            snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            game_uid      TEXT REFERENCES games(game_uid) ON DELETE CASCADE,
            sport         TEXT NOT NULL,
            market        TEXT NOT NULL,
            selection     TEXT NOT NULL,
            bookmaker     TEXT NOT NULL,
            price_decimal REAL NOT NULL,
            captured_at   TEXT NOT NULL,
            book_updated  TEXT,
            is_closing    INTEGER DEFAULT 0,
            -- 'snapshot' = observed by us at captured_at (timestamp is real);
            -- 'open'/'close' = phase published by the source without a capture
            -- time, so captured_at is nominal and nothing keys off it.
            phase         TEXT NOT NULL DEFAULT 'snapshot',
            source        TEXT NOT NULL,
            UNIQUE (game_uid, market, selection, bookmaker, captured_at, phase)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO odds_snapshots
            (snapshot_id, game_uid, sport, market, selection, bookmaker,
             price_decimal, captured_at, book_updated, is_closing, phase, source)
        SELECT snapshot_id, game_uid, sport, market, selection, bookmaker,
               price_decimal, captured_at, book_updated, is_closing,
               CASE WHEN is_closing = 1 THEN 'close' ELSE 'snapshot' END,
               source
        FROM odds_snapshots_v2
        """
    )
    conn.execute("DROP TABLE odds_snapshots_v2")
    conn.execute("CREATE INDEX idx_odds_game ON odds_snapshots(game_uid, market, captured_at)")
    conn.execute("CREATE INDEX idx_odds_captured ON odds_snapshots(captured_at)")
    conn.execute("CREATE INDEX idx_odds_phase ON odds_snapshots(game_uid, phase)")


# ---------------------------------------------------------------------------
# 004 — prediction versioning
# ---------------------------------------------------------------------------

def _migration_004(conn: sqlite3.Connection) -> None:
    """Predictions become versioned observations rather than one row per game.

    A prediction made before a lineup is published and one made after are
    different claims about the world. Both must survive, so a new prediction
    supersedes rather than overwrites, and every row records what was knowable
    when it was made.
    """
    _add_column(conn, "predictions", "prediction_stage", "TEXT NOT NULL DEFAULT 'scheduled'")
    _add_column(conn, "predictions", "lineup_state", "TEXT NOT NULL DEFAULT 'unknown'")
    _add_column(conn, "predictions", "supersedes_id", "INTEGER")
    _add_column(conn, "predictions", "superseded_at", "TEXT")
    _add_column(conn, "predictions", "feature_version", "TEXT")
    _add_column(conn, "predictions", "event_start_utc", "TEXT")
    _add_column(conn, "predictions", "seconds_to_event", "INTEGER")
    _add_column(conn, "predictions", "information_snapshot", "TEXT")
    _add_column(conn, "predictions", "settlement_state", "TEXT NOT NULL DEFAULT 'PENDING'")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_stage ON predictions(game_uid, prediction_stage)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_settlement ON predictions(settlement_state)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_active ON predictions(game_uid, superseded_at)"
    )


# ---------------------------------------------------------------------------
# 005 — CLV ledger
# ---------------------------------------------------------------------------

def _migration_005(conn: sqlite3.Connection) -> None:
    """CLV as a first-class dataset rather than a column on ``bets``.

    A CLV record exists for every *executable* recommendation, whether or not a
    paper bet was opened, because the question "were our prices better than the
    market" is separate from "did we stake it".
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clv_records (
            clv_id                INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id         INTEGER NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
            game_uid              TEXT NOT NULL REFERENCES games(game_uid) ON DELETE CASCADE,
            sport                 TEXT NOT NULL,
            league_id             TEXT,
            market                TEXT NOT NULL,
            selection             TEXT NOT NULL,

            entry_timestamp       TEXT NOT NULL,
            entry_odds            REAL NOT NULL,
            entry_book            TEXT,
            entry_market_prob     REAL,          -- no-vig consensus at entry
            entry_implied_prob    REAL,          -- gross implied at the taken price
            model_probability     REAL NOT NULL,

            closing_timestamp     TEXT,
            closing_odds          REAL,
            closing_book          TEXT,
            closing_implied_prob  REAL,
            closing_novig_prob    REAL,
            closing_source        TEXT,
            closing_policy        TEXT,          -- how the close was selected

            clv_price_pct         REAL,          -- closing/entry - 1, in percent
            clv_prob_points       REAL,          -- no-vig probability points gained
            clv_log_odds          REAL,
            beat_close            INTEGER,

            model_version         TEXT,
            model_id              TEXT,
            data_version          TEXT,
            prediction_stage      TEXT,
            lineup_state          TEXT,
            seconds_to_event      INTEGER,
            data_quality          REAL,
            edge                  REAL,

            status                TEXT NOT NULL DEFAULT 'PENDING',
            result                TEXT,          -- won|lost|push|void
            stake                 REAL,
            profit                REAL,
            settled_at            TEXT,
            created_at            TEXT NOT NULL,
            updated_at            TEXT,
            UNIQUE (prediction_id)
        )
        """
    )
    for statement in (
        "CREATE INDEX IF NOT EXISTS idx_clv_status ON clv_records(status)",
        "CREATE INDEX IF NOT EXISTS idx_clv_game ON clv_records(game_uid)",
        "CREATE INDEX IF NOT EXISTS idx_clv_sport ON clv_records(sport, entry_timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_clv_model ON clv_records(model_version)",
    ):
        conn.execute(statement)


# ---------------------------------------------------------------------------
# 006 — lineups and information events
# ---------------------------------------------------------------------------

def _migration_006(conn: sqlite3.Connection) -> None:
    """Timestamped availability observations.

    ``observed_at`` is when the platform saw this state; it is the only
    timestamp a backtest may filter on. Sources that publish lineups do not
    say when the lineup became public, so pretending otherwise would let a
    confirmed XI leak backwards into an earlier prediction.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lineup_observations (
            observation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            game_uid        TEXT NOT NULL REFERENCES games(game_uid) ON DELETE CASCADE,
            team_uid        TEXT NOT NULL REFERENCES teams(team_uid),
            sport           TEXT NOT NULL,
            player_uid      TEXT,
            player_name     TEXT NOT NULL,
            external_player_id TEXT,
            status          TEXT NOT NULL,   -- starter|bench|out|questionable|suspended|unused
            role            TEXT,            -- position abbreviation
            position_group  TEXT,            -- goalkeeper|defender|midfielder|forward|guard|...
            formation_place TEXT,
            formation       TEXT,
            lineup_state    TEXT NOT NULL,   -- projected|confirmed|final
            observed_at     TEXT NOT NULL,   -- when WE observed it
            source_timestamp TEXT,           -- when the source says it was true, if it says
            retrieved_at    TEXT NOT NULL,
            source          TEXT NOT NULL,
            UNIQUE (game_uid, team_uid, player_name, lineup_state, observed_at)
        )
        """
    )
    for statement in (
        "CREATE INDEX IF NOT EXISTS idx_lineup_game ON lineup_observations(game_uid, observed_at)",
        "CREATE INDEX IF NOT EXISTS idx_lineup_team ON lineup_observations(team_uid, observed_at)",
        "CREATE INDEX IF NOT EXISTS idx_lineup_state ON lineup_observations(game_uid, lineup_state)",
    ):
        conn.execute(statement)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS information_events (
            event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            game_uid       TEXT REFERENCES games(game_uid) ON DELETE CASCADE,
            team_uid       TEXT,
            sport          TEXT NOT NULL,
            kind           TEXT NOT NULL,   -- LINEUP_CONFIRMED|PLAYER_OUT|PLAYER_IN|ODDS_MOVE|...
            detail         TEXT,
            magnitude      REAL,            -- size of the change, where meaningful
            observed_at    TEXT NOT NULL,
            retrieved_at   TEXT NOT NULL,
            source         TEXT NOT NULL,
            UNIQUE (game_uid, kind, detail, observed_at)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_infoevent_game ON information_events(game_uid, observed_at)"
    )

    _add_column(conn, "games", "espn_event_id", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_espn ON games(espn_event_id)")


# ---------------------------------------------------------------------------
# 007 — model health and lifecycle
# ---------------------------------------------------------------------------

def _migration_007(conn: sqlite3.Connection) -> None:
    """Persisted health snapshots plus an explicit champion/candidate lifecycle.

    The prediction ledger stays the source of truth — every figure here is
    recomputable from it — but storing snapshots lets the dashboard show a
    trend without re-scoring the whole ledger on each request.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_health_snapshots (
            snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at     TEXT NOT NULL,
            sport           TEXT NOT NULL,
            league_id       TEXT,
            market          TEXT,
            model_version   TEXT,
            window_label    TEXT NOT NULL,   -- all_time|last_30d|last_100|season
            window_start    TEXT,
            window_end      TEXT,
            sample_size     INTEGER NOT NULL,
            brier           REAL,
            log_loss        REAL,
            accuracy        REAL,
            calibration_error REAL,
            market_brier    REAL,
            market_log_loss REAL,
            skill_vs_market REAL,            -- market log loss - model log loss
            clv_mean        REAL,
            clv_median      REAL,
            clv_positive_rate REAL,
            clv_sample      INTEGER,
            clv_ci_low      REAL,
            clv_ci_high     REAL,
            roi             REAL,
            profit          REAL,
            settled_bets    INTEGER,
            max_drawdown    REAL,
            prediction_volatility REAL,
            status          TEXT NOT NULL,
            status_reason   TEXT,
            metrics_json    TEXT,
            UNIQUE (computed_at, sport, market, model_version, window_label)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_health_lookup "
        "ON model_health_snapshots(sport, window_label, computed_at)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_lifecycle (
            lifecycle_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id      TEXT NOT NULL REFERENCES models(model_id),
            sport         TEXT NOT NULL,
            role          TEXT NOT NULL,     -- champion|candidate|retired
            promoted_at   TEXT NOT NULL,
            demoted_at    TEXT,
            reason        TEXT,
            evidence      TEXT,
            UNIQUE (model_id, role, promoted_at)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lifecycle_active "
        "ON model_lifecycle(sport, role, demoted_at)"
    )

    _add_column(conn, "source_status", "last_failure", "TEXT")
    _add_column(conn, "source_status", "failure_count", "INTEGER DEFAULT 0")
    _add_column(conn, "source_status", "success_count", "INTEGER DEFAULT 0")
    _add_column(conn, "source_status", "stale_count", "INTEGER DEFAULT 0")


def _migration_008(conn: sqlite3.Connection) -> None:
    """Separate model CLV from line-shopping CLV.

    Our entry price is the best available across books; the policy close is a
    consensus. Comparing the two makes CLV look positive on every single bet,
    because the best price is above the consensus by construction — I hit
    exactly that artefact on the first settlement run. The same-book columns
    answer "did the price move toward us at the book we actually used", which
    is the question about the model rather than about shopping.
    """
    _add_column(conn, "clv_records", "clv_same_book_pct", "REAL")
    _add_column(conn, "clv_records", "closing_same_book_odds", "REAL")
    _add_column(conn, "clv_records", "clv_basis", "TEXT")


# ---------------------------------------------------------------------------
# 009 — soccer match detail: events, box statistics, match context
# ---------------------------------------------------------------------------

def _migration_009(conn: sqlite3.Connection) -> None:
    """The observable record of what actually happened inside a match.

    Three deliberate choices here.

    ``match_events`` keeps ESPN's raw coordinates untouched in
    ``source_x``/``source_y``. Every normalisation into a pitch frame happens
    once, in Python, on the way out — if I stored a transformed number I would
    have no way to re-derive it after discovering the transform was wrong.

    ``observed_at`` is when the platform saw the event, and ``wallclock_utc``
    is when the source says it happened. Replay filters on the *event's* own
    clock; leakage checks filter on ``observed_at``. Conflating them is how a
    72nd-minute goal ends up visible at minute 32.

    Team and player statistics are stored long rather than wide because the
    feed's stat list is not stable: ESPN adds and removes metrics between
    competitions, and a wide table would need a migration every time.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_events (
            event_row_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            game_uid        TEXT NOT NULL REFERENCES games(game_uid) ON DELETE CASCADE,
            external_id     TEXT,             -- the source's play id
            sequence        INTEGER NOT NULL, -- order within the match
            event_type      TEXT NOT NULL,    -- normalised taxonomy (goal|shot_on_target|...)
            source_type     TEXT,             -- the source's own label, kept verbatim
            period          INTEGER,
            clock_seconds   REAL,             -- seconds elapsed within the period
            clock_display   TEXT,             -- "45+2'"
            minute          REAL,             -- match minute, added time folded in
            wallclock_utc   TEXT,             -- when the source says it happened
            team_uid        TEXT REFERENCES teams(team_uid),
            home_away       TEXT,
            player_uid      TEXT,
            player_name     TEXT,
            assist_player_uid  TEXT,
            assist_player_name TEXT,
            scoring_play    INTEGER DEFAULT 0,
            home_score      INTEGER,
            away_score      INTEGER,
            source_x        REAL,             -- raw source coordinates, untransformed
            source_y        REAL,
            source_x2       REAL,
            source_y2       REAL,
            text            TEXT,
            short_text      TEXT,
            observed_at     TEXT NOT NULL,    -- when WE saw it
            retrieved_at    TEXT NOT NULL,
            source          TEXT NOT NULL,
            UNIQUE (game_uid, source, external_id, event_type, sequence)
        )
        """
    )
    for statement in (
        "CREATE INDEX IF NOT EXISTS idx_mevent_game ON match_events(game_uid, sequence)",
        "CREATE INDEX IF NOT EXISTS idx_mevent_clock ON match_events(game_uid, period, clock_seconds)",
        "CREATE INDEX IF NOT EXISTS idx_mevent_type ON match_events(game_uid, event_type)",
        "CREATE INDEX IF NOT EXISTS idx_mevent_observed ON match_events(game_uid, observed_at)",
        "CREATE INDEX IF NOT EXISTS idx_mevent_player ON match_events(player_uid)",
    ):
        conn.execute(statement)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_team_stats (
            game_uid      TEXT NOT NULL REFERENCES games(game_uid) ON DELETE CASCADE,
            team_uid      TEXT NOT NULL,
            home_away     TEXT,
            stat_name     TEXT NOT NULL,
            stat_value    REAL,
            display_value TEXT,
            observed_at   TEXT NOT NULL,
            retrieved_at  TEXT NOT NULL,
            source        TEXT NOT NULL,
            PRIMARY KEY (game_uid, team_uid, stat_name)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_player_stats (
            game_uid      TEXT NOT NULL REFERENCES games(game_uid) ON DELETE CASCADE,
            team_uid      TEXT NOT NULL,
            player_uid    TEXT NOT NULL,
            player_name   TEXT NOT NULL,
            stat_name     TEXT NOT NULL,
            stat_value    REAL,
            display_value TEXT,
            observed_at   TEXT NOT NULL,
            retrieved_at  TEXT NOT NULL,
            source        TEXT NOT NULL,
            PRIMARY KEY (game_uid, player_uid, stat_name)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mpstats_player ON match_player_stats(player_uid, stat_name)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_context (
            game_uid        TEXT PRIMARY KEY REFERENCES games(game_uid) ON DELETE CASCADE,
            status_state    TEXT,    -- pre | in | post, as the source reports it
            status_name     TEXT,    -- STATUS_FULL_TIME, STATUS_HALFTIME, ...
            status_detail   TEXT,
            period          INTEGER,
            clock_display   TEXT,
            venue           TEXT,
            venue_city      TEXT,
            venue_country   TEXT,
            attendance      INTEGER,
            officials       TEXT,    -- JSON list
            home_formation  TEXT,
            away_formation  TEXT,
            home_color      TEXT,
            away_color      TEXT,
            home_logo       TEXT,
            away_logo       TEXT,
            home_form       TEXT,
            away_form       TEXT,
            observed_at     TEXT NOT NULL,
            retrieved_at    TEXT NOT NULL,
            source          TEXT NOT NULL
        )
        """
    )

    # Lineup rows predate canonical player identity; give them somewhere to
    # put it so event participants and roster entries resolve to one player.
    _add_column(conn, "lineup_observations", "jersey", "TEXT")
    _add_column(conn, "lineup_observations", "subbed_in", "INTEGER")
    _add_column(conn, "lineup_observations", "subbed_out", "INTEGER")


MIGRATIONS: tuple[Migration, ...] = (
    Migration(3, "odds_market_phase", _migration_003),
    Migration(4, "prediction_versioning", _migration_004),
    Migration(5, "clv_ledger", _migration_005),
    Migration(6, "lineups_and_information_events", _migration_006),
    Migration(7, "model_health_and_lifecycle", _migration_007),
    Migration(8, "same_book_clv", _migration_008),
    Migration(9, "soccer_match_detail", _migration_009),
)


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def migrate(db_path: Path | str | None = None) -> list[str]:
    """Apply pending migrations. Safe to call on every startup."""
    init_db(db_path)
    applied: list[str] = []

    with write_connection(db_path) as conn:
        # Foreign keys must be off while a table is rebuilt, otherwise the
        # rename step trips references from other tables mid-migration.
        conn.execute("PRAGMA foreign_keys=OFF")
        done = applied_versions(conn)
        for migration in MIGRATIONS:
            if migration.version in done:
                continue
            log.info("applying migration",
                     extra={"version": migration.version, "name": migration.name})
            migration.apply(conn)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) "
                "VALUES (?, ?, datetime('now'))",
                (migration.version, migration.name),
            )
            applied.append(f"{migration.version:03d}_{migration.name}")
        conn.execute("PRAGMA foreign_keys=ON")

    if applied:
        log.info("migrations applied", extra={"count": len(applied)})
    return applied


def schema_version(db_path: Path | str | None = None) -> int:
    with write_connection(db_path) as conn:
        done = applied_versions(conn)
    return max(done) if done else 2
