"""Typed persistence helpers.

All writes into the canonical schema flow through here so that identity
resolution and provenance stamping happen in exactly one place.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from ..config import SOCCER_LEAGUES, settings
from ..identity import NBA_TEAMS, club_id, canonical_club_name, nba_team, resolve_nba_team
from ..logging_setup import get_logger
from .connection import connect, query_df, upsert_rows, write_connection

log = get_logger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Identity helpers
# --------------------------------------------------------------------------

def nba_team_uid(value: str | int | None) -> str | None:
    abbr = resolve_nba_team(value)
    return f"nba:{abbr}" if abbr else None


def soccer_team_uid(value: str | None) -> str | None:
    cid = club_id(value)
    return f"soccer:{cid}" if cid else None


def game_uid_nba(nba_game_id: str) -> str:
    return f"nba:{str(nba_game_id).zfill(10)}"


def game_uid_soccer(league_id: str, date_iso: str, home_uid: str, away_uid: str) -> str:
    home = home_uid.split(":", 1)[-1]
    away = away_uid.split(":", 1)[-1]
    return f"soccer:{league_id}:{date_iso}:{home}-vs-{away}"


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

def ensure_leagues() -> None:
    rows: list[dict[str, Any]] = [
        {
            "league_id": "NBA", "sport": "nba", "name": "National Basketball Association",
            "country": "USA", "tier": 1, "strength": 1.0,
        }
    ]
    for league_id, meta in SOCCER_LEAGUES.items():
        rows.append(
            {
                "league_id": league_id, "sport": "soccer", "name": meta["name"],
                "country": meta["country"], "tier": meta["tier"], "strength": meta["strength"],
            }
        )
    upsert_rows("leagues", rows, conflict_columns=["league_id"])


def ensure_nba_teams() -> None:
    timestamp = now_iso()
    rows = [
        {
            "team_uid": f"nba:{t.abbr}", "sport": "nba", "canonical_name": t.full_name,
            "abbr": t.abbr, "country": "USA" if t.abbr != "TOR" else "Canada",
            "external_ids": json.dumps({"nba_team_id": t.team_id}),
            "lat": t.lat, "lon": t.lon, "tz": t.tz,
            "first_seen": timestamp, "last_seen": timestamp,
        }
        for t in NBA_TEAMS
    ]
    upsert_rows("teams", rows, conflict_columns=["team_uid"])


def ensure_soccer_teams(names: Iterable[str], country: str | None = None) -> dict[str, str]:
    """Register soccer clubs, returning ``{original_name: team_uid}``."""
    timestamp = now_iso()
    mapping: dict[str, str] = {}
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        uid = soccer_team_uid(name)
        if not uid:
            continue
        mapping[name] = uid
        rows[uid] = {
            "team_uid": uid, "sport": "soccer",
            "canonical_name": canonical_club_name(name), "abbr": None,
            "country": country, "external_ids": json.dumps({}),
            "lat": None, "lon": None, "tz": None,
            "first_seen": timestamp, "last_seen": timestamp,
        }
    if rows:
        # Preserve first_seen for clubs already known.
        existing = {
            r["team_uid"]
            for r in query_df(
                "SELECT team_uid FROM teams WHERE sport='soccer'"
            ).to_dict("records")
        }
        fresh = [r for uid, r in rows.items() if uid not in existing]
        updates = [
            {"team_uid": uid, "last_seen": timestamp, "canonical_name": r["canonical_name"]}
            for uid, r in rows.items()
            if uid in existing
        ]
        if fresh:
            upsert_rows("teams", fresh, conflict_columns=["team_uid"])
        if updates:
            with write_connection() as conn:
                conn.executemany(
                    "UPDATE teams SET last_seen = ?, canonical_name = ? WHERE team_uid = ?",
                    [(u["last_seen"], u["canonical_name"], u["team_uid"]) for u in updates],
                )
    return mapping


# --------------------------------------------------------------------------
# Fact tables
# --------------------------------------------------------------------------

def upsert_games(rows: Sequence[dict[str, Any]]) -> int:
    return upsert_rows("games", list(rows), conflict_columns=["game_uid"])


def upsert_nba_box(rows: Sequence[dict[str, Any]]) -> int:
    return upsert_rows("nba_team_game", list(rows), conflict_columns=["game_uid", "team_uid"])


def upsert_soccer_stats(rows: Sequence[dict[str, Any]]) -> int:
    return upsert_rows("soccer_match_stats", list(rows), conflict_columns=["game_uid"])


def upsert_odds(rows: Sequence[dict[str, Any]]) -> int:
    return upsert_rows(
        "odds_snapshots", list(rows),
        conflict_columns=["game_uid", "market", "selection", "bookmaker", "captured_at"],
        update=False,
    )


def record_source_status(
    source: str, dataset: str, *, status: str, rows: int | None = None,
    latency_ms: int | None = None, message: str | None = None, success: bool | None = None,
) -> None:
    """Track every fetch attempt so staleness and outages are observable."""
    timestamp = now_iso()
    succeeded = success if success is not None else (status == "ok")
    with write_connection() as conn:
        existing = conn.execute(
            "SELECT last_success FROM source_status WHERE source=? AND dataset=?",
            (source, dataset),
        ).fetchone()
        last_success = timestamp if succeeded else (existing["last_success"] if existing else None)
        conn.execute(
            """
            INSERT INTO source_status (source, dataset, last_attempt, last_success, status, rows, latency_ms, message)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(source, dataset) DO UPDATE SET
                last_attempt=excluded.last_attempt, last_success=excluded.last_success,
                status=excluded.status, rows=excluded.rows,
                latency_ms=excluded.latency_ms, message=excluded.message
            """,
            (source, dataset, timestamp, last_success, status, rows, latency_ms,
             (message or "")[:500] or None),
        )


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def load_games(sport: str, *, status: str | None = None, league_id: str | None = None,
               since: str | None = None, until: str | None = None) -> pd.DataFrame:
    clauses = ["g.sport = ?"]
    params: list[Any] = [sport]
    if status:
        clauses.append("g.status = ?")
        params.append(status)
    if league_id:
        clauses.append("g.league_id = ?")
        params.append(league_id)
    if since:
        clauses.append("g.game_date >= ?")
        params.append(since)
    if until:
        clauses.append("g.game_date <= ?")
        params.append(until)
    sql = f"""
        SELECT g.*, th.canonical_name AS home_name, ta.canonical_name AS away_name
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        WHERE {' AND '.join(clauses)}
        ORDER BY g.game_date, g.game_uid
    """
    return query_df(sql, params)


def load_nba_team_games(since: str | None = None) -> pd.DataFrame:
    """Team-game box scores joined with game context, chronologically ordered."""
    params: list[Any] = []
    where = ""
    if since:
        where = "WHERE g.game_date >= ?"
        params.append(since)
    sql = f"""
        SELECT g.game_uid, g.game_date, g.season, g.status,
               g.home_score, g.away_score, g.neutral_site,
               t.team_uid, t.opp_uid, t.is_home, t.won,
               t.min, t.fgm, t.fga, t.fg3m, t.fg3a, t.ftm, t.fta,
               t.oreb, t.dreb, t.reb, t.ast, t.stl, t.blk, t.tov, t.pf,
               t.pts, t.plus_minus
        FROM nba_team_game t
        JOIN games g ON g.game_uid = t.game_uid
        {where}
        ORDER BY g.game_date, g.game_uid, t.is_home DESC
    """
    df = query_df(sql, params)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_soccer_matches(league_ids: Sequence[str] | None = None,
                        status: str = "final") -> pd.DataFrame:
    clauses = ["g.sport = 'soccer'"]
    params: list[Any] = []
    if status:
        clauses.append("g.status = ?")
        params.append(status)
    if league_ids:
        placeholders = ",".join("?" for _ in league_ids)
        clauses.append(f"g.league_id IN ({placeholders})")
        params.extend(league_ids)
    sql = f"""
        SELECT g.game_uid, g.league_id, g.season, g.game_date, g.kickoff_utc, g.status,
               g.home_team_uid, g.away_team_uid, g.home_score, g.away_score,
               th.canonical_name AS home_name, ta.canonical_name AS away_name,
               s.home_shots, s.away_shots, s.home_sot, s.away_sot,
               s.home_corners, s.away_corners, s.home_fouls, s.away_fouls,
               s.home_yellow, s.away_yellow, s.home_red, s.away_red, s.referee
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        LEFT JOIN soccer_match_stats s ON s.game_uid = g.game_uid
        WHERE {' AND '.join(clauses)}
        ORDER BY g.game_date, g.game_uid
    """
    df = query_df(sql, params)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_odds_wide(sport: str = "soccer", market: str = "1x2",
                   bookmakers: Sequence[str] | None = None) -> pd.DataFrame:
    """Pivot odds snapshots to one row per game.

    Columns are named ``odds_{bookmaker}_{selection}_{open|close}`` so a
    backtest can address decision-time and closing prices explicitly and can
    never accidentally stake off a closing line.
    """
    params: list[Any] = [sport, market]
    book_clause = ""
    if bookmakers:
        book_clause = f"AND bookmaker IN ({','.join('?' for _ in bookmakers)})"
        params.extend(bookmakers)
    raw = query_df(
        f"""
        SELECT game_uid, bookmaker, selection, is_closing, AVG(price_decimal) AS price
        FROM odds_snapshots
        WHERE sport = ? AND market = ? {book_clause}
        GROUP BY game_uid, bookmaker, selection, is_closing
        """,
        params,
    )
    if raw.empty:
        return raw
    raw["phase"] = raw["is_closing"].map({0: "open", 1: "close"})
    raw["column"] = ("odds_" + raw["bookmaker"] + "_" + raw["selection"] + "_" + raw["phase"])
    wide = raw.pivot_table(index="game_uid", columns="column", values="price", aggfunc="mean")
    return wide.reset_index()


def latest_odds(game_uid: str, market: str | None = None) -> pd.DataFrame:
    """Most recent price per (market, selection, bookmaker) for one game."""
    params: list[Any] = [game_uid]
    market_clause = ""
    if market:
        market_clause = "AND market = ?"
        params.append(market)
    sql = f"""
        SELECT o.* FROM odds_snapshots o
        JOIN (
            SELECT market, selection, bookmaker, MAX(captured_at) AS mx
            FROM odds_snapshots WHERE game_uid = ? {market_clause}
            GROUP BY market, selection, bookmaker
        ) latest
          ON latest.market = o.market AND latest.selection = o.selection
         AND latest.bookmaker = o.bookmaker AND latest.mx = o.captured_at
        WHERE o.game_uid = ?
    """
    params.append(game_uid)
    return query_df(sql, params)


def odds_history(game_uid: str, market: str, selection: str | None = None) -> pd.DataFrame:
    params: list[Any] = [game_uid, market]
    clause = ""
    if selection:
        clause = "AND selection = ?"
        params.append(selection)
    return query_df(
        f"SELECT * FROM odds_snapshots WHERE game_uid = ? AND market = ? {clause} "
        "ORDER BY captured_at",
        params,
    )


def source_status_table() -> pd.DataFrame:
    return query_df("SELECT * FROM source_status ORDER BY source, dataset")
