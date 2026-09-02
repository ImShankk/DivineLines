"""Migration from the legacy v1 database into the canonical schema.

The legacy ``data/processed/nba_data.db`` is treated as read-only source data;
it is never modified, so the original application keeps working.

It contains a real, silent corruption that this migration fixes: the 2025-26
season was ingested twice, once with zero-padded ``GAME_ID`` values
(``0022500002``) and once without (``22500002``).  The legacy primary key
``(TEAM_ID, GAME_ID)`` treats those as different games, so every rolling
window silently counted those games twice.  Normalising the id to its padded
form collapses the duplicates.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import settings
from ..logging_setup import get_logger
from .connection import init_db, query_df
from .repository import (
    ensure_leagues,
    ensure_nba_teams,
    game_uid_nba,
    nba_team_uid,
    now_iso,
    record_source_status,
    upsert_games,
    upsert_nba_box,
)
from .validation import validate_games, validate_nba_box

log = get_logger(__name__)

LEGACY_SOURCE = "legacy_nba_data_db"

BOX_COLUMNS = {
    "MIN": "min", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a",
    "FTM": "ftm", "FTA": "fta", "OREB": "oreb", "DREB": "dreb", "REB": "reb",
    "AST": "ast", "STL": "stl", "BLK": "blk", "TOV": "tov", "PF": "pf",
    "PTS": "pts", "PLUS_MINUS": "plus_minus",
}


def season_label_from_season_id(season_id: Any, game_date: pd.Timestamp) -> str:
    """``'22025'`` -> ``'2025-26'``; falls back to the date when malformed."""
    text = str(season_id or "").strip()
    if len(text) == 5 and text.isdigit():
        start = int(text[1:])
        return f"{start}-{str(start + 1)[-2:]}"
    start = game_date.year if game_date.month >= 8 else game_date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def normalise_legacy_game_logs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean the legacy frame: pad ids, drop duplicates, resolve identity."""
    stats: dict[str, int] = {"input_rows": len(df)}

    df = df.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.strip().str.zfill(10)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df[df["GAME_DATE"].notna()]
    stats["dropped_bad_dates"] = stats["input_rows"] - len(df)

    before = len(df)
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).drop_duplicates(
        subset=["GAME_ID", "TEAM_ID"], keep="first"
    )
    stats["duplicate_team_games_removed"] = before - len(df)

    df["team_uid"] = df["TEAM_ID"].map(nba_team_uid)
    unresolved = df["team_uid"].isna().sum()
    if unresolved:
        # Fall back to the abbreviation before giving up on a row.
        mask = df["team_uid"].isna()
        df.loc[mask, "team_uid"] = df.loc[mask, "TEAM_ABBREVIATION"].map(nba_team_uid)
    stats["unresolved_teams"] = int(df["team_uid"].isna().sum())
    df = df[df["team_uid"].notna()]

    df["is_home"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False).astype(int)

    # The legacy sync captured games that were still in progress (no WL, MIN of
    # 0 or a partial total).  Those are not results and must never reach the
    # feature builder; they are kept only as scheduled fixtures.
    finished = (
        df["WL"].astype(str).str.upper().isin(["W", "L"])
        & (pd.to_numeric(df["MIN"], errors="coerce").fillna(0) >= 200)
    )
    finished_games = df.loc[finished].groupby("GAME_ID").size()
    complete_game_ids = set(finished_games[finished_games == 2].index)
    stats["partial_games_excluded"] = int(df["GAME_ID"].nunique() - len(complete_game_ids))
    df.loc[:, "is_final"] = df["GAME_ID"].isin(complete_game_ids)

    sizes = df.groupby("GAME_ID").size()
    complete = sizes[sizes == 2].index
    stats["incomplete_games_dropped"] = int((sizes != 2).sum())
    df = df[df["GAME_ID"].isin(complete)]

    # A game must have exactly one home and one away row.
    home_counts = df.groupby("GAME_ID")["is_home"].sum()
    valid = home_counts[home_counts == 1].index
    stats["ambiguous_home_away_dropped"] = int((home_counts != 1).sum())
    df = df[df["GAME_ID"].isin(valid)]

    stats["output_rows"] = len(df)
    stats["games"] = int(df["GAME_ID"].nunique())
    return df, stats


def _build_frames(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    retrieved = now_iso()
    home = df[df["is_home"] == 1].set_index("GAME_ID")
    away = df[df["is_home"] == 0].set_index("GAME_ID")

    games: list[dict] = []
    for game_id in home.index:
        if game_id not in away.index:
            continue
        h = home.loc[game_id]
        a = away.loc[game_id]
        date = h["GAME_DATE"]
        is_final = bool(h["is_final"])
        games.append(
            {
                "game_uid": game_uid_nba(game_id),
                "sport": "nba",
                "league_id": "NBA",
                "season": season_label_from_season_id(h.get("SEASON_ID"), date),
                "game_date": date.strftime("%Y-%m-%d"),
                "kickoff_utc": None,
                "status": "final" if is_final else "scheduled",
                "home_team_uid": h["team_uid"],
                "away_team_uid": a["team_uid"],
                "home_score": float(h["PTS"]) if is_final and pd.notna(h["PTS"]) else None,
                "away_score": float(a["PTS"]) if is_final and pd.notna(a["PTS"]) else None,
                "neutral_site": 0,
                "venue": None,
                "source": LEGACY_SOURCE,
                "retrieved_at": retrieved,
            }
        )

    final_rows = df[df["is_final"]]
    opponent = final_rows.groupby("GAME_ID")["team_uid"].transform(lambda s: s.iloc[::-1].values)
    box: list[dict] = []
    for row, opp_uid in zip(final_rows.to_dict("records"), opponent):
        record = {
            "game_uid": game_uid_nba(row["GAME_ID"]),
            "team_uid": row["team_uid"],
            "opp_uid": opp_uid,
            "is_home": int(row["is_home"]),
            "won": 1 if str(row.get("WL", "")).upper() == "W" else 0,
            "source": LEGACY_SOURCE,
            "retrieved_at": retrieved,
        }
        for legacy_col, col in BOX_COLUMNS.items():
            value = row.get(legacy_col)
            record[col] = float(value) if pd.notna(value) else None
        box.append(record)
    return games, box


def migrate_legacy_nba(legacy_db: Path | None = None, *, strict: bool = True) -> dict[str, int]:
    """Import the legacy NBA database into the canonical schema."""
    legacy_path = Path(legacy_db or settings.paths.legacy_nba_db)
    if not legacy_path.exists():
        log.warning("legacy database not found", extra={"path": str(legacy_path)})
        record_source_status(LEGACY_SOURCE, "game_logs", status="error",
                             message=f"missing {legacy_path}")
        return {"input_rows": 0, "games": 0}

    init_db()
    ensure_leagues()
    ensure_nba_teams()

    with sqlite3.connect(f"file:{legacy_path}?mode=ro", uri=True) as conn:
        raw = pd.read_sql_query("SELECT * FROM game_logs", conn)

    clean, stats = normalise_legacy_game_logs(raw)
    if clean.empty:
        record_source_status(LEGACY_SOURCE, "game_logs", status="error",
                             message="no usable rows after normalisation")
        return stats

    games, box = _build_frames(clean)

    games_df = pd.DataFrame(games)
    game_report = validate_games(games_df, dataset="nba_games_migration")
    game_report.log()
    game_report.persist()
    if strict:
        game_report.raise_if_critical()

    box_df = pd.DataFrame(box)
    box_report = validate_nba_box(box_df)
    box_report.log()
    box_report.persist()
    if strict:
        box_report.raise_if_critical()

    upsert_games(games)
    upsert_nba_box(box)
    record_source_status(LEGACY_SOURCE, "game_logs", status="ok", rows=len(box),
                         message=f"{stats['duplicate_team_games_removed']} duplicate team-games removed")

    stats["games_written"] = len(games)
    stats["box_rows_written"] = len(box)
    log.info("legacy NBA migration complete", extra=stats)
    return stats


def migration_summary() -> pd.DataFrame:
    return query_df(
        "SELECT season, COUNT(*) AS games, MIN(game_date) AS first_game, "
        "MAX(game_date) AS last_game FROM games WHERE sport='nba' GROUP BY season ORDER BY season"
    )


if __name__ == "__main__":  # pragma: no cover - operational entry point
    result = migrate_legacy_nba()
    print(result)
    print(migration_summary().to_string(index=False))
