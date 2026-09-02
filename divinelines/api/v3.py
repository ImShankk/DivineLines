"""V3 API surface: model health, CLV, timelines, lineups, source health.

Mounted onto the existing app rather than replacing it — the V2 routes are a
contract other things already use.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..analytics.clv_analysis import clv_report, clv_vs_profit
from ..analytics.clv_skill import decompose_clv
from ..analytics.model_health import (
    WINDOWS,
    compute_health,
    detect_regression,
    persist_snapshot,
    prediction_stability,
)
from ..analytics.timeline import event_timeline, lineup_impact_dataset, prediction_versions
from ..betting.closing_line import closing_line_coverage
from ..betting.settlement import settlement_state
from ..config import settings
from ..data.freshness import freshness_report
from ..db.connection import query_df
from ..logging_setup import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["v3"])


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


# --------------------------------------------------------------------- health

@router.get("/model-health")
def model_health(sport: str = Query("nba", pattern="^(nba|soccer)$"),
                 window: str = Query("all_time"),
                 market: str | None = None,
                 model_version: str | None = None,
                 persist: bool = False) -> dict[str, Any]:
    """Predictive health and betting health, kept separate on purpose."""
    windows = dict(WINDOWS)
    if window not in windows:
        raise HTTPException(status_code=422,
                            detail=f"unknown window '{window}'; choose from {sorted(windows)}")

    result = compute_health(sport, market=market, model_version=model_version,
                            window_label=window, window_days=windows[window])
    if persist:
        persist_snapshot(result)

    payload = result.to_dict()
    payload["available_windows"] = sorted(windows)
    payload["note"] = (
        "Predictive health asks whether the probabilities are good. Betting "
        "health asks whether acting on them made money and whether our prices "
        "beat the close. A model can pass one and fail the other."
    )
    return payload


@router.get("/model-health/all")
def model_health_all(sport: str = Query("nba", pattern="^(nba|soccer)$")) -> dict[str, Any]:
    """Every window at once, for the dashboard's trend view."""
    return {
        "sport": sport,
        "windows": {
            label: compute_health(sport, window_label=label, window_days=days).to_dict()
            for label, days in WINDOWS
        },
        "regression": detect_regression(sport),
        "stability": prediction_stability(sport),
    }


@router.get("/model-health/history")
def model_health_history(sport: str = Query("nba", pattern="^(nba|soccer)$"),
                         limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    frame = query_df(
        "SELECT * FROM model_health_snapshots WHERE sport = ? "
        "ORDER BY computed_at DESC LIMIT ?",
        (sport, limit),
    )
    return {"snapshots": _records(frame), "count": int(len(frame))}


@router.get("/models/lifecycle")
def model_lifecycle(sport: str | None = None) -> dict[str, Any]:
    clause = "WHERE l.sport = ?" if sport else ""
    params = [sport] if sport else []
    frame = query_df(
        f"""
        SELECT l.*, m.model_version, m.trained_at
        FROM model_lifecycle l LEFT JOIN models m ON m.model_id = l.model_id
        {clause} ORDER BY l.promoted_at DESC
        """,
        params,
    )
    return {"lifecycle": _records(frame)}


# ------------------------------------------------------------------------ CLV

@router.get("/clv")
def clv(sport: str | None = Query(None, pattern="^(nba|soccer)$"),
        basis: str = Query("consensus", pattern="^(consensus|same_book)$")) -> dict[str, Any]:
    """Full CLV report: distribution, cohorts, cumulative series."""
    report = clv_report(sport, basis=basis)
    report["vs_profit"] = clv_vs_profit(sport)
    report["disclaimer"] = (
        "CLV is a market-efficiency diagnostic, not profit. A bet can lose while "
        "beating the close and win while losing to it."
    )
    return report


@router.get("/clv/skill")
def clv_skill(sport: str = Query("nba", pattern="^(nba|soccer)$"),
              market: str = "h2h", mode: str = "backtest") -> dict[str, Any]:
    """Decompose CLV into line shopping, market drift and model selection."""
    return decompose_clv(sport, market, mode)


@router.get("/clv/coverage")
def clv_coverage(sport: str | None = None) -> dict[str, Any]:
    """How many finished games have a resolvable close.

    A CLV average computed over a small slice of games is not a statement about
    the portfolio, so coverage sits next to the number everywhere it appears.
    """
    frame = closing_line_coverage(sport)
    return {"coverage": _records(frame), "settlement_state": settlement_state()}


@router.post("/settle")
def run_settlement(sport: str | None = None, day: str | None = None,
                   dry_run: bool = False, full_reconcile: bool = False) -> dict[str, Any]:
    from ..betting.settlement import settle

    return settle(sport=sport, day=day, dry_run=dry_run,
                  full_reconcile=full_reconcile).to_dict()


# ------------------------------------------------------------------- timelines

@router.get("/events/{game_uid:path}/timeline")
def timeline(game_uid: str, market: str | None = None) -> dict[str, Any]:
    result = event_timeline(game_uid, market)
    if not result.get("found"):
        raise HTTPException(status_code=404, detail=f"unknown game '{game_uid}'")
    return result


@router.get("/events/{game_uid:path}/predictions")
def versions(game_uid: str) -> dict[str, Any]:
    frame = prediction_versions(game_uid)
    return {"versions": _records(frame), "count": int(len(frame))}


# --------------------------------------------------------------------- lineups

@router.get("/lineups/{game_uid:path}")
def lineups(game_uid: str) -> dict[str, Any]:
    from ..pipeline.ingest_lineups import latest_lineup_state

    frame = query_df(
        """
        SELECT l.*, t.canonical_name AS team_name
        FROM lineup_observations l
        LEFT JOIN teams t ON t.team_uid = l.team_uid
        WHERE l.game_uid = ?
          AND l.observed_at = (SELECT MAX(observed_at) FROM lineup_observations
                               WHERE game_uid = l.game_uid AND team_uid = l.team_uid)
        ORDER BY l.team_uid, l.status DESC, l.formation_place
        """,
        (game_uid,),
    )
    state = latest_lineup_state(game_uid)
    teams: dict[str, Any] = {}
    for record in _records(frame):
        team = record.get("team_name") or record.get("team_uid")
        entry = teams.setdefault(team, {"formation": record.get("formation"),
                                        "starters": [], "bench": [],
                                        "observed_at": record.get("observed_at")})
        bucket = "starters" if record.get("status") == "starter" else "bench"
        entry[bucket].append({
            "player": record.get("player_name"),
            "position_group": record.get("position_group"),
            "role": record.get("role"),
        })

    return {
        "game_uid": game_uid, "lineup_state": state, "teams": teams,
        "freshness": _lineup_freshness(frame),
        "note": ("'final' means the XI was observed after kick-off — a historical "
                 "record, not information that was available beforehand."),
    }


def _lineup_freshness(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"state": "missing", "observed_at": None, "age_minutes": None}
    observed = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce").max()
    if pd.isna(observed):
        return {"state": "unknown", "observed_at": None, "age_minutes": None}
    age = (datetime.now(timezone.utc) - observed.to_pydatetime()).total_seconds() / 60
    state = "fresh" if age < 60 else ("aging" if age < 360 else "stale")
    return {"state": state, "observed_at": str(observed), "age_minutes": round(age, 1)}


@router.get("/lineups")
def lineup_coverage(sport: str = Query("soccer", pattern="^(nba|soccer)$")) -> dict[str, Any]:
    frame = query_df(
        """
        SELECT l.lineup_state, COUNT(DISTINCT l.game_uid) AS games,
               COUNT(*) AS player_rows, MAX(l.observed_at) AS latest
        FROM lineup_observations l WHERE l.sport = ? GROUP BY l.lineup_state
        """,
        (sport,),
    )
    impact = lineup_impact_dataset(sport)
    return {
        "sport": sport, "coverage": _records(frame),
        "impact_pairs": int(len(impact)),
        "impact_sample": _records(impact.head(25)) if not impact.empty else [],
        "note": ("Impact pairs are fixtures with both a pre-lineup and a post-lineup "
                 "prediction. They accumulate forward; historical lineups carry no "
                 "publication timestamp and cannot be replayed."),
    }


# --------------------------------------------------------------- data quality

@router.get("/source-health")
def source_health() -> dict[str, Any]:
    frame = query_df("SELECT * FROM source_status ORDER BY source, dataset")
    freshness = {f.dataset: f.to_dict() for f in freshness_report()}
    return {
        "sources": _records(frame),
        "freshness": freshness,
        "settlement": settlement_state(),
    }


@router.get("/data-quality")
def data_quality() -> dict[str, Any]:
    """What is missing, stale or inconsistent right now."""
    issues = query_df(
        "SELECT dataset, severity, code, COUNT(*) AS n, MAX(detected_at) AS latest "
        "FROM validation_issues GROUP BY dataset, severity, code ORDER BY n DESC LIMIT 50"
    )
    missing_odds = query_df(
        """
        SELECT g.sport, COUNT(*) AS finished_without_price
        FROM games g LEFT JOIN (SELECT DISTINCT game_uid FROM odds_snapshots) o
          ON o.game_uid = g.game_uid
        WHERE g.status = 'final' AND o.game_uid IS NULL GROUP BY g.sport
        """
    )
    upcoming_without_lineups = query_df(
        """
        SELECT COUNT(*) AS n FROM games g
        LEFT JOIN (SELECT DISTINCT game_uid FROM lineup_observations) l
          ON l.game_uid = g.game_uid
        WHERE g.status = 'scheduled' AND l.game_uid IS NULL
        """
    )
    stale = [f.to_dict() for f in freshness_report() if f.state in ("stale", "missing")]

    return {
        "validation_issues": _records(issues),
        "finished_games_without_price": _records(missing_odds),
        "upcoming_without_lineups": int(upcoming_without_lineups["n"].iloc[0])
        if not upcoming_without_lineups.empty else 0,
        "stale_sources": stale,
        "closing_line_coverage": _records(closing_line_coverage()),
    }
