"""DivineLines API.

Serves the analytics application and preserves the original ``POST
/api/predict`` contract so nothing that already talks to this service breaks.

Two operational concerns are handled here rather than left to each endpoint:
expensive work (feature building, model loading) is cached with a TTL, and
failures return a proper HTTP status with a readable message instead of a
200 containing ``{"message": "Engine Error: ..."}``.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..config import SOCCER_LEAGUES, settings
from ..data.freshness import freshness_report
from ..db.connection import query_df, table_exists
from ..db.repository import load_games, odds_history, source_status_table
from ..db.validation import run_database_health_checks
from ..logging_setup import get_logger
from ..models.registry import get_metrics, list_models
from ..version import API_VERSION
from .schemas import (
    GameOut,
    HealthResponse,
    LegacyMatchup,
    PerformanceResponse,
    ScanRequest,
    ScanResponse,
    SourceHealth,
)

log = get_logger(__name__)

app = FastAPI(
    title="DivineLines API",
    version=API_VERSION,
    description="Quantitative sports research, market analysis and risk management.",
)

app.add_middleware(
    CORSMiddleware,
    # The service binds to localhost and serves a local dashboard; tighten this
    # before exposing it on a network. A regex rather than a fixed list because
    # Vite walks up from 5173 whenever that port is busy, and a dev server on
    # 5175 failing CORS looks exactly like a broken endpoint.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):(4173|5173|517[4-9]|518\d)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Small TTL cache — scans rebuild features and must not run per request.
# --------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, Any]] = {}


def cached(key: str, ttl: int, producer: Callable[[], Any]) -> tuple[Any, bool]:
    now = time.monotonic()
    entry = _CACHE.get(key)
    if entry and now - entry[0] < ttl:
        return entry[1], True
    value = producer()
    _CACHE[key] = (now, value)
    return value, False


def invalidate_cache(prefix: str = "") -> None:
    for key in [k for k in _CACHE if k.startswith(prefix)]:
        _CACHE.pop(key, None)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records.

    pandas uses NaN/NaT for missing values; JSON and pydantic both need
    ``null``. Converting in one place stops a single missing column from
    turning an endpoint into a 500.
    """
    if frame is None or frame.empty:
        return []
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict("records")


def scalar(value: Any) -> Any:
    """Single value -> JSON-safe scalar."""
    if value is None:
        return None
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


# --------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------

system_router = APIRouter(tags=["system"])


@system_router.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "DivineLines",
        "version": API_VERSION,
        "mode": settings.mode,
        "status": "online",
        "docs": "/docs",
    }


@system_router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    database: dict[str, Any] = {"connected": False}
    try:
        counts = query_df(
            "SELECT sport, COUNT(*) AS games, MAX(game_date) AS latest FROM games GROUP BY sport"
        )
        database = {
            "connected": True,
            "path": str(settings.paths.db_path),
            "games": records(counts),
            "odds_snapshots": int(query_df("SELECT COUNT(*) AS n FROM odds_snapshots")["n"].iloc[0]),
            "predictions": int(query_df("SELECT COUNT(*) AS n FROM predictions")["n"].iloc[0]),
        }
    except Exception as exc:
        database = {"connected": False, "error": str(exc)}

    statuses = source_status_table()
    freshness = {f.dataset: f for f in freshness_report()}
    sources: list[SourceHealth] = []
    for _, row in statuses.iterrows():
        entry = freshness.get(row["dataset"])
        sources.append(
            SourceHealth(
                source=row["source"], dataset=row["dataset"], status=scalar(row.get("status")),
                last_success=scalar(row.get("last_success")),
                age_minutes=entry.age_minutes if entry else None,
                state=entry.state if entry else "unknown",
                message=scalar(row.get("message")),
            )
        )

    models = list_models(limit=5)
    model_rows = [
        {
            "model_id": row["model_id"], "sport": row["sport"], "kind": row["kind"],
            "model_version": row["model_version"], "trained_at": row["trained_at"],
        }
        for _, row in models.iterrows()
    ] if not models.empty else []

    report = run_database_health_checks()
    validation = {
        "ok": report.ok,
        "critical": [{"code": i.code, "detail": i.detail} for i in report.critical],
        "warnings": [{"code": i.code, "detail": i.detail} for i in report.warnings],
    }

    if not database.get("connected") or not report.ok:
        status = "down" if not database.get("connected") else "degraded"
    elif not model_rows or any(s.state in ("stale", "missing") for s in sources):
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(status=status, mode=settings.mode, checked_at=checked_at,
                          database=database, sources=sources, models=model_rows,
                          validation=validation)


@system_router.get("/api/config")
def configuration() -> dict[str, Any]:
    """Non-secret configuration, so the UI can show the policy it is applying."""
    return {
        "mode": settings.mode,
        "betting": {
            "bankroll": settings.betting.bankroll,
            "kelly_fraction": settings.betting.kelly_fraction,
            "max_stake_pct": settings.betting.max_stake_pct,
            "max_slate_exposure_pct": settings.betting.max_slate_exposure_pct,
            "max_game_exposure_pct": settings.betting.max_game_exposure_pct,
            "max_team_exposure_pct": settings.betting.max_team_exposure_pct,
            "min_edge": settings.betting.min_edge,
            "min_edge_score": settings.betting.min_edge_score,
            "model_outlier_threshold": settings.betting.model_outlier_threshold,
        },
        "model": {
            "calibration": settings.model.calibration_method,
            "random_seed": settings.model.random_seed,
            "shrinkage_prior_games": settings.model.shrinkage_prior_games,
        },
        "leagues": {
            "soccer": [{"id": k, **v} for k, v in SOCCER_LEAGUES.items()],
            "nba": [{"id": "NBA", "name": "National Basketball Association"}],
        },
        "odds_api_configured": bool(settings.sources.odds_api_key),
    }


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

predictions_router = APIRouter(prefix="/api", tags=["predictions"])


@predictions_router.get("/predictions", response_model=ScanResponse)
def get_predictions(
    sport: str = Query("nba", pattern="^(nba|soccer)$"),
    days_ahead: int = Query(3, ge=1, le=14),
    min_edge: float | None = Query(None, ge=0, le=1),
    league_id: str | None = None,
) -> ScanResponse:
    from ..pipeline.predict import generate_nba_predictions, generate_soccer_predictions

    def produce():
        if sport == "nba":
            return generate_nba_predictions(days_ahead=days_ahead)
        return generate_soccer_predictions(days_ahead=days_ahead)

    try:
        result, was_cached = cached(f"scan:{sport}:{days_ahead}", 600, produce)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = result.to_dict()
    if min_edge is not None:
        payload["opportunities"] = [
            o for o in payload["opportunities"]
            if (o.get("edge") or 0) >= min_edge
        ]
    if league_id:
        payload["opportunities"] = [o for o in payload["opportunities"]
                                    if o["league_id"] == league_id]
        payload["predictions"] = [o for o in payload["predictions"]
                                  if o["league_id"] == league_id]
    payload["cached"] = was_cached
    return ScanResponse(**payload)


@predictions_router.post("/scan")
def run_scan(request: ScanRequest) -> dict[str, Any]:
    from ..pipeline.predict import scan

    invalidate_cache("scan:")
    return scan(request.sports, paper_trade=request.paper_trade,
                days_ahead=request.days_ahead)


@predictions_router.get("/predictions/history")
def prediction_history(sport: str | None = None, limit: int = Query(200, ge=1, le=2000)
                       ) -> dict[str, Any]:
    clause = "WHERE p.sport = ?" if sport else ""
    params = [sport] if sport else []
    rows = query_df(
        f"""
        SELECT p.*, g.home_team_uid, g.away_team_uid, g.status AS game_status,
               g.home_score, g.away_score,
               th.canonical_name AS home_name, ta.canonical_name AS away_name
        FROM predictions p
        LEFT JOIN games g ON g.game_uid = p.game_uid
        LEFT JOIN teams th ON th.team_uid = g.home_team_uid
        LEFT JOIN teams ta ON ta.team_uid = g.away_team_uid
        {clause}
        ORDER BY p.created_at DESC, p.prediction_id DESC LIMIT {int(limit)}
        """,
        params,
    )
    return {"predictions": records(rows), "count": int(len(rows))}


@predictions_router.post("/predict")
def legacy_predict(matchup: LegacyMatchup) -> dict[str, Any]:
    """Original endpoint, preserved.

    Same request and response shape as v1 so existing clients keep working,
    but the probability now comes from the calibrated ensemble and the EV
    figures are computed against de-vigged multi-book consensus prices.
    """
    from ..identity import resolve_nba_team
    from .legacy import legacy_matchup_response

    home = resolve_nba_team(matchup.home)
    away = resolve_nba_team(matchup.away)
    if not home or not away:
        raise HTTPException(status_code=422,
                            detail=f"Unknown team: {matchup.home if not home else matchup.away}")
    if home == away:
        raise HTTPException(status_code=422, detail="A team cannot play itself.")
    try:
        return legacy_matchup_response(home, away)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# --------------------------------------------------------------------------
# Games, teams, odds
# --------------------------------------------------------------------------

data_router = APIRouter(prefix="/api", tags=["data"])


@data_router.get("/games")
def games(sport: str = Query("nba", pattern="^(nba|soccer)$"),
          status: str | None = Query(None, pattern="^(scheduled|final)$"),
          league_id: str | None = None,
          days: int = Query(14, ge=1, le=400),
          limit: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    if status == "final":
        frame = load_games(sport, status="final", league_id=league_id,
                           since=str(today - timedelta(days=days)))
        frame = frame.tail(limit)
    else:
        frame = load_games(sport, status=status, league_id=league_id,
                           since=str(today - timedelta(days=1)),
                           until=str(today + timedelta(days=days)))
        frame = frame.head(limit)
    return {"games": records(frame), "count": int(len(frame))}


@data_router.get("/games/{game_uid:path}")
def game_detail(game_uid: str) -> dict[str, Any]:
    frame = query_df(
        """
        SELECT g.*, th.canonical_name AS home_name, ta.canonical_name AS away_name
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        WHERE g.game_uid = ?
        """,
        (game_uid,),
    )
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"unknown game '{game_uid}'")

    game = {k: scalar(v) for k, v in frame.iloc[0].to_dict().items()}
    markets = query_df(
        "SELECT DISTINCT market FROM odds_snapshots WHERE game_uid = ?", (game_uid,)
    )
    movement = {}
    for market in markets["market"].tolist() if not markets.empty else []:
        history = odds_history(game_uid, market)
        if history.empty:
            continue
        movement[market] = _summarise_movement(history)

    predictions = query_df(
        "SELECT * FROM predictions WHERE game_uid = ? ORDER BY created_at DESC LIMIT 20",
        (game_uid,),
    )
    return {
        "game": game,
        "odds_movement": movement,
        "predictions": records(predictions),
    }


def _summarise_movement(history: pd.DataFrame) -> dict[str, Any]:
    """Opening, current and (when known) closing price per selection."""
    output: dict[str, Any] = {"selections": {}, "series": []}
    for selection, group in history.groupby("selection"):
        ordered = group.sort_values("captured_at")
        opening = float(ordered["price_decimal"].iloc[0])
        current = float(ordered["price_decimal"].iloc[-1])
        closing_rows = ordered[ordered["is_closing"] == 1]
        output["selections"][selection] = {
            "opening": opening,
            "current": current,
            "closing": float(closing_rows["price_decimal"].iloc[-1]) if not closing_rows.empty else None,
            "movement_pct": round((current / opening - 1.0) * 100, 3) if opening else None,
            "snapshots": int(len(ordered)),
        }
        output["series"].extend(
            {
                "selection": selection,
                "captured_at": row["captured_at"],
                "price": float(row["price_decimal"]),
                "bookmaker": row["bookmaker"],
            }
            for _, row in ordered.iterrows()
        )
    return output


@data_router.get("/odds/{game_uid:path}")
def odds_for_game(game_uid: str, market: str | None = None) -> dict[str, Any]:
    from ..db.repository import latest_odds

    latest = latest_odds(game_uid, market)
    if latest.empty:
        raise HTTPException(status_code=404, detail="no odds recorded for this game")
    return {
        "game_uid": game_uid,
        "latest": records(latest),
        "movement": _summarise_movement(odds_history(game_uid, market or
                                                     str(latest["market"].iloc[0]))),
    }


@data_router.get("/teams")
def teams(sport: str = Query("nba", pattern="^(nba|soccer)$"),
          limit: int = Query(400, ge=1, le=1000)) -> dict[str, Any]:
    frame = query_df(
        "SELECT team_uid, canonical_name, abbr, country, sport FROM teams "
        "WHERE sport = ? ORDER BY canonical_name LIMIT ?",
        (sport, limit),
    )
    return {"teams": records(frame), "count": int(len(frame))}


@data_router.get("/teams/{team_uid:path}")
def team_detail(team_uid: str, last: int = Query(15, ge=1, le=60)) -> dict[str, Any]:
    team = query_df("SELECT * FROM teams WHERE team_uid = ?", (team_uid,))
    if team.empty:
        raise HTTPException(status_code=404, detail=f"unknown team '{team_uid}'")

    recent = query_df(
        """
        SELECT g.game_uid, g.game_date, g.season, g.league_id, g.status,
               g.home_team_uid, g.away_team_uid, g.home_score, g.away_score,
               th.canonical_name AS home_name, ta.canonical_name AS away_name
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        WHERE (g.home_team_uid = ? OR g.away_team_uid = ?) AND g.status = 'final'
        ORDER BY g.game_date DESC LIMIT ?
        """,
        (team_uid, team_uid, last),
    )

    injuries = query_df(
        """
        SELECT p.full_name, p.position, s.status, s.detail, s.expected_return, s.as_of
        FROM player_status s LEFT JOIN players p ON p.player_uid = s.player_uid
        JOIN (SELECT player_uid, MAX(retrieved_at) mx FROM player_status GROUP BY player_uid) l
          ON l.player_uid = s.player_uid AND l.mx = s.retrieved_at
        WHERE s.team_uid = ? ORDER BY s.status
        """,
        (team_uid,),
    )
    return {
        "team": {k: scalar(v) for k, v in team.iloc[0].to_dict().items()},
        "recent_games": records(recent),
        "availability": records(injuries),
    }


@data_router.get("/injuries")
def injuries(sport: str = Query("nba", pattern="^(nba|soccer)$")) -> dict[str, Any]:
    frame = query_df(
        """
        SELECT p.full_name, p.position, s.team_uid, t.canonical_name AS team_name,
               s.status, s.detail, s.expected_return, s.as_of, s.retrieved_at, s.source
        FROM player_status s
        JOIN (SELECT player_uid, MAX(retrieved_at) mx FROM player_status
              WHERE sport = ? GROUP BY player_uid) l
          ON l.player_uid = s.player_uid AND l.mx = s.retrieved_at
        LEFT JOIN players p ON p.player_uid = s.player_uid
        LEFT JOIN teams t ON t.team_uid = s.team_uid
        WHERE s.sport = ?
        ORDER BY t.canonical_name, s.status
        """,
        (sport, sport),
    )
    from ..data.freshness import assess

    freshness = assess("injuries", source="espn_nba" if sport == "nba" else None)
    return {
        "injuries": records(frame),
        "count": int(len(frame)),
        "freshness": freshness.to_dict(),
    }


@data_router.get("/search")
def search(q: str = Query(..., min_length=2), limit: int = Query(20, ge=1, le=100)
           ) -> dict[str, Any]:
    pattern = f"%{q.strip()}%"
    teams_found = query_df(
        "SELECT team_uid, canonical_name, sport FROM teams WHERE canonical_name LIKE ? LIMIT ?",
        (pattern, limit),
    )
    players_found = query_df(
        "SELECT player_uid, full_name, sport, team_uid FROM players WHERE full_name LIKE ? LIMIT ?",
        (pattern, limit),
    )
    games_found = query_df(
        """
        SELECT g.game_uid, g.game_date, g.sport, g.league_id,
               th.canonical_name AS home_name, ta.canonical_name AS away_name
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        WHERE th.canonical_name LIKE ? OR ta.canonical_name LIKE ?
        ORDER BY g.game_date DESC LIMIT ?
        """,
        (pattern, pattern, limit),
    )
    return {
        "teams": records(teams_found),
        "players": records(players_found),
        "games": records(games_found),
    }


# --------------------------------------------------------------------------
# Performance and models
# --------------------------------------------------------------------------

analytics_router = APIRouter(prefix="/api", tags=["analytics"])


@analytics_router.get("/performance", response_model=PerformanceResponse)
def performance(mode: str | None = None) -> PerformanceResponse:
    from ..betting.ledger import bankroll_curve, performance_summary

    overall = performance_summary(None, mode)
    by_sport = performance_summary("sport", mode)
    by_edge = performance_summary("edge_bucket", mode)
    by_odds = performance_summary("odds_bucket", mode)
    curve = bankroll_curve(mode)

    clv_frame = query_df(
        "SELECT clv FROM bets WHERE clv IS NOT NULL" + (" AND mode = ?" if mode else ""),
        [mode] if mode else [],
    )
    clv = {
        "n": int(len(clv_frame)),
        "mean_pct": float(clv_frame["clv"].mean()) if not clv_frame.empty else None,
        "beat_close_rate": float((clv_frame["clv"] > 0).mean()) if not clv_frame.empty else None,
    }
    open_bets = int(query_df("SELECT COUNT(*) AS n FROM bets WHERE status='open'")["n"].iloc[0])

    note = None
    if overall.empty:
        note = ("No settled bets yet. Performance populates as predictions are recorded "
                "and graded — run a scan in paper mode, then settle after games finish.")

    return PerformanceResponse(
        overall=records(overall), by_sport=records(by_sport),
        by_edge_bucket=records(by_edge), by_odds_bucket=records(by_odds),
        bankroll_curve=(records(curve.assign(settled_at=curve["settled_at"].astype(str)))
                        if not curve.empty else []),
        clv=clv, open_bets=open_bets, note=note,
    )


@analytics_router.get("/models")
def models(sport: str | None = None, limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    frame = list_models(sport, limit=limit)
    records = []
    for _, row in frame.iterrows() if not frame.empty else []:
        record = row.to_dict()
        record["metrics"] = get_metrics(row["model_id"])
        records.append(record)
    return {"models": records, "count": len(records)}


@analytics_router.get("/experiments")
def experiments(name: str | None = None) -> dict[str, Any]:
    from ..models.registry import experiment_results

    frame = experiment_results(name)
    return {"experiments": records(frame)}


@analytics_router.get("/backtests")
def backtests() -> dict[str, Any]:
    """Stored backtest summaries written by the CLI."""
    import json

    results: dict[str, Any] = {}
    for path in sorted(settings.paths.artifacts_dir.glob("*backtest_summary.json")):
        try:
            results[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("unreadable backtest artifact",
                        extra={"path": str(path), "error": str(exc)})
    if not results:
        return {"backtests": {}, "note": "Run `divinelines backtest` to generate results."}
    return {"backtests": results}


from .soccer import router as soccer_router  # noqa: E402  (kept local to avoid a cycle)
from .v3 import router as v3_router  # noqa: E402

app.include_router(system_router)
app.include_router(predictions_router)
app.include_router(data_router)
app.include_router(analytics_router)
app.include_router(v3_router)
app.include_router(soccer_router)


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("divinelines.api.app:app" if reload else app, host=host, port=port,
                reload=reload)


if __name__ == "__main__":  # pragma: no cover
    run()
