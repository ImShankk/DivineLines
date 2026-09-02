"""Model health.

Two questions that V2 kept conflating, separated here for good:

* **Predictive health** — are the probabilities any good? Brier, log loss,
  calibration, and skill against the market. Needs only graded predictions.
* **Betting health** — does acting on them make money, and do our prices beat
  the close? Needs prices and settled stakes.

A model can be predictively excellent and commercially useless (the market is
simply better priced), which is exactly what V2 found for soccer. Reporting a
single "model score" would have hidden that.

Status is derived from evidence with stated thresholds, never from vibes, and
``MARKET_BEATING`` requires beating the market's own log loss on a sample large
enough to mean something.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..betting.clv import MIN_SAMPLE_FOR_INFERENCE, summarise_clv
from ..db.connection import query_df, upsert_rows
from ..db.repository import now_iso
from ..logging_setup import get_logger
from ..models.calibration import (
    evaluate_probabilities,
    expected_calibration_error,
    reliability_curve,
)

log = get_logger(__name__)

STATUS_INSUFFICIENT = "INSUFFICIENT_SAMPLE"
STATUS_UNPROVEN = "UNPROVEN"
STATUS_VALIDATED = "VALIDATED_FOR_PREDICTION"
STATUS_MARKET_BEATING = "MARKET_BEATING"
STATUS_DEGRADED = "DEGRADED"

#: Thresholds behind the status labels. Written down so a status can be argued
#: with rather than trusted.
MIN_SAMPLE_FOR_STATUS = 100
#: Brier skill over the base rate needed before probabilities are called useful.
MIN_BRIER_SKILL = 0.02
#: Log-loss advantage over the market needed for MARKET_BEATING. Small edges on
#: small samples are noise, so this is paired with a sample floor.
MIN_MARKET_EDGE = 0.005
MIN_SAMPLE_FOR_MARKET_CLAIM = 200
#: Recent-vs-historical log-loss deterioration that counts as degradation.
DEGRADATION_THRESHOLD = 0.03

WINDOWS: tuple[tuple[str, int | None], ...] = (
    ("all_time", None),
    ("last_90d", 90),
    ("last_30d", 30),
)


@dataclass
class HealthResult:
    sport: str
    market: str | None
    model_version: str | None
    window_label: str
    sample_size: int
    predictive: dict[str, Any] = field(default_factory=dict)
    market_comparison: dict[str, Any] = field(default_factory=dict)
    betting: dict[str, Any] = field(default_factory=dict)
    calibration_curve: list[dict[str, Any]] = field(default_factory=list)
    stability: dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_INSUFFICIENT
    status_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sport": self.sport, "market": self.market,
            "model_version": self.model_version, "window": self.window_label,
            "sample_size": self.sample_size,
            "predictive": self.predictive, "market_comparison": self.market_comparison,
            "betting": self.betting, "calibration_curve": self.calibration_curve,
            "stability": self.stability,
            "status": self.status, "status_reason": self.status_reason,
        }


def _graded_predictions(sport: str | None, market: str | None, model_version: str | None,
                        since: str | None) -> pd.DataFrame:
    """Predictions on finished games, with the outcome attached.

    Superseded predictions are excluded: if a prediction was replaced after a
    lineup arrived, scoring both would double-count the same match and let the
    later, better-informed version flatter the average.
    """
    clauses = ["g.status = 'final'", "p.superseded_at IS NULL"]
    params: list[Any] = []
    if sport:
        clauses.append("p.sport = ?")
        params.append(sport)
    if market:
        clauses.append("p.market = ?")
        params.append(market)
    if model_version:
        clauses.append("p.model_version = ?")
        params.append(model_version)
    if since:
        clauses.append("p.created_at >= ?")
        params.append(since)

    return query_df(
        f"""
        SELECT p.prediction_id, p.created_at, p.sport, p.league_id, p.market, p.selection,
               p.model_prob, p.market_prob, p.price_decimal, p.model_version,
               p.prediction_stage, p.lineup_state, p.game_uid,
               g.home_score, g.away_score,
               c.clv_price_pct, c.clv_same_book_pct, c.result, c.stake, c.profit
        FROM predictions p
        JOIN games g ON g.game_uid = p.game_uid
        LEFT JOIN clv_records c ON c.prediction_id = p.prediction_id
        WHERE {' AND '.join(clauses)}
        ORDER BY p.created_at
        """,
        params,
    )


def _outcome(row: pd.Series) -> int | None:
    """1 if the predicted selection occurred, else 0."""
    home, away = row["home_score"], row["away_score"]
    if pd.isna(home) or pd.isna(away):
        return None
    selection = str(row["selection"])
    if home > away:
        winner = "home"
    elif home < away:
        winner = "away"
    else:
        winner = "draw"
    if selection in ("home", "away", "draw"):
        return int(selection == winner)
    if selection.startswith(("over_", "under_")):
        try:
            side, line = selection.split("_", 1)
            total = float(home) + float(away)
            return int(total > float(line)) if side == "over" else int(total < float(line))
        except ValueError:
            return None
    return None


def compute_health(sport: str, *, market: str | None = None,
                   model_version: str | None = None,
                   window_label: str = "all_time",
                   window_days: int | None = None) -> HealthResult:
    since = None
    if window_days:
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    frame = _graded_predictions(sport, market, model_version, since)
    result = HealthResult(sport=sport, market=market, model_version=model_version,
                          window_label=window_label, sample_size=0)
    if frame.empty:
        result.status_reason = "no graded predictions in this window"
        return result

    frame = frame.copy()
    frame["outcome"] = frame.apply(_outcome, axis=1)
    frame = frame[frame["outcome"].notna()]
    result.sample_size = len(frame)
    if frame.empty:
        result.status_reason = "no gradeable predictions in this window"
        return result

    y_true = frame["outcome"].astype(int).to_numpy()
    y_prob = frame["model_prob"].astype(float).to_numpy()
    metrics = evaluate_probabilities(y_true, y_prob)
    result.predictive = metrics.to_dict()
    result.calibration_curve = reliability_curve(y_true, y_prob)

    # Market comparison on the subset that actually had a market price, so the
    # two log losses are computed over identical rows.
    priced = frame[frame["market_prob"].notna()]
    if not priced.empty:
        market_true = priced["outcome"].astype(int).to_numpy()
        market_prob = priced["market_prob"].astype(float).to_numpy()
        model_on_same = priced["model_prob"].astype(float).to_numpy()
        model_metrics = evaluate_probabilities(market_true, model_on_same)
        market_metrics = evaluate_probabilities(market_true, market_prob)
        result.market_comparison = {
            "n": int(len(priced)),
            "model_log_loss": model_metrics.log_loss,
            "market_log_loss": market_metrics.log_loss,
            "model_brier": model_metrics.brier,
            "market_brier": market_metrics.brier,
            "skill_vs_market": round(market_metrics.log_loss - model_metrics.log_loss, 6),
            "beats_market": model_metrics.log_loss < market_metrics.log_loss,
        }

    clv_values = frame["clv_price_pct"].dropna().tolist()
    same_book = frame["clv_same_book_pct"].dropna().tolist()
    graded = frame[frame["result"].notna()]
    staked = float(graded["stake"].fillna(0).sum()) if not graded.empty else 0.0
    profit = float(graded["profit"].fillna(0).sum()) if not graded.empty else 0.0
    clv_summary = summarise_clv(clv_values)

    result.betting = {
        "clv": clv_summary.to_dict(),
        "roi_interval": roi_interval(
            graded["stake"].fillna(0).tolist() if not graded.empty else [],
            graded["profit"].fillna(0).tolist() if not graded.empty else [],
        ),
        "clv_same_book": summarise_clv(same_book).to_dict() if same_book else None,
        "settled_bets": int(len(graded)),
        "staked": round(staked, 2),
        "profit": round(profit, 2),
        "roi": round(profit / staked, 4) if staked > 0 else None,
        "max_drawdown": _max_drawdown(graded["profit"].fillna(0).tolist())
        if not graded.empty else None,
    }
    result.stability = prediction_stability(sport, model_version)

    result.status, result.status_reason = _classify(result)
    return result


def roi_interval(stakes: Sequence[float], profits: Sequence[float],
                 *, confidence: float = 0.95) -> dict[str, Any]:
    """ROI with a confidence interval on the per-unit return.

    A bare ROI figure invites the reader to treat +4.8% as an established edge.
    Betting returns are extremely heavy-tailed, so the interval is usually wide
    enough to make clear that a few thousand bets is not proof of anything.
    """
    stake_array = np.asarray(list(stakes), dtype=float)
    profit_array = np.asarray(list(profits), dtype=float)
    mask = stake_array > 0
    stake_array, profit_array = stake_array[mask], profit_array[mask]
    if stake_array.size == 0:
        return {"n": 0, "roi": None, "ci_low": None, "ci_high": None,
                "significant": False, "interpretation": "no settled stakes"}

    returns = profit_array / stake_array
    n = int(returns.size)
    roi = float(profit_array.sum() / stake_array.sum())
    std = float(returns.std(ddof=1)) if n > 1 else 0.0

    if n < 30:
        return {"n": n, "roi": round(roi, 5), "ci_low": None, "ci_high": None,
                "significant": False,
                "interpretation": f"sample too small for an interval (n={n})"}

    mean_return = float(returns.mean())
    if std == 0:
        # Every bet returned the same amount. Degenerate, but the interval is
        # genuinely a point, so the sign is certain rather than unknown.
        return {"n": n, "roi": round(roi, 5), "mean_unit_return": round(mean_return, 5),
                "ci_low": round(mean_return, 5), "ci_high": round(mean_return, 5),
                "significant": mean_return != 0,
                "interpretation": ("every settled bet returned the same amount; "
                                   "no variance to account for")}

    z = 1.959963985 if abs(confidence - 0.95) < 1e-9 else 2.575829304
    margin = z * std / math.sqrt(n)
    low, high = mean_return - margin, mean_return + margin
    significant = low > 0 or high < 0
    return {
        "n": n, "roi": round(roi, 5),
        "mean_unit_return": round(mean_return, 5),
        "ci_low": round(low, 5), "ci_high": round(high, 5),
        "significant": significant,
        "interpretation": (
            "profitable at 95% confidence" if significant and low > 0
            else "losing at 95% confidence" if significant
            else "consistent with break-even once variance is accounted for"
        ),
    }


def _max_drawdown(profits: Sequence[float]) -> float:
    if not profits:
        return 0.0
    equity = np.cumsum(profits)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity - peak))


def _classify(result: HealthResult) -> tuple[str, str]:
    """Map evidence onto a status, with the reason spelled out."""
    n = result.sample_size
    if n < MIN_SAMPLE_FOR_STATUS:
        return (STATUS_INSUFFICIENT,
                f"{n} graded predictions; {MIN_SAMPLE_FOR_STATUS} needed before any "
                f"claim is meaningful")

    skill = result.predictive.get("brier_skill", 0.0) or 0.0
    comparison = result.market_comparison or {}
    market_n = comparison.get("n", 0)
    market_edge = comparison.get("skill_vs_market")

    if skill < 0:
        return (STATUS_DEGRADED,
                f"Brier skill {skill:+.4f}: worse than always predicting the base rate")

    if (market_edge is not None and market_edge > MIN_MARKET_EDGE
            and market_n >= MIN_SAMPLE_FOR_MARKET_CLAIM):
        return (STATUS_MARKET_BEATING,
                f"log loss {market_edge:.4f} better than the no-vig market over "
                f"{market_n} priced predictions")

    if skill >= MIN_BRIER_SKILL:
        reason = (f"Brier skill {skill:+.4f} over the base rate on {n} predictions; ")
        if market_edge is None:
            reason += "no market comparison available"
        elif market_n < MIN_SAMPLE_FOR_MARKET_CLAIM:
            reason += (f"market comparison sample too small "
                       f"({market_n} < {MIN_SAMPLE_FOR_MARKET_CLAIM})")
        else:
            reason += f"but the market is still better by {abs(market_edge):.4f} log loss"
        return STATUS_VALIDATED, reason

    return (STATUS_UNPROVEN,
            f"Brier skill {skill:+.4f} below the {MIN_BRIER_SKILL} threshold on "
            f"{n} predictions")


def prediction_stability(sport: str, model_version: str | None = None) -> dict[str, Any]:
    """How much the probability moves between versions of the same prediction.

    A model that swings 51% -> 72% -> 48% on one fixture as information trickles
    in is not responding to news, it is unstable. Measured only where a game
    actually has multiple prediction versions.
    """
    clauses = ["p.sport = ?"]
    params: list[Any] = [sport]
    if model_version:
        clauses.append("p.model_version = ?")
        params.append(model_version)

    frame = query_df(
        f"""
        SELECT p.game_uid, p.market, p.selection, p.model_prob, p.created_at
        FROM predictions p
        WHERE {' AND '.join(clauses)}
        ORDER BY p.game_uid, p.market, p.selection, p.created_at
        """,
        params,
    )
    if frame.empty:
        return {"n_series": 0, "note": "no predictions"}

    moves: list[float] = []
    ranges: list[float] = []
    for _, group in frame.groupby(["game_uid", "market", "selection"]):
        if len(group) < 2:
            continue
        probs = group["model_prob"].astype(float).to_numpy()
        moves.extend(np.abs(np.diff(probs)).tolist())
        ranges.append(float(probs.max() - probs.min()))

    if not moves:
        return {"n_series": 0,
                "note": "no fixture yet has more than one prediction version"}

    return {
        "n_series": len(ranges),
        "mean_abs_move": round(float(np.mean(moves)), 5),
        "max_move": round(float(np.max(moves)), 5),
        "mean_range": round(float(np.mean(ranges)), 5),
        "p95_range": round(float(np.percentile(ranges, 95)), 5),
    }


def health_report(sports: Iterable[str] = ("nba", "soccer"),
                  markets: Iterable[str | None] = (None,)) -> dict[str, Any]:
    """Health across sports, markets and time windows."""
    report: dict[str, Any] = {"computed_at": now_iso(), "sports": {}}
    for sport in sports:
        entries: dict[str, Any] = {}
        for window_label, days in WINDOWS:
            entries[window_label] = compute_health(
                sport, window_label=window_label, window_days=days
            ).to_dict()
        by_market: dict[str, Any] = {}
        for market in markets:
            if market is None:
                continue
            by_market[market] = compute_health(sport, market=market).to_dict()
        report["sports"][sport] = {"windows": entries, "markets": by_market}
    return report


def persist_snapshot(result: HealthResult) -> None:
    """Store a health snapshot. Aggregates stay reproducible from the ledger."""
    upsert_rows("model_health_snapshots", [{
        "computed_at": now_iso(), "sport": result.sport, "league_id": None,
        "market": result.market, "model_version": result.model_version,
        "window_label": result.window_label, "window_start": None, "window_end": None,
        "sample_size": result.sample_size,
        "brier": result.predictive.get("brier"),
        "log_loss": result.predictive.get("log_loss"),
        "accuracy": result.predictive.get("accuracy"),
        "calibration_error": result.predictive.get("ece"),
        "market_brier": (result.market_comparison or {}).get("market_brier"),
        "market_log_loss": (result.market_comparison or {}).get("market_log_loss"),
        "skill_vs_market": (result.market_comparison or {}).get("skill_vs_market"),
        "clv_mean": (result.betting.get("clv") or {}).get("mean_clv_price_pct"),
        "clv_median": (result.betting.get("clv") or {}).get("median_clv_price_pct"),
        "clv_positive_rate": (result.betting.get("clv") or {}).get("beat_close_rate"),
        "clv_sample": (result.betting.get("clv") or {}).get("n"),
        "clv_ci_low": (result.betting.get("clv") or {}).get("ci_low"),
        "clv_ci_high": (result.betting.get("clv") or {}).get("ci_high"),
        "roi": result.betting.get("roi"), "profit": result.betting.get("profit"),
        "settled_bets": result.betting.get("settled_bets"),
        "max_drawdown": result.betting.get("max_drawdown"),
        "prediction_volatility": result.stability.get("mean_abs_move"),
        "status": result.status, "status_reason": result.status_reason,
        "metrics_json": json.dumps(result.to_dict(), default=str),
    }])


def detect_regression(sport: str, *, recent_days: int = 30) -> dict[str, Any]:
    """Compare recent predictive quality against the all-time record.

    Deploying a model that has quietly got worse is the failure this guards
    against; it reports rather than acts, because a small recent sample is a
    bad reason to roll anything back automatically.
    """
    recent = compute_health(sport, window_label="recent", window_days=recent_days)
    lifetime = compute_health(sport, window_label="all_time")

    if recent.sample_size < MIN_SAMPLE_FOR_INFERENCE or lifetime.sample_size == 0:
        return {
            "sport": sport, "regression": False,
            "reason": f"recent sample too small ({recent.sample_size})",
            "recent_n": recent.sample_size, "lifetime_n": lifetime.sample_size,
        }

    recent_ll = recent.predictive.get("log_loss")
    lifetime_ll = lifetime.predictive.get("log_loss")
    if recent_ll is None or lifetime_ll is None or lifetime_ll <= 0:
        return {"sport": sport, "regression": False, "reason": "log loss unavailable"}

    delta = (recent_ll - lifetime_ll) / lifetime_ll
    return {
        "sport": sport,
        "regression": delta > DEGRADATION_THRESHOLD,
        "recent_log_loss": round(recent_ll, 5),
        "lifetime_log_loss": round(lifetime_ll, 5),
        "relative_change": round(delta, 5),
        "threshold": DEGRADATION_THRESHOLD,
        "recent_n": recent.sample_size, "lifetime_n": lifetime.sample_size,
        "reason": (f"recent log loss {delta:+.2%} vs lifetime"
                   if delta > DEGRADATION_THRESHOLD else "within tolerance"),
    }
