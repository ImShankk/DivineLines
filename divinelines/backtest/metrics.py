"""Backtest metrics.

Win rate alone says almost nothing: a model can win 60% of its bets and still
lose money at short prices, or win 40% and profit at long ones.  These metrics
cover profitability, risk, and — most importantly for small samples —
probability quality and closing-line value, which converge much faster than
profit does.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..models.calibration import evaluate_probabilities, reliability_curve


@dataclass
class BetResult:
    """One simulated bet, with everything needed to grade and diagnose it."""

    game_uid: str
    date: pd.Timestamp
    league_id: str | None
    market: str
    selection: str
    price_decimal: float
    stake: float
    model_probability: float
    market_probability: float | None
    edge: float
    won: bool
    push: bool = False
    closing_price: float | None = None
    closing_probability: float | None = None

    @property
    def profit(self) -> float:
        if self.push:
            return 0.0
        return self.stake * (self.price_decimal - 1.0) if self.won else -self.stake

    @property
    def clv_price_pct(self) -> float | None:
        if not self.closing_price:
            return None
        return (self.price_decimal / self.closing_price - 1.0) * 100.0


@dataclass
class BacktestMetrics:
    bets: int
    staked: float
    profit: float
    roi: float
    hit_rate: float
    avg_price: float
    avg_edge: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe: float
    volatility: float
    final_bankroll: float
    starting_bankroll: float
    clv_mean_pct: float | None
    clv_beat_rate: float | None
    brier: float | None
    log_loss: float | None
    ece: float | None
    n_predictions: int
    #: Profit if every bet were priced at its no-vig fair value: the gap
    #: between this and realised profit is what the market's margin costs.
    expected_profit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in payload.items()}


def bankroll_path(results: Sequence[BetResult], starting_bankroll: float) -> pd.DataFrame:
    rows = []
    bankroll = starting_bankroll
    peak = starting_bankroll
    for bet in sorted(results, key=lambda b: b.date):
        bankroll += bet.profit
        peak = max(peak, bankroll)
        rows.append(
            {
                "date": bet.date,
                "profit": bet.profit,
                "bankroll": bankroll,
                "drawdown": bankroll - peak,
                "drawdown_pct": (bankroll - peak) / peak if peak else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarise(results: Sequence[BetResult], *, starting_bankroll: float = 1000.0,
              predictions: pd.DataFrame | None = None,
              probability_column: str = "model_probability",
              outcome_column: str = "outcome") -> BacktestMetrics:
    """Aggregate simulated bets into the full metric set.

    ``predictions`` (all graded predictions, not only the ones bet) is used for
    the probability-quality metrics, because calibration should be judged on
    every forecast, not on the filtered subset that cleared the EV threshold.
    """
    if not results:
        empty_probability = (None, None, None, 0)
        if predictions is not None and not predictions.empty:
            metrics = evaluate_probabilities(
                predictions[outcome_column].astype(int), predictions[probability_column]
            )
            empty_probability = (metrics.brier, metrics.log_loss, metrics.ece, metrics.n)
        return BacktestMetrics(
            bets=0, staked=0.0, profit=0.0, roi=0.0, hit_rate=0.0, avg_price=0.0,
            avg_edge=0.0, max_drawdown=0.0, max_drawdown_pct=0.0, sharpe=0.0,
            volatility=0.0, final_bankroll=starting_bankroll,
            starting_bankroll=starting_bankroll, clv_mean_pct=None, clv_beat_rate=None,
            brier=empty_probability[0], log_loss=empty_probability[1],
            ece=empty_probability[2], n_predictions=empty_probability[3],
        )

    staked = float(sum(b.stake for b in results))
    profit = float(sum(b.profit for b in results))
    path = bankroll_path(results, starting_bankroll)
    returns = np.array([b.profit / b.stake for b in results if b.stake > 0], dtype=float)

    clv_values = [b.clv_price_pct for b in results if b.clv_price_pct is not None]
    expected_profit = None
    if all(b.market_probability is not None for b in results):
        expected_profit = float(sum(
            b.stake * (b.market_probability * (b.price_decimal - 1.0) - (1 - b.market_probability))
            for b in results
        ))

    brier = log_loss = ece = None
    n_predictions = 0
    if predictions is not None and not predictions.empty:
        metrics = evaluate_probabilities(
            predictions[outcome_column].astype(int), predictions[probability_column]
        )
        brier, log_loss, ece, n_predictions = (
            metrics.brier, metrics.log_loss, metrics.ece, metrics.n
        )

    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    mean_return = float(returns.mean()) if len(returns) else 0.0

    return BacktestMetrics(
        bets=len(results),
        staked=staked,
        profit=profit,
        roi=profit / staked if staked else 0.0,
        hit_rate=float(np.mean([b.won for b in results])),
        avg_price=float(np.mean([b.price_decimal for b in results])),
        avg_edge=float(np.mean([b.edge for b in results])),
        max_drawdown=float(path["drawdown"].min()) if not path.empty else 0.0,
        max_drawdown_pct=float(path["drawdown_pct"].min()) if not path.empty else 0.0,
        # Per-bet Sharpe analogue: mean return over its standard deviation.
        sharpe=float(mean_return / volatility * np.sqrt(len(returns))) if volatility > 0 else 0.0,
        volatility=volatility,
        final_bankroll=starting_bankroll + profit,
        starting_bankroll=starting_bankroll,
        clv_mean_pct=float(np.mean(clv_values)) if clv_values else None,
        clv_beat_rate=float(np.mean([v > 0 for v in clv_values])) if clv_values else None,
        brier=brier, log_loss=log_loss, ece=ece, n_predictions=n_predictions,
        expected_profit=expected_profit,
    )


def bucket_analysis(results: Sequence[BetResult], by: str = "edge") -> pd.DataFrame:
    """Realised performance by edge band or price band.

    If larger predicted edges do not produce better realised results, the model
    is overconfident in its tails — which is invisible in an aggregate ROI.
    """
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(
        [
            {
                "edge": b.edge, "price": b.price_decimal, "stake": b.stake,
                "profit": b.profit, "won": int(b.won),
                "model_probability": b.model_probability,
                "market_probability": b.market_probability,
                "league_id": b.league_id, "selection": b.selection,
            }
            for b in results
        ]
    )
    if by == "edge":
        frame["bucket"] = pd.cut(frame["edge"], [-1, 0.02, 0.04, 0.06, 0.10, 1.0],
                                 labels=["<2%", "2-4%", "4-6%", "6-10%", "10%+"])
    elif by == "price":
        frame["bucket"] = pd.cut(frame["price"], [1.0, 1.5, 2.0, 3.0, 5.0, 1000.0],
                                 labels=["heavy fav", "fav", "even", "dog", "longshot"])
    elif by in frame.columns:
        frame["bucket"] = frame[by]
    else:
        raise ValueError(f"unknown bucket dimension '{by}'")

    grouped = frame.groupby("bucket", observed=True).agg(
        bets=("profit", "size"),
        staked=("stake", "sum"),
        profit=("profit", "sum"),
        hit_rate=("won", "mean"),
        avg_edge=("edge", "mean"),
        avg_price=("price", "mean"),
    )
    grouped["roi"] = grouped["profit"] / grouped["staked"].replace(0, np.nan)
    return grouped.reset_index()


def calibration_report(predictions: pd.DataFrame, probability_column: str = "model_probability",
                       outcome_column: str = "outcome", bins: int = 10) -> dict[str, Any]:
    if predictions.empty:
        return {"metrics": None, "curve": []}
    y_true = predictions[outcome_column].astype(int).to_numpy()
    y_prob = predictions[probability_column].astype(float).to_numpy()
    return {
        "metrics": evaluate_probabilities(y_true, y_prob).to_dict(),
        "curve": reliability_curve(y_true, y_prob, bins),
    }
