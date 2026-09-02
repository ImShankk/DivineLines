"""Walk-forward backtesting and market simulation.

Two rules govern everything here, because breaking either one produces
backtests that look wonderful and mean nothing:

1. **Only past data trains a model.**  Each period is predicted by a model
   fitted exclusively on games that finished before that period began, and
   refitted as the window advances — which is how the live system actually
   evolves.
2. **Only prices available at decision time are used.**  Simulated bets take
   the pre-match price (``is_closing = 0``); the closing price is loaded
   *afterwards* and used solely to measure closing-line value.  Staking off the
   closing line is the single most common way a soccer backtest lies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..betting.ev import expected_value
from ..betting.kelly import recommend_stake
from ..betting.odds_math import remove_vig
from ..config import settings
from ..logging_setup import get_logger
from .metrics import BacktestMetrics, BetResult, summarise

log = get_logger(__name__)


@dataclass
class Fold:
    period: str
    train_rows: int
    valid_rows: int
    test_rows: int
    train_end: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    folds: list[Fold]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_predictions": int(len(self.predictions)),
            "folds": [
                {
                    "period": f.period, "train_rows": f.train_rows, "valid_rows": f.valid_rows,
                    "test_rows": f.test_rows, "train_end": f.train_end, "metrics": f.metrics,
                }
                for f in self.folds
            ],
        }


def walk_forward(
    dataset: pd.DataFrame,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame],
    *,
    period_column: str = "season",
    date_column: str = "game_date",
    min_train_rows: int = 500,
    valid_fraction: float = 0.15,
    periods: Sequence[str] | None = None,
) -> WalkForwardResult:
    """Roll forward one period at a time, retraining before each.

    ``fit_predict(train, valid, test)`` must return a frame indexed like
    ``test`` containing at least a ``model_probability`` column; anything else
    it returns (component probabilities, model id) is carried through.
    """
    ordered = dataset.sort_values(date_column).reset_index(drop=True)
    all_periods = list(dict.fromkeys(ordered[period_column].tolist()))
    target_periods = list(periods) if periods else all_periods

    outputs: list[pd.DataFrame] = []
    folds: list[Fold] = []

    for period in target_periods:
        test = ordered[ordered[period_column] == period]
        if test.empty:
            continue
        cutoff = test[date_column].min()
        history = ordered[ordered[date_column] < cutoff]
        if len(history) < min_train_rows:
            log.info("skipping period: insufficient history",
                     extra={"period": period, "rows": len(history)})
            continue

        split_at = int(len(history) * (1 - valid_fraction))
        train, valid = history.iloc[:split_at], history.iloc[split_at:]
        if valid.empty:
            train, valid = history, history.tail(max(50, len(history) // 10))

        predictions = fit_predict(train, valid, test)
        predictions = predictions.copy()
        predictions["period"] = period
        outputs.append(predictions)
        folds.append(
            Fold(
                period=str(period), train_rows=len(train), valid_rows=len(valid),
                test_rows=len(test), train_end=str(history[date_column].max().date()),
            )
        )
        log.info("walk-forward period complete",
                 extra={"period": period, "train": len(train), "test": len(test)})

    combined = (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame())
    return WalkForwardResult(predictions=combined, folds=folds)


# --------------------------------------------------------------------------
# Market simulation
# --------------------------------------------------------------------------

@dataclass
class BettingPolicy:
    """The rules a simulated bettor follows.  Mirrors the live risk config."""

    min_edge: float = settings.betting.min_edge
    kelly_multiplier: float = settings.betting.kelly_fraction
    max_stake_pct: float = settings.betting.max_stake_pct
    bankroll: float = settings.betting.bankroll
    #: Stake as a fixed fraction of the starting bankroll instead of Kelly.
    flat_stake_pct: float | None = None
    min_price: float = settings.betting.min_decimal_odds
    max_price: float = settings.betting.max_decimal_odds
    #: Cap on how far the model may disagree with the market before the bet is
    #: rejected as a probable model error rather than an edge.
    max_disagreement: float = settings.betting.model_outlier_threshold
    #: Compound the bankroll across bets, as a real bettor would.
    compound: bool = True
    devig_method: str = "power"


def simulate_market(
    predictions: pd.DataFrame,
    *,
    price_columns: dict[str, str],
    closing_columns: dict[str, str] | None = None,
    outcome_column: str = "outcome_selection",
    policy: BettingPolicy | None = None,
    probability_columns: dict[str, str] | None = None,
    consensus_columns: dict[str, str] | None = None,
) -> tuple[list[BetResult], BacktestMetrics]:
    """Simulate betting a set of predictions against real historical prices.

    ``price_columns`` maps selection -> the decision-time price actually taken;
    ``consensus_columns`` maps selection -> the prices used to derive the fair
    (no-vig) market probability, which defaults to the same book;
    ``probability_columns`` maps selection -> model probability column;
    ``closing_columns`` (optional) maps selection -> closing price column, used
    only for CLV after the bet has been placed.
    """
    policy = policy or BettingPolicy()
    probability_columns = probability_columns or {
        selection: f"prob_{selection}" for selection in price_columns
    }
    consensus_columns = consensus_columns or price_columns
    selections = list(price_columns.keys())

    results: list[BetResult] = []
    bankroll = policy.bankroll

    for _, row in predictions.sort_values("game_date").iterrows():
        prices = {s: row.get(price_columns[s]) for s in selections}
        if any(p is None or pd.isna(p) or float(p) <= 1.0 for p in prices.values()):
            continue
        consensus = {s: row.get(consensus_columns[s]) for s in selections}
        if any(p is None or pd.isna(p) or float(p) <= 1.0 for p in consensus.values()):
            continue
        try:
            fair = dict(zip(selections,
                            remove_vig([float(consensus[s]) for s in selections],
                                       policy.devig_method)))
        except (ValueError, ZeroDivisionError):
            continue

        for selection in selections:
            model_probability = row.get(probability_columns[selection])
            if model_probability is None or pd.isna(model_probability):
                continue
            model_probability = float(model_probability)
            price = float(prices[selection])
            if not policy.min_price <= price <= policy.max_price:
                continue

            edge = model_probability - fair[selection]
            if edge < policy.min_edge:
                continue
            if abs(edge) > policy.max_disagreement:
                # Treat an extreme disagreement as a model outlier, not a gift.
                continue

            ev = expected_value(model_probability, price, fair[selection])
            if ev.ev_per_unit <= 0:
                continue

            active_bankroll = bankroll if policy.compound else policy.bankroll
            if policy.flat_stake_pct is not None:
                stake = policy.flat_stake_pct * active_bankroll
            else:
                stake = recommend_stake(
                    model_probability=model_probability,
                    price_decimal=price,
                    market_probability=fair[selection],
                    confidence=1.0,
                    bankroll=active_bankroll,
                    kelly_multiplier=policy.kelly_multiplier,
                    max_stake_pct=policy.max_stake_pct,
                ).stake
            if stake <= 0:
                continue

            won = str(row[outcome_column]) == selection
            closing_price = None
            closing_probability = None
            if closing_columns and selection in closing_columns:
                raw_closing = row.get(closing_columns[selection])
                if raw_closing is not None and not pd.isna(raw_closing) and float(raw_closing) > 1:
                    closing_price = float(raw_closing)
                    closing_all = [row.get(closing_columns.get(s)) for s in selections]
                    if all(v is not None and not pd.isna(v) and float(v) > 1 for v in closing_all):
                        closing_fair = dict(zip(selections, remove_vig(
                            [float(v) for v in closing_all], policy.devig_method)))
                        closing_probability = closing_fair[selection]

            bet = BetResult(
                game_uid=str(row.get("game_uid")),
                date=pd.Timestamp(row["game_date"]),
                league_id=row.get("league_id"),
                market=str(row.get("market", "1x2")),
                selection=selection,
                price_decimal=price,
                stake=float(stake),
                model_probability=model_probability,
                market_probability=fair[selection],
                edge=edge,
                won=won,
                closing_price=closing_price,
                closing_probability=closing_probability,
            )
            results.append(bet)
            bankroll += bet.profit

    graded = predictions.copy()
    metrics = summarise(results, starting_bankroll=policy.bankroll,
                        predictions=graded if "model_probability" in graded.columns else None)
    return results, metrics
