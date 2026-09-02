"""Soccer walk-forward backtest against real historical prices.

This is the platform's most trustworthy evaluation, because football-data
publishes both a pre-match and a closing price for every match.  Bets are
placed at the **pre-match** price, exactly as a bettor deciding before
kick-off would, and the closing price is used afterwards only to measure CLV.

Each season is predicted by a model fitted solely on earlier seasons, then the
window rolls forward — approximating how the live system actually evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import SOCCER_LEAGUES, settings
from ..db.repository import load_odds_wide, load_soccer_matches
from ..features.soccer_features import SoccerFeatureConfig, build_soccer_dataset
from ..logging_setup import get_logger
from ..models.calibration import multiclass_brier, multiclass_log_loss
from ..models.soccer_model import SoccerMatchModel, resolve_soccer_features
from .metrics import BacktestMetrics, BetResult, bucket_analysis, summarise
from .walkforward import BettingPolicy, simulate_market

log = get_logger(__name__)

#: Decision-time book.  ``market_avg`` is football-data's pre-match average
#: across bookmakers — a conservative choice, since a real bettor shopping for
#: the best price would do better than the average.
DEFAULT_BOOK = "market_avg"

#: CLV is measured against Pinnacle's closing price.  Pinnacle is the standard
#: sharp reference; measuring "best price now" against "average price at close"
#: would flatter every bet, because the best price is above the average by
#: construction.
DEFAULT_CLOSING_BOOK = "pinnacle"


@dataclass
class SoccerBacktestResult:
    predictions: pd.DataFrame
    bets: list[BetResult]
    metrics: BacktestMetrics
    probability_metrics: dict[str, Any]
    by_edge: pd.DataFrame
    by_price: pd.DataFrame
    by_league: pd.DataFrame
    folds: list[dict[str, Any]] = field(default_factory=list)
    totals_metrics: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "totals_metrics": self.totals_metrics,
            "probability_metrics": self.probability_metrics,
            "by_edge": self.by_edge.to_dict("records") if not self.by_edge.empty else [],
            "by_price": self.by_price.to_dict("records") if not self.by_price.empty else [],
            "by_league": self.by_league.to_dict("records") if not self.by_league.empty else [],
            "folds": self.folds,
        }


def prepare_dataset(league_ids: Sequence[str] | None = None,
                    config: SoccerFeatureConfig | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features joined to prices, plus the raw match frame for model fitting."""
    league_ids = list(league_ids or SOCCER_LEAGUES.keys())
    matches = load_soccer_matches(league_ids)
    if matches.empty:
        raise RuntimeError("no soccer matches ingested — run `divinelines refresh --sport soccer`")

    dataset, _ = build_soccer_dataset(matches, config)
    for market in ("1x2", "totals"):
        odds = load_odds_wide("soccer", market)
        if not odds.empty:
            dataset = dataset.merge(odds, on="game_uid", how="left")
    dataset = dataset[dataset["outcome"].notna()].copy()
    dataset["outcome"] = dataset["outcome"].astype(int)
    dataset = attach_market_probabilities(dataset)
    return dataset, matches


def attach_market_probabilities(dataset: pd.DataFrame, book: str = DEFAULT_BOOK
                                ) -> pd.DataFrame:
    """Add de-vigged decision-time market probabilities.

    These come from the *pre-match* price only.  They are legitimate inputs (a
    bettor can see them before kick-off) but they are kept in clearly named
    columns so no analysis can confuse them with the closing line.
    """
    from ..betting.odds_math import remove_vig

    columns = {s: f"odds_{book}_{s}_open" for s in ("home", "draw", "away")}
    if not all(c in dataset.columns for c in columns.values()):
        for selection in ("home", "draw", "away"):
            dataset[f"market_prob_{selection}"] = np.nan
        return dataset

    values = {s: [] for s in columns}
    for _, row in dataset.iterrows():
        prices = [row.get(columns[s]) for s in ("home", "draw", "away")]
        if any(p is None or pd.isna(p) or float(p) <= 1.0 for p in prices):
            for selection in columns:
                values[selection].append(np.nan)
            continue
        fair = remove_vig([float(p) for p in prices])
        for selection, probability in zip(("home", "draw", "away"), fair):
            values[selection].append(probability)
    for selection in ("home", "draw", "away"):
        dataset[f"market_prob_{selection}"] = values[selection]
    return dataset


def run_soccer_backtest(
    league_ids: Sequence[str] | None = None,
    *,
    variant: str = "full",
    seasons: Sequence[str] | None = None,
    policy: BettingPolicy | None = None,
    book: str = DEFAULT_BOOK,
    closing_book: str | None = None,
    min_train_matches: int = 1500,
    use_market: bool = False,
    price_book: str | None = None,
) -> SoccerBacktestResult:
    """Backtest 1X2 betting.

    ``book`` is the consensus used to judge fair value; ``price_book`` is where
    the bet is struck.  Setting ``price_book='market_best'`` simulates line
    shopping — taking the best price on offer while judging value against the
    market average — which is a legitimate source of edge in its own right and
    is exactly what the live scanner does with multi-book odds.
    """
    dataset, matches = prepare_dataset(league_ids)
    closing_book = closing_book or DEFAULT_CLOSING_BOOK
    price_book = price_book or book
    policy = policy or BettingPolicy()

    price_columns = {s: f"odds_{price_book}_{s}_open" for s in ("home", "draw", "away")}
    consensus_columns = {s: f"odds_{book}_{s}_open" for s in ("home", "draw", "away")}
    closing_columns = {s: f"odds_{closing_book}_{s}_close" for s in ("home", "draw", "away")}
    missing = [c for c in price_columns.values() if c not in dataset.columns]
    if missing:
        raise RuntimeError(f"price columns unavailable in ingested odds: {missing}")

    all_seasons = sorted(dataset["season"].unique())
    target_seasons = list(seasons) if seasons else all_seasons

    features_all = resolve_soccer_features(variant, dataset.columns)
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []

    for season in target_seasons:
        test = dataset[dataset["season"] == season]
        if test.empty:
            continue
        cutoff = test["game_date"].min()
        history = dataset[dataset["game_date"] < cutoff]
        if len(history) < min_train_matches:
            log.info("skipping season: insufficient history",
                     extra={"season": season, "history": len(history)})
            continue

        split_at = int(len(history) * 0.85)
        train, valid = history.iloc[:split_at], history.iloc[split_at:]
        raw_history = matches[matches["game_date"] < cutoff]

        model = SoccerMatchModel(
            features=list(features_all), variant=variant, use_market=use_market,
            use_classifier=variant not in ("elo_only", "dixon_coles_only"),
        )
        model.fit(train, valid, raw_matches=raw_history)

        probabilities = model.predict_proba(test)
        fold_predictions = test.copy()
        fold_predictions["prob_home"] = probabilities[:, 0]
        fold_predictions["prob_draw"] = probabilities[:, 1]
        fold_predictions["prob_away"] = probabilities[:, 2]
        fold_predictions["market"] = "1x2"

        # Totals come straight from the Dixon-Coles scoreline distribution —
        # the same fitted goal expectations, no separate model.
        totals = [
            model.dixon_coles.predict_match(row["home_team_uid"], row["away_team_uid"],
                                            row.get("league_id"))["totals"]
            for _, row in test.iterrows()
        ]
        fold_predictions["prob_over_2.5"] = [t["over_2.5"] for t in totals]
        fold_predictions["prob_under_2.5"] = [t["under_2.5"] for t in totals]
        predictions.append(fold_predictions)

        y_true = test["outcome"].to_numpy()
        folds.append(
            {
                "season": season,
                "train_matches": int(len(train)),
                "valid_matches": int(len(valid)),
                "test_matches": int(len(test)),
                "log_loss": round(multiclass_log_loss(y_true, probabilities), 5),
                "brier": round(multiclass_brier(y_true, probabilities), 5),
                "accuracy": round(float(np.mean(probabilities.argmax(axis=1) == y_true)), 4),
                "blend_weights": model.metrics["blend_weights"],
            }
        )
        log.info("soccer fold complete", extra=folds[-1])

    if not predictions:
        raise RuntimeError("no seasons had enough history to backtest")

    combined = pd.concat(predictions, ignore_index=True)
    bets, metrics = simulate_market(
        combined,
        price_columns=price_columns,
        consensus_columns=consensus_columns,
        closing_columns=closing_columns,
        outcome_column="outcome_selection",
        policy=policy,
        probability_columns={"home": "prob_home", "draw": "prob_draw", "away": "prob_away"},
    )

    y_true = combined["outcome"].to_numpy()
    matrix = combined[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    market_matrix = _market_probabilities(combined, consensus_columns)
    probability_metrics = {
        "model": {
            "log_loss": round(multiclass_log_loss(y_true, matrix), 5),
            "brier": round(multiclass_brier(y_true, matrix), 5),
            "accuracy": round(float(np.mean(matrix.argmax(axis=1) == y_true)), 4),
            "n": int(len(combined)),
        },
        "market_novig": (
            {
                "log_loss": round(multiclass_log_loss(y_true[market_matrix[1]],
                                                      market_matrix[0]), 5),
                "brier": round(multiclass_brier(y_true[market_matrix[1]], market_matrix[0]), 5),
                "n": int(market_matrix[1].sum()),
            }
            if market_matrix[1].any() else None
        ),
    }

    totals_metrics = _simulate_totals(combined, policy)

    return SoccerBacktestResult(
        predictions=combined,
        bets=bets,
        metrics=metrics,
        totals_metrics=totals_metrics,
        probability_metrics=probability_metrics,
        by_edge=bucket_analysis(bets, "edge"),
        by_price=bucket_analysis(bets, "price"),
        by_league=bucket_analysis(bets, "league_id"),
        folds=folds,
    )


def _simulate_totals(combined: pd.DataFrame, policy: BettingPolicy) -> dict[str, Any] | None:
    """Over/under 2.5 simulation from the same Dixon-Coles fit."""
    price_columns = {
        "over_2.5": "odds_market_avg_over_2.5_open",
        "under_2.5": "odds_market_avg_under_2.5_open",
    }
    if not all(c in combined.columns for c in price_columns.values()):
        return None
    frame = combined.dropna(subset=list(price_columns.values()) + ["total_goals"]).copy()
    if frame.empty:
        return None
    frame["outcome_selection_totals"] = np.where(
        frame["total_goals"] > 2.5, "over_2.5", "under_2.5"
    )
    bets, metrics = simulate_market(
        frame,
        price_columns=price_columns,
        outcome_column="outcome_selection_totals",
        policy=policy,
        probability_columns={"over_2.5": "prob_over_2.5", "under_2.5": "prob_under_2.5"},
    )
    summary = metrics.to_dict()
    summary["by_price"] = (bucket_analysis(bets, "price").to_dict("records")
                           if bets else [])
    return summary


def _market_probabilities(frame: pd.DataFrame, price_columns: dict[str, str]
                          ) -> tuple[np.ndarray, np.ndarray]:
    """No-vig market probabilities, used as the benchmark to beat."""
    from ..betting.odds_math import remove_vig

    rows: list[list[float]] = []
    mask: list[bool] = []
    for _, row in frame.iterrows():
        prices = [row.get(price_columns[s]) for s in ("home", "draw", "away")]
        if any(p is None or pd.isna(p) or float(p) <= 1.0 for p in prices):
            mask.append(False)
            continue
        try:
            rows.append(remove_vig([float(p) for p in prices]))
            mask.append(True)
        except (ValueError, ZeroDivisionError):
            mask.append(False)
    return (np.asarray(rows, dtype=float) if rows else np.zeros((0, 3))), np.asarray(mask)
