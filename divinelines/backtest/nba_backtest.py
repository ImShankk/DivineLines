"""NBA walk-forward backtest.

An important honesty constraint shapes this module: **the platform has no
historical NBA odds.**  The-Odds-API's free tier serves current prices only, so
there is no way to reconstruct what a book offered on a night in 2023.

So this module reports **no ROI figure at all**.  It evaluates probability
quality — log loss, Brier, calibration, skill over the base rate — on genuine
out-of-sample seasons, which needs no odds and is the part that can be trusted.

A synthetic market was tried and deliberately removed: any price series
invented from the model's own components is beatable by construction, and the
resulting "backtest" showed a 10% ROI that meant nothing.  What replaces it is
a break-even table showing, for each probability band, the realised frequency
and the price the model would need — enough to judge where the edge lives
without pretending to know what a book offered in 2023.

Live NBA predictions are priced against real multi-book odds, and their CLV is
tracked forward from the day the platform starts recording snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import settings
from ..db.repository import load_nba_team_games
from ..features.nba_features import NbaFeatureConfig, build_nba_dataset
from ..logging_setup import get_logger
from ..models.calibration import evaluate_probabilities, reliability_curve
from ..models.nba_model import NbaWinModel, resolve_features

log = get_logger(__name__)

NO_ODDS_NOTE = (
    "No historical NBA odds exist in this platform, so no ROI, profit or CLV "
    "figure is reported for NBA backtests. Probability quality is measured "
    "instead; betting performance is tracked forward in the live ledger."
)


@dataclass
class NbaBacktestResult:
    predictions: pd.DataFrame
    probability_metrics: dict[str, Any]
    reliability: list[dict[str, float]]
    folds: list[dict[str, Any]]
    break_even: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_variant: str = "full"
    note: str = NO_ODDS_NOTE

    def summary(self) -> dict[str, Any]:
        return {
            "probability_metrics": self.probability_metrics,
            "reliability": self.reliability,
            "folds": self.folds,
            "break_even": self.break_even.to_dict("records") if not self.break_even.empty else [],
            "feature_variant": self.feature_variant,
            "note": self.note,
        }


def prepare_nba_dataset(config: NbaFeatureConfig | None = None) -> pd.DataFrame:
    team_games = load_nba_team_games()
    if team_games.empty:
        raise RuntimeError("no NBA games ingested — run `divinelines refresh --sport nba`")
    dataset, _ = build_nba_dataset(team_games, config)
    return dataset


def run_nba_backtest(
    dataset: pd.DataFrame | None = None,
    *,
    variant: str = "full",
    seasons: Sequence[str] | None = None,
    min_train_rows: int = 800,
) -> NbaBacktestResult:
    """Refit before every season and predict it out of sample."""
    dataset = prepare_nba_dataset() if dataset is None else dataset
    dataset = dataset[dataset["home_win"].notna()].copy()
    features = resolve_features(variant, dataset.columns)

    all_seasons = sorted(dataset["season"].unique())
    target_seasons = list(seasons) if seasons else all_seasons

    fold_predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []

    for season in target_seasons:
        test = dataset[dataset["season"] == season]
        if test.empty:
            continue
        cutoff = test["game_date"].min()
        history = dataset[dataset["game_date"] < cutoff]
        if len(history) < min_train_rows:
            log.info("skipping season: insufficient history",
                     extra={"season": season, "rows": len(history)})
            continue

        split_at = int(len(history) * 0.85)
        train, valid = history.iloc[:split_at], history.iloc[split_at:]
        model = NbaWinModel(features=list(features), variant=variant).fit(train, valid)

        detail = model.predict_detail(test)
        frame = test.copy()
        frame["model_probability"] = detail["probability"]
        frame["prob_xgboost"] = detail["xgboost"]
        frame["prob_logistic"] = detail["logistic"]
        frame["prob_elo"] = detail["elo"]
        frame["agreement"] = detail["agreement"]
        frame["outcome"] = frame["home_win"].astype(int)
        fold_predictions.append(frame)

        metrics = evaluate_probabilities(frame["outcome"], frame["model_probability"])
        folds.append(
            {
                "season": season,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "test_rows": int(len(test)),
                "metrics": metrics.to_dict(),
                "blend_weights": model.metrics["blend_weights"],
            }
        )
        log.info("NBA fold complete", extra={"season": season, **metrics.to_dict()})

    if not fold_predictions:
        raise RuntimeError("no seasons had enough history to backtest")

    combined = pd.concat(fold_predictions, ignore_index=True)
    y_true = combined["outcome"].to_numpy()

    probability_metrics = {
        "ensemble": evaluate_probabilities(y_true, combined["model_probability"]).to_dict(),
        "xgboost": evaluate_probabilities(y_true, combined["prob_xgboost"]).to_dict(),
        "logistic": evaluate_probabilities(y_true, combined["prob_logistic"]).to_dict(),
        "elo": evaluate_probabilities(y_true, combined["prob_elo"]).to_dict(),
        "home_team_baseline": evaluate_probabilities(
            y_true, np.full(len(y_true), float(y_true.mean()))
        ).to_dict(),
    }

    return NbaBacktestResult(
        predictions=combined,
        probability_metrics=probability_metrics,
        reliability=reliability_curve(y_true, combined["model_probability"].to_numpy()),
        folds=folds,
        break_even=break_even_table(combined),
        feature_variant=variant,
    )


def break_even_table(predictions: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """Realised frequency and required price, per predicted-probability band.

    For each band this reports what the model said, what actually happened, and
    the decimal price at which a bet on that band breaks even.  Comparing the
    required price against prices a book typically offers is the honest way to
    ask "is there edge here?" when no historical prices are available.
    """
    if predictions.empty:
        return pd.DataFrame()

    frame = predictions[["model_probability", "outcome"]].copy()
    frame["band"] = pd.cut(frame["model_probability"], np.linspace(0.0, 1.0, bins + 1),
                           include_lowest=True)
    grouped = frame.groupby("band", observed=True).agg(
        games=("outcome", "size"),
        predicted=("model_probability", "mean"),
        realised=("outcome", "mean"),
    ).reset_index()
    grouped = grouped[grouped["games"] > 0]
    grouped["calibration_gap"] = grouped["realised"] - grouped["predicted"]
    grouped["fair_price_model"] = 1.0 / grouped["predicted"].clip(lower=1e-6)
    grouped["fair_price_realised"] = 1.0 / grouped["realised"].clip(lower=1e-6)
    grouped["band"] = grouped["band"].astype(str)
    return grouped
