"""Training entry points.

Trains, evaluates and registers models for both sports.  Registration records
the data fingerprint, feature set, hyperparameters, seed and validation
metrics, so any prediction can later be traced back to exactly what produced
it — and so "did version 3 actually beat version 2?" is answerable from the
prediction ledger rather than from memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from ..config import SOCCER_LEAGUES, settings
from ..db.repository import load_nba_team_games, load_soccer_matches
from ..features.nba_features import FEATURE_SET_VERSION as NBA_FEATURES_VERSION
from ..features.nba_features import build_nba_dataset
from ..features.soccer_features import FEATURE_SET_VERSION as SOCCER_FEATURES_VERSION
from ..features.soccer_features import build_soccer_dataset
from ..logging_setup import get_logger
from ..models.nba_model import (
    DEFAULT_VARIANT as NBA_DEFAULT_VARIANT,
    MODEL_VERSION as NBA_MODEL_VERSION,
    NbaWinModel,
    chronological_split,
    resolve_features,
)
from ..models.nba_player_impact import fit_margin_to_probability
from ..models.registry import ModelRecord, data_version_hash, make_model_id, register
from ..models.soccer_model import (
    MODEL_VERSION as SOCCER_MODEL_VERSION,
    SoccerMatchModel,
    resolve_soccer_features,
)

log = get_logger(__name__)


@dataclass
class TrainingOutcome:
    model_id: str
    sport: str
    metrics: dict[str, Any]
    n_train: int
    n_valid: int
    n_test: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "sport": self.sport, "metrics": self.metrics,
            "n_train": self.n_train, "n_valid": self.n_valid, "n_test": self.n_test,
        }


def train_nba(*, variant: str | None = None, register_model: bool = True) -> TrainingOutcome:
    variant = variant or NBA_DEFAULT_VARIANT
    team_games = load_nba_team_games()
    if team_games.empty:
        raise RuntimeError("no NBA data — run `divinelines refresh --sport nba` first")

    dataset, builder = build_nba_dataset(team_games)
    dataset = dataset[dataset["home_win"].notna()].copy()
    split = chronological_split(dataset)
    features = resolve_features(variant, dataset.columns)

    model = NbaWinModel(features=features, variant=variant,
                        feature_set_version=NBA_FEATURES_VERSION)
    model.fit(split.train, split.valid)

    test_metrics = model.evaluate(split.test).to_dict()
    metrics = dict(model.metrics)
    metrics["test"] = test_metrics
    metrics["split"] = split.describe()
    metrics["feature_importance"] = model.feature_importance(12)

    # The margin -> probability conversion used by the injury adjustment is
    # fitted here so it travels with the model rather than floating free.
    logits_per_point = fit_margin_to_probability(dataset)
    metrics["logits_per_margin_point"] = logits_per_point

    model_id = make_model_id("nba", "ensemble", NBA_MODEL_VERSION)
    if register_model:
        register(
            ModelRecord(
                model_id=model_id, sport="nba", kind="ensemble",
                model_version=NBA_MODEL_VERSION, feature_set=features,
                feature_set_version=NBA_FEATURES_VERSION,
                hyperparameters=model.xgb_params,
                league_id="NBA",
                train_start=str(split.train["game_date"].min().date()),
                train_end=str(split.train["game_date"].max().date()),
                valid_start=str(split.valid["game_date"].min().date()),
                valid_end=str(split.valid["game_date"].max().date()),
                metrics=metrics, data_version=data_version_hash(),
                random_seed=settings.model.random_seed,
                notes=f"variant={variant}",
            ),
            payload={
                "model": model,
                "feature_builder": builder,
                "logits_per_margin_point": logits_per_point,
                "trained_through": str(dataset["game_date"].max().date()),
            },
        )
    return TrainingOutcome(model_id, "nba", metrics, len(split.train), len(split.valid),
                           len(split.test))


def train_soccer(league_ids: Sequence[str] | None = None, *, variant: str = "full",
                 use_market: bool = False, register_model: bool = True) -> TrainingOutcome:
    league_ids = list(league_ids or SOCCER_LEAGUES.keys())
    matches = load_soccer_matches(league_ids)
    if matches.empty:
        raise RuntimeError("no soccer data — run `divinelines refresh --sport soccer` first")

    dataset, builder = build_soccer_dataset(matches)
    dataset = dataset[dataset["outcome"].notna()].copy()
    dataset["outcome"] = dataset["outcome"].astype(int)

    if use_market:
        from ..backtest.soccer_backtest import attach_market_probabilities
        from ..db.repository import load_odds_wide

        odds = load_odds_wide("soccer", "1x2")
        if not odds.empty:
            dataset = dataset.merge(odds, on="game_uid", how="left")
        dataset = attach_market_probabilities(dataset)

    ordered = dataset.sort_values("game_date").reset_index(drop=True)
    test_start = int(len(ordered) * (1 - settings.model.test_fraction))
    valid_start = int(test_start * 0.85)
    train, valid, test = (ordered.iloc[:valid_start], ordered.iloc[valid_start:test_start],
                          ordered.iloc[test_start:])

    features = resolve_soccer_features(variant, dataset.columns)
    model = SoccerMatchModel(features=features, variant=variant, use_market=use_market,
                             feature_set_version=SOCCER_FEATURES_VERSION)
    model.fit(train, valid, raw_matches=matches[matches["game_date"] <= train["game_date"].max()])

    from ..models.calibration import multiclass_brier, multiclass_log_loss

    test_probabilities = model.predict_proba(test)
    y_test = test["outcome"].to_numpy()
    metrics = dict(model.metrics)
    metrics["test"] = {
        "log_loss": multiclass_log_loss(y_test, test_probabilities),
        "brier": multiclass_brier(y_test, test_probabilities),
        "n": int(len(test)),
    }
    metrics["leagues"] = league_ids

    model_id = make_model_id("soccer", "ensemble", SOCCER_MODEL_VERSION)
    if register_model:
        register(
            ModelRecord(
                model_id=model_id, sport="soccer", kind="ensemble",
                model_version=SOCCER_MODEL_VERSION, feature_set=features,
                feature_set_version=SOCCER_FEATURES_VERSION,
                hyperparameters={"dixon_coles": vars(model.dc_config),
                                 "use_market": use_market},
                train_start=str(train["game_date"].min().date()),
                train_end=str(train["game_date"].max().date()),
                valid_start=str(valid["game_date"].min().date()),
                valid_end=str(valid["game_date"].max().date()),
                metrics=metrics, data_version=data_version_hash(),
                random_seed=settings.model.random_seed,
                notes=f"variant={variant}; leagues={','.join(league_ids)}",
            ),
            payload={
                "model": model,
                "feature_builder": builder,
                "trained_through": str(dataset["game_date"].max().date()),
                "leagues": league_ids,
            },
        )
    return TrainingOutcome(model_id, "soccer", metrics, len(train), len(valid), len(test))


def train_all() -> list[TrainingOutcome]:
    outcomes: list[TrainingOutcome] = []
    for trainer in (train_nba, train_soccer):
        try:
            outcomes.append(trainer())
        except Exception as exc:  # one sport failing must not block the other
            log.error("training failed", extra={"trainer": trainer.__name__, "error": str(exc)})
    return outcomes
