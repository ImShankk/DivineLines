"""NBA win-probability model.

An ensemble of three components that make different mistakes:

* **XGBoost** — captures interactions between form, rest and matchup features;
* **regularised logistic regression** — a stable, low-variance linear view that
  degrades gracefully when a team has little data;
* **Elo** — a pure rating baseline with no feature engineering to overfit.

Blend weights are fitted on a validation block that neither the models nor the
calibrator can see during training, and the blended probability is then
calibrated.  Components are kept individually accessible so the dashboard can
show model agreement, which is a genuine signal about confidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import settings
from ..logging_setup import get_logger
from .calibration import (
    CalibrationMetrics,
    Calibrator,
    calibration_quality,
    evaluate_probabilities,
    reliability_curve,
)

log = get_logger(__name__)

MODEL_VERSION = "nba-ens-2.0"
TARGET = "home_win"

#: Feature groups exist so ablation can switch whole ideas on and off rather
#: than individual columns.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "form_short": tuple(
        f"diff_{m}_r5" for m in ("ortg", "drtg", "net_rating", "pace", "efg", "tov_rate", "win")
    ),
    "form_medium": tuple(
        f"diff_{m}_r10" for m in
        ("ortg", "drtg", "net_rating", "pace", "efg", "ts", "tov_rate", "oreb_pct",
         "ft_rate", "fg3a_rate", "opp_efg", "win")
    ),
    "form_long": tuple(
        f"diff_{m}_r20" for m in ("ortg", "drtg", "net_rating", "efg", "opp_efg", "win")
    ),
    "form_ewma": tuple(
        f"diff_{m}_ewma" for m in ("ortg", "drtg", "net_rating", "efg", "opp_efg", "win")
    ),
    "form_shrunk": tuple(
        f"diff_{m}_shrunk" for m in ("ortg", "drtg", "net_rating", "efg", "opp_efg", "win")
    ),
    "ratings": ("diff_elo", "elo_home_prob", "diff_adj_offense", "diff_adj_defense",
                "diff_adj_net", "adj_matchup_home", "adj_matchup_away"),
    "schedule": ("diff_rest_days", "home_is_b2b", "away_is_b2b", "diff_games_last_7",
                 "home_three_in_four", "away_three_in_four", "diff_travel_km",
                 "diff_tz_shift", "home_road_trip", "away_road_trip"),
    "h2h": ("h2h_games", "h2h_home_win_pct", "h2h_avg_margin"),
    "context": ("neutral_site", "early_season", "min_season_games", "diff_season_games"),
}

#: The variant used unless one is named explicitly.
#:
#: Chosen by the ablation study in ``models.experiments``, not by preference:
#: over three fully out-of-sample seasons (n=2289) the ratings-only feature set
#: produced the best log loss (0.6122) and Brier score, and every richer set was
#: slightly *worse* — form (0.6165), form+schedule (0.6179), everything (0.6183).
#: The gaps are small relative to sampling noise, which is exactly the argument
#: for the simpler model: the extra features buy no measurable accuracy while
#: adding variance and failure modes. Richer variants remain available and are
#: re-tested by ``divinelines ablate``.
DEFAULT_VARIANT = "elo_only"

#: Named variants used by the ablation study.
FEATURE_VARIANTS: dict[str, tuple[str, ...]] = {
    "elo_only": ("ratings",),
    "form_only": ("form_short", "form_medium", "form_long"),
    "form_ewma": ("form_short", "form_medium", "form_long", "form_ewma"),
    "form_ratings": ("form_short", "form_medium", "form_long", "form_ewma", "ratings"),
    "form_ratings_schedule": ("form_short", "form_medium", "form_long", "form_ewma",
                              "ratings", "schedule"),
    "full": tuple(FEATURE_GROUPS.keys()),
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 320,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "min_child_weight": 12,
    "reg_lambda": 3.0,
    "reg_alpha": 0.5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
}


def resolve_features(variant: str | Sequence[str] = "full",
                     available: Iterable[str] | None = None) -> list[str]:
    """Expand a variant name (or list of groups) into concrete column names."""
    groups = FEATURE_VARIANTS.get(variant, None) if isinstance(variant, str) else tuple(variant)
    if groups is None:
        raise ValueError(f"unknown feature variant '{variant}'")
    columns: list[str] = []
    for group in groups:
        columns.extend(FEATURE_GROUPS[group])
    if available is not None:
        available_set = set(available)
        missing = [c for c in columns if c not in available_set]
        if missing:
            log.warning("dropping unavailable features", extra={"missing": missing[:8],
                                                                "count": len(missing)})
        columns = [c for c in columns if c in available_set]
    return list(dict.fromkeys(columns))


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

@dataclass
class ChronoSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict[str, Any]:
        def rng(frame: pd.DataFrame) -> dict[str, Any]:
            if frame.empty:
                return {"n": 0}
            return {
                "n": len(frame),
                "start": str(frame["game_date"].min().date()),
                "end": str(frame["game_date"].max().date()),
            }
        return {"train": rng(self.train), "valid": rng(self.valid), "test": rng(self.test)}


def chronological_split(dataset: pd.DataFrame, *, valid_fraction: float = 0.15,
                        test_fraction: float | None = None) -> ChronoSplit:
    """Split strictly by time.

    A random split would let the model see the second half of a season while
    predicting the first, which inflates every metric.  Sports data must always
    be split chronologically.
    """
    test_fraction = settings.model.test_fraction if test_fraction is None else test_fraction
    ordered = dataset.sort_values("game_date").reset_index(drop=True)
    n = len(ordered)
    test_start = int(n * (1 - test_fraction))
    valid_start = int(test_start * (1 - valid_fraction))
    return ChronoSplit(
        train=ordered.iloc[:valid_start].copy(),
        valid=ordered.iloc[valid_start:test_start].copy(),
        test=ordered.iloc[test_start:].copy(),
    )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class ComponentPrediction:
    xgboost: np.ndarray
    logistic: np.ndarray
    elo: np.ndarray

    def as_matrix(self) -> np.ndarray:
        return np.column_stack([self.xgboost, self.logistic, self.elo])

    def agreement(self) -> np.ndarray:
        """1 - normalised spread across components (1.0 = full agreement)."""
        matrix = self.as_matrix()
        spread = matrix.max(axis=1) - matrix.min(axis=1)
        return np.clip(1.0 - spread / 0.25, 0.0, 1.0)


@dataclass
class NbaWinModel:
    """Fitted ensemble plus everything needed to reproduce and explain it."""

    features: list[str]
    xgb_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))
    calibration_method: str = settings.model.calibration_method
    model_version: str = MODEL_VERSION
    feature_set_version: str = "nba_v2.0"
    variant: str = "full"

    xgb_model: xgb.XGBClassifier | None = None
    logistic_model: Pipeline | None = None
    weights: np.ndarray | None = None
    calibrator: Calibrator | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reliability: list[dict[str, float]] = field(default_factory=list)
    training_rows: int = 0

    # ------------------------------------------------------------------- fit

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "NbaWinModel":
        X_train, y_train = self._matrix(train), train[TARGET].astype(int).to_numpy()
        X_valid, y_valid = self._matrix(valid), valid[TARGET].astype(int).to_numpy()
        self.training_rows = len(train)

        seed = settings.model.random_seed
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params, random_state=seed)
        self.xgb_model.fit(X_train, y_train, verbose=False)

        self.logistic_model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.15, max_iter=2000, random_state=seed)),
            ]
        )
        self.logistic_model.fit(X_train, y_train)

        components = self._components(valid)
        self.weights = self._fit_weights(components.as_matrix(), y_valid)

        blended_valid = components.as_matrix() @ self.weights
        self.calibrator = Calibrator(self.calibration_method).fit(blended_valid, y_valid)
        calibrated_valid = self.calibrator.transform(blended_valid)

        self.metrics = {
            "validation": evaluate_probabilities(y_valid, calibrated_valid).to_dict(),
            "validation_uncalibrated": evaluate_probabilities(y_valid, blended_valid).to_dict(),
            "components_validation": {
                "xgboost": evaluate_probabilities(y_valid, components.xgboost).to_dict(),
                "logistic": evaluate_probabilities(y_valid, components.logistic).to_dict(),
                "elo": evaluate_probabilities(y_valid, components.elo).to_dict(),
            },
            "blend_weights": {
                "xgboost": float(self.weights[0]),
                "logistic": float(self.weights[1]),
                "elo": float(self.weights[2]),
            },
            "n_train": len(train),
            "n_valid": len(valid),
        }
        self.reliability = reliability_curve(y_valid, calibrated_valid)
        log.info("fitted NBA model", extra={"features": len(self.features),
                                            "weights": self.metrics["blend_weights"],
                                            "valid_logloss": self.metrics["validation"]["log_loss"]})
        return self

    def evaluate(self, frame: pd.DataFrame) -> CalibrationMetrics:
        probabilities = self.predict_proba(frame)
        return evaluate_probabilities(frame[TARGET].astype(int).to_numpy(), probabilities)

    # --------------------------------------------------------------- predict

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        blended = self._components(frame).as_matrix() @ self.weights
        if self.calibrator is not None:
            return self.calibrator.transform(blended)
        return blended

    def predict_detail(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Calibrated probability plus per-component views and agreement."""
        components = self._components(frame)
        blended = components.as_matrix() @ self.weights
        calibrated = (self.calibrator.transform(blended)
                      if self.calibrator is not None else blended)
        return {
            "probability": calibrated,
            "uncalibrated": blended,
            "xgboost": components.xgboost,
            "logistic": components.logistic,
            "elo": components.elo,
            "agreement": components.agreement(),
        }

    def feature_importance(self, top: int = 15) -> list[dict[str, Any]]:
        if self.xgb_model is None:
            return []
        importances = self.xgb_model.feature_importances_
        ordered = sorted(zip(self.features, importances), key=lambda kv: -kv[1])[:top]
        return [{"feature": name, "importance": float(value)} for name, value in ordered]

    def explain(self, frame: pd.DataFrame, *, top: int = 6) -> list[list[dict[str, Any]]]:
        """Per-prediction contributions from the boosted trees.

        Uses XGBoost's exact SHAP values (``pred_contribs``), so the numbers add
        up to the model's own margin rather than being a plausible-looking
        approximation.
        """
        if self.xgb_model is None:
            return []
        matrix = xgb.DMatrix(self._matrix(frame), feature_names=self.features)
        contributions = self.xgb_model.get_booster().predict(matrix, pred_contribs=True)
        explanations: list[list[dict[str, Any]]] = []
        for row in contributions:
            pairs = sorted(
                zip(self.features, row[:-1]), key=lambda kv: -abs(kv[1])
            )[:top]
            explanations.append(
                [
                    {"feature": name, "contribution": float(value),
                     "direction": "home" if value > 0 else "away"}
                    for name, value in pairs
                ]
            )
        return explanations

    def calibration_quality(self) -> float:
        validation = self.metrics.get("validation")
        if not validation:
            return 0.3
        metrics = CalibrationMetrics(**{k: validation[k] for k in
                                        ("n", "brier", "log_loss", "ece", "accuracy",
                                         "base_rate", "brier_skill")})
        return calibration_quality(metrics)

    # -------------------------------------------------------------- internals

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise KeyError(f"feature frame is missing columns: {missing[:5]}")
        return frame[self.features].astype(float).to_numpy()

    def _components(self, frame: pd.DataFrame) -> ComponentPrediction:
        matrix = self._matrix(frame)
        xgb_probs = self.xgb_model.predict_proba(matrix)[:, 1]
        logistic_probs = self.logistic_model.predict_proba(matrix)[:, 1]
        if "elo_home_prob" in frame.columns:
            elo_probs = frame["elo_home_prob"].astype(float).fillna(0.5).to_numpy()
        else:
            elo_probs = np.full(len(frame), 0.5)
        return ComponentPrediction(xgb_probs, logistic_probs, np.clip(elo_probs, 0.01, 0.99))

    @staticmethod
    def _fit_weights(component_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Weights on the simplex that minimise validation log loss."""
        def objective(weights: np.ndarray) -> float:
            blended = np.clip(component_matrix @ weights, 1e-6, 1 - 1e-6)
            return float(-np.mean(y_true * np.log(blended) + (1 - y_true) * np.log(1 - blended)))

        n = component_matrix.shape[1]
        start = np.full(n, 1.0 / n)
        result = minimize(
            objective, start, method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 300, "ftol": 1e-9},
        )
        weights = result.x if result.success else start
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        return weights / total if total > 0 else start
