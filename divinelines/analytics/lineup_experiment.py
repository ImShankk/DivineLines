"""Do lineups actually improve the model?

This is the question V3 exists to answer, and the answer has to survive an
honest test rather than an intuition that lineups "obviously" matter.

## What can and cannot be measured

ESPN publishes the XI for a finished match but not *when* that XI became
public. So a historical experiment cannot reproduce the live situation ("we
learned the lineup roughly an hour before kick-off"). What it can do is
establish an **upper bound**: give the model the actual starting XI — strictly
more information than any live system could have — and see whether it helps at
all. If perfect lineup knowledge does not improve out-of-sample predictions,
then chasing lineups in production cannot be worth it, and no amount of
plumbing will change that.

Everything here is labelled ``oracle`` for that reason. It is a bound, not a
backtest.

## Design

Both arms are trained and scored on **identical rows** — the matches where
lineup features exist for both teams and enough history has accumulated for
"regular starter" to mean anything. The baseline arm gets the production
model's own probabilities; the lineup arm gets those plus the lineup features.
Any difference is therefore attributable to the lineup information and nothing
else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import settings
from ..db.repository import load_soccer_matches
from ..features.lineup_features import LINEUP_FEATURES, attach_lineup_features
from ..features.soccer_features import build_soccer_dataset
from ..logging_setup import get_logger
from ..models.calibration import multiclass_brier, multiclass_log_loss
from ..models.registry import record_experiment
from ..models.soccer_model import SoccerMatchModel, resolve_soccer_features

log = get_logger(__name__)

BASELINE_FEATURES = ("dc_prob_home", "dc_prob_draw", "dc_prob_away",
                     "elo_prob_home", "elo_prob_draw", "elo_prob_away")


@dataclass
class ArmResult:
    name: str
    features: list[str]
    n_train: int
    n_test: int
    log_loss: float
    brier: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.name, "n_features": len(self.features),
            "n_train": self.n_train, "n_test": self.n_test,
            "log_loss": round(self.log_loss, 5), "brier": round(self.brier, 5),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class LineupExperiment:
    coverage: dict[str, Any]
    arms: list[ArmResult] = field(default_factory=list)
    verdict: str = ""
    delta_log_loss: float | None = None
    available: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available, "reason": self.reason,
            "coverage": self.coverage,
            "arms": [arm.to_dict() for arm in self.arms],
            "delta_log_loss": (None if self.delta_log_loss is None
                               else round(self.delta_log_loss, 5)),
            "verdict": self.verdict,
            "caveat": (
                "ORACLE BOUND: the actual starting XI is used, which is more "
                "information than a live system could have, because the source "
                "publishes no lineup timestamps. Treat this as the ceiling on "
                "what lineup data could contribute, not as achievable performance."
            ),
        }


def _fit_arm(name: str, train: pd.DataFrame, test: pd.DataFrame,
             features: Sequence[str]) -> ArmResult:
    """Multinomial logistic on the given features.

    A linear model on top of the production probabilities is deliberate: with a
    few hundred usable matches, anything with more capacity would fit noise and
    the ablation would measure the classifier rather than the lineup data.
    """
    columns = [f for f in features if f in train.columns]
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=2000,
                                   random_state=settings.model.random_seed)),
    ])
    X_train = train[columns].astype(float).to_numpy()
    y_train = train["outcome"].astype(int).to_numpy()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(test[columns].astype(float).to_numpy())
    y_test = test["outcome"].astype(int).to_numpy()

    # Align columns to the full 3-class space in case a class is absent in train.
    if probabilities.shape[1] != 3:
        full = np.full((len(test), 3), 1e-6)
        for index, label in enumerate(pipeline.named_steps["clf"].classes_):
            full[:, int(label)] = probabilities[:, index]
        probabilities = full / full.sum(axis=1, keepdims=True)

    return ArmResult(
        name=name, features=columns, n_train=len(train), n_test=len(test),
        log_loss=multiclass_log_loss(y_test, probabilities),
        brier=multiclass_brier(y_test, probabilities),
        accuracy=float(np.mean(probabilities.argmax(axis=1) == y_test)),
    )


def run_lineup_experiment(league_ids: Sequence[str] = ("ENG_PL",),
                          *, test_fraction: float = 0.3,
                          min_rows: int = 150,
                          experiment_name: str = "soccer_lineup_oracle") -> LineupExperiment:
    """Baseline vs baseline+lineups on identical rows, split chronologically."""
    matches = load_soccer_matches(list(league_ids))
    if matches.empty:
        return LineupExperiment({}, available=False, reason="no soccer matches ingested")

    dataset, _ = build_soccer_dataset(matches)
    dataset = dataset[dataset["outcome"].notna()].copy()
    dataset["outcome"] = dataset["outcome"].astype(int)

    augmented, coverage = attach_lineup_features(dataset, allow_final=True)

    # The production model supplies the baseline probabilities so the two arms
    # differ only by the lineup columns.
    model = SoccerMatchModel(features=resolve_soccer_features("full", augmented.columns),
                             use_classifier=False)
    split_at = int(len(augmented) * 0.6)
    model.fit(augmented.iloc[:split_at], augmented.iloc[split_at:int(len(augmented) * 0.8)],
              raw_matches=matches)
    augmented = model.attach_dc_features(augmented)

    usable = augmented[augmented["diff_xi_regular_share"].notna()].copy()
    coverage["usable_for_experiment"] = int(len(usable))
    if len(usable) < min_rows:
        return LineupExperiment(
            coverage, available=False,
            reason=(f"only {len(usable)} matches have usable lineup features "
                    f"(need {min_rows}); ingest more history with "
                    f"`divinelines lineups --historical`"),
        )

    usable = usable.sort_values("game_date").reset_index(drop=True)
    cut = int(len(usable) * (1 - test_fraction))
    train, test = usable.iloc[:cut], usable.iloc[cut:]

    baseline = _fit_arm("baseline", train, test, BASELINE_FEATURES)
    with_lineups = _fit_arm("baseline + lineups", train, test,
                            list(BASELINE_FEATURES) + list(LINEUP_FEATURES))

    delta = with_lineups.log_loss - baseline.log_loss
    if delta < -0.005:
        verdict = ("Lineup features improved out-of-sample log loss. Worth promoting "
                   "into the champion model, subject to live validation.")
    elif delta > 0.005:
        verdict = ("Lineup features made out-of-sample log loss WORSE. Not promoted; "
                   "kept available for research only.")
    else:
        verdict = ("No measurable difference. Lineup features are not promoted — an "
                   "intuitively useful feature that does not survive testing stays out "
                   "of the champion model.")

    experiment = LineupExperiment(coverage=coverage, arms=[baseline, with_lineups],
                                  delta_log_loss=delta, verdict=verdict)

    for arm in experiment.arms:
        record_experiment(
            name=experiment_name, sport="soccer", variant=arm.name,
            feature_set=arm.features, model_kind="logistic",
            metrics=arm.to_dict(), n_train=arm.n_train, n_valid=arm.n_test,
            valid_range=f"{test['game_date'].min():%Y-%m-%d}..{test['game_date'].max():%Y-%m-%d}",
        )

    log.info("lineup experiment complete",
             extra={"delta_log_loss": round(delta, 5), "n_test": baseline.n_test,
                    "usable": len(usable)})
    return experiment
