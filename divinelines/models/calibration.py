"""Probability calibration and calibration diagnostics.

Accuracy is close to worthless for a betting model: a model that is 55%
accurate but systematically says 70% when it means 60% will lose money on
every one of those bets.  What matters is that a stated 70% wins about 70% of
the time.

Calibrators are always fitted on a *later* block than the model itself and
applied to a still-later block, so calibration never sees its own evaluation
data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..logging_setup import get_logger

log = get_logger(__name__)

EPSILON = 1e-9


# --------------------------------------------------------------------- metrics

def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPSILON, 1 - EPSILON)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def multiclass_log_loss(y_true: Sequence[int], y_prob: np.ndarray) -> float:
    """Log loss for a K-class problem with integer class labels."""
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPSILON, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    indices = np.asarray(y_true, dtype=int)
    return float(-np.mean(np.log(y_prob[np.arange(len(indices)), indices])))


def multiclass_brier(y_true: Sequence[int], y_prob: np.ndarray) -> float:
    y_prob = np.asarray(y_prob, dtype=float)
    onehot = np.zeros_like(y_prob)
    onehot[np.arange(len(y_true)), np.asarray(y_true, dtype=int)] = 1.0
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: Sequence[int], y_prob: Sequence[float],
                               bins: int = 10) -> float:
    """Weighted mean gap between stated confidence and realised frequency."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob > low) & (y_prob <= high)
        if not mask.any():
            continue
        total += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(total)


def reliability_curve(y_true: Sequence[int], y_prob: Sequence[float],
                      bins: int = 10) -> list[dict[str, float]]:
    """Points for a reliability diagram, with bin counts so noise is visible."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    points: list[dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob > low) & (y_prob <= high)
        if not mask.any():
            continue
        points.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "predicted": float(y_prob[mask].mean()),
                "observed": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return points


@dataclass
class CalibrationMetrics:
    n: int
    brier: float
    log_loss: float
    ece: float
    accuracy: float
    base_rate: float
    #: Brier skill versus always predicting the base rate.  Positive means the
    #: model adds information; negative means it is worse than a constant.
    brier_skill: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in payload.items()}


def evaluate_probabilities(y_true: Sequence[int], y_prob: Sequence[float]) -> CalibrationMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    reference = brier_score(y_true, np.full_like(y_prob, base_rate)) if len(y_true) else 0.0
    model_brier = brier_score(y_true, y_prob) if len(y_true) else 0.0
    return CalibrationMetrics(
        n=int(len(y_true)),
        brier=model_brier,
        log_loss=log_loss_score(y_true, y_prob) if len(y_true) else 0.0,
        ece=expected_calibration_error(y_true, y_prob) if len(y_true) else 0.0,
        accuracy=float(((y_prob >= 0.5).astype(float) == y_true).mean()) if len(y_true) else 0.0,
        base_rate=base_rate,
        brier_skill=float(1.0 - model_brier / reference) if reference > 0 else 0.0,
    )


# ------------------------------------------------------------------ calibrators

class Calibrator:
    """Maps raw model scores to calibrated probabilities."""

    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "platt", "none"):
            raise ValueError(f"unknown calibration method '{method}'")
        self.method = method
        self._model: Any = None
        self.fitted = False
        self.n_fit = 0

    def fit(self, y_prob: Sequence[float], y_true: Sequence[int]) -> "Calibrator":
        y_prob = np.asarray(y_prob, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        self.n_fit = len(y_true)

        if self.method == "none" or self.n_fit < 50 or len(np.unique(y_true)) < 2:
            if self.method != "none":
                log.warning("insufficient data to calibrate; passing probabilities through",
                            extra={"n": self.n_fit, "method": self.method})
            self.method = "none"
            self.fitted = True
            return self

        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            self._model.fit(y_prob, y_true)
        else:
            self._model = LogisticRegression(C=1e6, solver="lbfgs")
            self._model.fit(_logit(y_prob).reshape(-1, 1), y_true)
        self.fitted = True
        return self

    def transform(self, y_prob: Sequence[float]) -> np.ndarray:
        y_prob = np.asarray(y_prob, dtype=float)
        if not self.fitted:
            raise RuntimeError("Calibrator.transform called before fit")
        if self.method == "none":
            return y_prob
        if self.method == "isotonic":
            return np.clip(self._model.predict(y_prob), 0.005, 0.995)
        return np.clip(self._model.predict_proba(_logit(y_prob).reshape(-1, 1))[:, 1],
                       0.005, 0.995)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "n_fit": self.n_fit, "fitted": self.fitted}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def calibration_quality(metrics: CalibrationMetrics, *, ece_reference: float = 0.05) -> float:
    """Map calibration error onto a 0-1 quality score for the edge score."""
    if metrics.n < 50:
        return 0.3
    return float(np.clip(1.0 - metrics.ece / ece_reference, 0.0, 1.0))
