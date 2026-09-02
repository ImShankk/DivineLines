"""Feature ablation and experiment tracking.

A feature earns its place by improving out-of-sample performance, not by
sounding sophisticated.  This runs the same time-aware evaluation across
progressively larger feature sets and records every result, so the question
"did adding schedule/fatigue features actually help?" has an answer in the
database rather than an opinion.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import settings
from ..logging_setup import get_logger
from .calibration import evaluate_probabilities, multiclass_brier, multiclass_log_loss
from .registry import record_experiment

log = get_logger(__name__)


def run_nba_ablation(dataset: pd.DataFrame | None = None,
                     seasons: Sequence[str] | None = None,
                     *, experiment_name: str = "nba_feature_ablation") -> pd.DataFrame:
    """Evaluate each NBA feature variant with identical walk-forward folds."""
    from ..backtest.nba_backtest import prepare_nba_dataset, run_nba_backtest
    from .nba_model import FEATURE_VARIANTS

    dataset = prepare_nba_dataset() if dataset is None else dataset
    dataset = dataset[dataset["home_win"].notna()].copy()
    if seasons is None:
        available = sorted(dataset["season"].unique())
        seasons = available[-2:] if len(available) > 2 else available

    rows: list[dict[str, Any]] = []
    for variant in FEATURE_VARIANTS:
        try:
            result = run_nba_backtest(dataset, variant=variant, seasons=list(seasons))
        except RuntimeError as exc:
            log.warning("ablation variant failed", extra={"variant": variant, "error": str(exc)})
            continue
        metrics = result.probability_metrics["ensemble"]
        rows.append(
            {
                "variant": variant,
                "n": metrics["n"],
                "log_loss": round(metrics["log_loss"], 5),
                "brier": round(metrics["brier"], 5),
                "accuracy": round(metrics["accuracy"], 4),
                "ece": round(metrics["ece"], 5),
                "brier_skill": round(metrics["brier_skill"], 5),
            }
        )
        record_experiment(
            name=experiment_name, sport="nba", variant=variant,
            feature_set=result.predictions.attrs.get("features", []),
            model_kind="ensemble", metrics=metrics,
            valid_range=",".join(map(str, seasons)), n_valid=metrics["n"],
        )
        log.info("ablation variant complete", extra=rows[-1])

    frame = pd.DataFrame(rows).sort_values("log_loss")
    if not frame.empty:
        best = frame.iloc[0]
        frame["delta_vs_best"] = (frame["log_loss"] - best["log_loss"]).round(5)
    return frame


def run_soccer_ablation(league_ids: Sequence[str] | None = None,
                        seasons: Sequence[str] | None = None,
                        *, experiment_name: str = "soccer_feature_ablation") -> pd.DataFrame:
    from ..backtest.soccer_backtest import run_soccer_backtest
    from .soccer_model import SOCCER_VARIANTS

    rows: list[dict[str, Any]] = []
    for variant in SOCCER_VARIANTS:
        try:
            result = run_soccer_backtest(league_ids, variant=variant, seasons=seasons)
        except RuntimeError as exc:
            log.warning("ablation variant failed", extra={"variant": variant, "error": str(exc)})
            continue
        model_metrics = result.probability_metrics["model"]
        market = result.probability_metrics.get("market_novig") or {}
        betting = result.metrics.to_dict()
        rows.append(
            {
                "variant": variant,
                "n": model_metrics["n"],
                "log_loss": round(model_metrics["log_loss"], 5),
                "brier": round(model_metrics["brier"], 5),
                "accuracy": round(model_metrics["accuracy"], 4),
                "market_log_loss": round(market.get("log_loss", float("nan")), 5),
                "beats_market": bool(model_metrics["log_loss"] < market.get("log_loss", 0)),
                "bets": betting["bets"],
                "roi": round(betting["roi"], 4),
            }
        )
        record_experiment(
            name=experiment_name, sport="soccer", variant=variant,
            feature_set=[], model_kind="ensemble",
            metrics={**model_metrics, "roi": betting["roi"], "bets": betting["bets"]},
            valid_range=",".join(map(str, seasons or [])), n_valid=model_metrics["n"],
        )
        log.info("ablation variant complete", extra=rows[-1])

    return pd.DataFrame(rows).sort_values("log_loss") if rows else pd.DataFrame()


def run_ablation(sport: str, seasons: Sequence[str] | None = None) -> pd.DataFrame:
    if sport == "nba":
        return run_nba_ablation(seasons=seasons)
    return run_soccer_ablation(seasons=seasons)
