"""Does the *model* generate CLV, or is it just line shopping?

The first replay of the 2023-24 NBA season produced +7.2% mean CLV over 2,456
predictions with a 69% beat-close rate. That looks like a strong edge and is
almost entirely an artefact:

* the ledger holds **both sides** of every game, so a model with no information
  at all appears in this sample exactly as often as a good one;
* entry prices are the **best of ~13 books** at the open while the close is a
  consensus, and the book offering the outlier opening price is by selection the
  one most likely to move back toward the pack.

A number that a zero-skill model would reproduce is not evidence. This module
separates the three effects that get mixed together:

1. **Shopping CLV** — taking the best of many books. Real money, no model.
2. **Market drift** — open-to-close movement of the consensus itself, which
   applies to every selection regardless of what we thought.
3. **Selection CLV** — do the selections the model *recommends* beat the close by
   more than the ones it rejects? This is the only one that measures the model.

The headline number the platform reports is (3), and it is reported as a
difference with a confidence interval, not as a raw average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..betting.clv import MIN_SAMPLE_FOR_INFERENCE
from ..config import settings
from ..db.connection import query_df
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class GroupStats:
    label: str
    n: int
    mean: float
    median: float
    positive_rate: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "n": self.n, "mean": round(self.mean, 4),
                "median": round(self.median, 4),
                "positive_rate": round(self.positive_rate, 4), "std": round(self.std, 4)}


@dataclass
class SkillTest:
    """Recommended vs rejected selections, on identical prices."""

    recommended: GroupStats
    rejected: GroupStats
    difference: float
    ci_low: float | None
    ci_high: float | None
    significant: bool
    interpretation: str
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended": self.recommended.to_dict(),
            "rejected": self.rejected.to_dict(),
            "difference": round(self.difference, 4),
            "ci_low": None if self.ci_low is None else round(self.ci_low, 4),
            "ci_high": None if self.ci_high is None else round(self.ci_high, 4),
            "significant": self.significant,
            "interpretation": self.interpretation,
            "basis": self.basis,
        }


def _consensus_prices(sport: str, market: str, phase: str) -> pd.DataFrame:
    """Median price per (game, selection) for one market phase."""
    frame = query_df(
        """
        SELECT game_uid, selection, bookmaker, price_decimal
        FROM odds_snapshots
        WHERE sport = ? AND market = ? AND phase = ?
        """,
        (sport, market, phase),
    )
    if frame.empty:
        return frame
    grouped = (frame.groupby(["game_uid", "selection"])["price_decimal"]
               .median().reset_index()
               .rename(columns={"price_decimal": f"{phase}_consensus"}))
    return grouped


def build_skill_frame(sport: str = "nba", market: str = "h2h",
                      mode: str = "backtest") -> pd.DataFrame:
    """Predictions joined to consensus open and close prices.

    Using the consensus on *both* ends removes the shopping effect entirely, so
    what remains is the movement of the market itself plus whatever the model's
    choice of side adds.
    """
    predictions = query_df(
        """
        SELECT p.prediction_id, p.game_uid, p.selection, p.model_prob, p.market_prob,
               p.edge, p.price_decimal AS entry_price, p.ev_per_unit, p.stake,
               p.model_version, p.sport, p.market, c.clv_price_pct, c.clv_same_book_pct,
               c.result
        FROM predictions p
        LEFT JOIN clv_records c ON c.prediction_id = p.prediction_id
        WHERE p.sport = ? AND p.market = ? AND p.mode = ?
        """,
        (sport, market, mode),
    )
    if predictions.empty:
        return predictions

    opens = _consensus_prices(sport, market, "open")
    closes = _consensus_prices(sport, market, "close")
    if opens.empty or closes.empty:
        return pd.DataFrame()

    merged = (predictions
              .merge(opens, on=["game_uid", "selection"], how="inner")
              .merge(closes, on=["game_uid", "selection"], how="inner"))
    if merged.empty:
        return merged

    # Consensus-to-consensus CLV: no book selection on either side.
    merged["clv_consensus_pct"] = (
        merged["open_consensus"] / merged["close_consensus"] - 1.0
    ) * 100.0
    merged["recommended"] = (
        (merged["edge"].fillna(-1) >= settings.betting.min_edge)
        & (merged["ev_per_unit"].fillna(-1) > 0)
    )
    return merged


def _stats(label: str, values: Sequence[float]) -> GroupStats:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return GroupStats(label, 0, 0.0, 0.0, 0.0, 0.0)
    return GroupStats(
        label=label, n=int(array.size), mean=float(array.mean()),
        median=float(np.median(array)),
        positive_rate=float((array > 0).mean()),
        std=float(array.std(ddof=1)) if array.size > 1 else 0.0,
    )


def skill_test(frame: pd.DataFrame, column: str, basis: str) -> SkillTest:
    """Welch's t-test on the difference in mean CLV between groups."""
    recommended = frame[frame["recommended"]][column].dropna()
    rejected = frame[~frame["recommended"]][column].dropna()

    stats_recommended = _stats("recommended", recommended)
    stats_rejected = _stats("rejected", rejected)
    difference = stats_recommended.mean - stats_rejected.mean

    ci_low = ci_high = None
    significant = False
    if (stats_recommended.n >= MIN_SAMPLE_FOR_INFERENCE
            and stats_rejected.n >= MIN_SAMPLE_FOR_INFERENCE):
        # Welch: the two groups have no reason to share a variance, and the
        # recommended group is usually much the smaller of the two.
        standard_error = math.sqrt(
            stats_recommended.std ** 2 / stats_recommended.n
            + stats_rejected.std ** 2 / stats_rejected.n
        )
        if standard_error > 0:
            margin = 1.959963985 * standard_error
            ci_low, ci_high = difference - margin, difference + margin
            significant = ci_low > 0 or ci_high < 0

    if stats_recommended.n < MIN_SAMPLE_FOR_INFERENCE:
        interpretation = (f"insufficient recommended sample "
                          f"(n={stats_recommended.n}, need {MIN_SAMPLE_FOR_INFERENCE})")
    elif not significant:
        interpretation = ("no measurable difference between recommended and rejected "
                          "selections — consistent with the model adding no information "
                          "beyond the market")
    elif difference > 0:
        interpretation = ("recommended selections beat the close by more than rejected "
                          "ones — evidence the model's choice of side carries information")
    else:
        interpretation = ("recommended selections beat the close by LESS than rejected "
                          "ones — the model's choice of side is actively unhelpful")

    return SkillTest(stats_recommended, stats_rejected, difference,
                     ci_low, ci_high, significant, interpretation, basis)


def decompose_clv(sport: str = "nba", market: str = "h2h",
                  mode: str = "backtest") -> dict[str, Any]:
    """Split observed CLV into shopping, market drift and model selection."""
    frame = build_skill_frame(sport, market, mode)
    if frame.empty:
        return {"available": False,
                "reason": "no replayed predictions joined to open and close consensus prices"}

    shopping = float((frame["entry_price"] / frame["open_consensus"] - 1.0).mean() * 100.0)
    drift = float(frame["clv_consensus_pct"].mean())
    reported = float(frame["clv_price_pct"].dropna().mean()) if frame["clv_price_pct"].notna().any() else None

    tests = {
        "consensus_to_consensus": skill_test(
            frame, "clv_consensus_pct",
            "consensus open vs consensus close — no book selection on either side",
        ).to_dict(),
    }
    if frame["clv_same_book_pct"].notna().any():
        tests["same_book"] = skill_test(
            frame, "clv_same_book_pct",
            "entry price vs the same book's close",
        ).to_dict()

    both_sides = _both_sides_control(frame)

    return {
        "available": True,
        "sport": sport, "market": market, "mode": mode, "n": int(len(frame)),
        "components": {
            "reported_clv_mean_pct": None if reported is None else round(reported, 4),
            "line_shopping_pct": round(shopping, 4),
            "market_drift_pct": round(drift, 4),
            "note": ("Reported CLV mixes three things. Line shopping is the gain from "
                     "taking the best of many books; market drift is the consensus "
                     "moving between open and close and applies to every selection; "
                     "only the skill test below measures the model."),
        },
        "skill_tests": tests,
        "both_sides_control": both_sides,
    }


def _both_sides_control(frame: pd.DataFrame) -> dict[str, Any]:
    """Sanity control: CLV averaged over both sides of the same game.

    Both sides of a two-way market cannot both beat the close on a fair
    comparison, so if this is strongly positive the measurement is picking up
    the price basis rather than any opinion about the game.
    """
    paired = frame.groupby("game_uid")["clv_consensus_pct"].agg(["mean", "count"])
    complete = paired[paired["count"] >= 2]
    if complete.empty:
        return {"n_games": 0, "note": "no games with both selections priced"}
    values = complete["mean"].to_numpy()
    return {
        "n_games": int(len(complete)),
        "mean_of_both_sides_pct": round(float(values.mean()), 4),
        "note": ("Averaging both sides of a game should sit near zero once the "
                 "price basis is consistent; a large value means the comparison "
                 "itself is biased, not that the model is winning."),
    }
