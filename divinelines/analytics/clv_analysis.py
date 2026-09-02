"""CLV cohort analysis.

One global CLV number hides everything worth knowing. The interesting question
is never "is our CLV positive" but "*where* is it positive" — which sport, which
market, which model version, and above all **how long before the event** the
prediction was made. If the platform only beats the close inside the last hour,
that says something quite different from beating it three days out.

Every cohort carries its own sample size and interval, and cohorts too small to
support inference say so rather than reporting a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..betting.clv import MIN_SAMPLE_FOR_INFERENCE, summarise_clv
from ..db.connection import query_df
from ..logging_setup import get_logger

log = get_logger(__name__)

#: Time-to-event cohorts. The boundaries follow how information actually
#: arrives in these sports: team news lands the day before, lineups roughly an
#: hour out, and the sharpest money arrives in the final minutes.
TIME_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("24h+", 86400, float("inf")),
    ("6-24h", 21600, 86400),
    ("1-6h", 3600, 21600),
    ("30-60m", 1800, 3600),
    ("<30m", 0, 1800),
)

CLV_BASES = {
    "consensus": "clv_price_pct",
    "same_book": "clv_same_book_pct",
}


def load_clv_frame(sport: str | None = None, *, basis: str = "consensus",
                   since: str | None = None) -> pd.DataFrame:
    column = CLV_BASES.get(basis, "clv_price_pct")
    clauses = [f"c.{column} IS NOT NULL"]
    params: list[Any] = []
    if sport:
        clauses.append("c.sport = ?")
        params.append(sport)
    if since:
        clauses.append("c.entry_timestamp >= ?")
        params.append(since)

    frame = query_df(
        f"""
        SELECT c.*, g.game_date, g.league_id AS game_league
        FROM clv_records c
        LEFT JOIN games g ON g.game_uid = c.game_uid
        WHERE {' AND '.join(clauses)}
        ORDER BY c.entry_timestamp
        """,
        params,
    )
    if not frame.empty:
        frame["clv"] = frame[column]
    return frame


def _bucket_time_to_event(seconds: Any) -> str:
    if seconds is None or pd.isna(seconds):
        return "unknown"
    value = float(seconds)
    for label, low, high in TIME_BUCKETS:
        if low <= value < high:
            return label
    return "unknown"


def _bucket_numeric(value: Any, edges: Sequence[float], labels: Sequence[str]) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return str(pd.cut([float(value)], list(edges), labels=list(labels))[0])


def add_cohort_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["time_to_event"] = out["seconds_to_event"].map(_bucket_time_to_event)
    out["edge_bucket"] = out["edge"].map(
        lambda v: _bucket_numeric(v, [-1, 0.02, 0.04, 0.06, 0.10, 1.0],
                                  ["<2%", "2-4%", "4-6%", "6-10%", "10%+"])
    )
    out["odds_bucket"] = out["entry_odds"].map(
        lambda v: _bucket_numeric(v, [1.0, 1.5, 2.0, 3.0, 5.0, 1000.0],
                                  ["heavy fav", "fav", "even", "dog", "longshot"])
    )
    out["probability_bucket"] = out["model_probability"].map(
        lambda v: _bucket_numeric(v, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                  ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    )
    out["quality_bucket"] = out["data_quality"].map(
        lambda v: _bucket_numeric(v, [0, 60, 80, 90, 100],
                                  ["<60", "60-80", "80-90", "90+"])
    )
    out["lineup_cohort"] = out["lineup_state"].fillna("unknown")
    return out


COHORT_DIMENSIONS: tuple[str, ...] = (
    "sport", "league_id", "market", "closing_book", "closing_source", "model_version",
    "time_to_event", "edge_bucket", "odds_bucket", "probability_bucket",
    "quality_bucket", "lineup_cohort", "prediction_stage",
)


def cohort_table(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """CLV by one dimension, with sample-size-aware interpretation."""
    if frame.empty or dimension not in frame.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(dimension, dropna=False):
        summary = summarise_clv(group["clv"].tolist())
        rows.append({
            "cohort": "unknown" if pd.isna(value) else str(value),
            "n": summary.n,
            "mean_clv": round(summary.mean_clv_price_pct, 3),
            "median_clv": round(summary.median_clv_price_pct, 3),
            "positive_rate": round(summary.beat_close_rate, 4),
            "std": round(summary.std_clv_price_pct, 3),
            "ci_low": None if summary.ci_low is None else round(summary.ci_low, 3),
            "ci_high": None if summary.ci_high is None else round(summary.ci_high, 3),
            "significant": summary.significant,
            "interpretation": summary.interpretation,
            "settled": int(group["result"].notna().sum()) if "result" in group else 0,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def clv_report(sport: str | None = None, *, basis: str = "consensus") -> dict[str, Any]:
    """Everything the CLV dashboard needs, in one pass over the ledger."""
    frame = add_cohort_columns(load_clv_frame(sport, basis=basis))
    overall = summarise_clv(frame["clv"].tolist() if not frame.empty else [])

    cohorts: dict[str, list[dict[str, Any]]] = {}
    for dimension in COHORT_DIMENSIONS:
        table = cohort_table(frame, dimension)
        if not table.empty:
            cohorts[dimension] = table.to_dict("records")

    cumulative: list[dict[str, Any]] = []
    if not frame.empty:
        running = frame.sort_values("entry_timestamp").copy()
        running["cumulative_mean"] = running["clv"].expanding().mean()
        cumulative = [
            {
                "entry_timestamp": str(row["entry_timestamp"]),
                "clv": round(float(row["clv"]), 4),
                "cumulative_mean": round(float(row["cumulative_mean"]), 4),
                "n": index + 1,
            }
            for index, (_, row) in enumerate(running.iterrows())
        ]

    distribution: list[dict[str, Any]] = []
    if not frame.empty:
        edges = [-100, -10, -5, -2, 0, 2, 5, 10, 100]
        labels = ["<-10%", "-10..-5%", "-5..-2%", "-2..0%", "0..2%", "2..5%", "5..10%", ">10%"]
        binned = pd.cut(frame["clv"], edges, labels=labels)
        counts = binned.value_counts().reindex(labels, fill_value=0)
        distribution = [{"bucket": label, "count": int(counts[label])} for label in labels]

    return {
        "basis": basis,
        "basis_description": (
            "entry price (best available) vs consensus close"
            if basis == "consensus"
            else "entry price vs the same bookmaker's close — isolates the model "
                 "from line shopping"
        ),
        "overall": overall.to_dict(),
        "sufficient_sample": overall.n >= MIN_SAMPLE_FOR_INFERENCE,
        "min_sample_for_inference": MIN_SAMPLE_FOR_INFERENCE,
        "cohorts": cohorts,
        "cumulative": cumulative,
        "distribution": distribution,
        "sample_size": overall.n,
    }


def clv_vs_profit(sport: str | None = None) -> dict[str, Any]:
    """CLV and ROI side by side, deliberately never merged into one number.

    A bet can lose while beating the close and win while losing to it. Keeping
    them adjacent — and labelled — is the whole point.
    """
    frame = load_clv_frame(sport)
    if frame.empty:
        return {"clv": None, "roi": None, "note": "no settled CLV records yet"}

    from .model_health import roi_interval

    graded = frame[frame["result"].notna()]
    staked = float(graded["stake"].fillna(0).sum()) if not graded.empty else 0.0
    profit = float(graded["profit"].fillna(0).sum()) if not graded.empty else 0.0
    summary = summarise_clv(frame["clv"].tolist())

    # The interval travels with the ROI so the dashboard cannot render a
    # non-significant return as a green edge.
    interval = roi_interval(
        graded["stake"].fillna(0).tolist() if not graded.empty else [],
        graded["profit"].fillna(0).tolist() if not graded.empty else [],
    )

    return {
        "clv": summary.to_dict(),
        "roi": round(profit / staked, 4) if staked > 0 else None,
        "roi_interval": interval,
        "profit": round(profit, 2),
        "staked": round(staked, 2),
        "settled_bets": int(len(graded)),
        "clv_sample": summary.n,
        "note": ("CLV and ROI measure different things: CLV is a market-efficiency "
                 "diagnostic, ROI is realised money. Neither implies the other."),
    }
