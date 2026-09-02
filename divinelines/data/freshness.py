"""Data freshness and provenance.

Cached data must never be trusted indefinitely.  Odds go stale in minutes,
injuries in hours, standings after each game.  This module turns those rules
into a single API used by the pipeline, the API layer and the dashboard, so a
number displayed to a user always carries how old it is.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from ..config import FRESHNESS_TTL
from ..db.connection import query_df
from ..logging_setup import get_logger

log = get_logger(__name__)

STATE_FRESH = "fresh"
STATE_AGING = "aging"
STATE_STALE = "stale"
STATE_MISSING = "missing"


def _parse_ts(value: Any) -> datetime | None:
    if value in (None, "", "None"):
        return None
    try:
        stamp = pd.to_datetime(value, utc=True, format="mixed")
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.to_pydatetime()


@dataclass
class Freshness:
    """How old a dataset is, and whether that is acceptable."""

    dataset: str
    kind: str                 # key into FRESHNESS_TTL
    source: str | None
    last_success: datetime | None
    ttl_seconds: int
    age_seconds: float | None
    state: str
    message: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.state in (STATE_FRESH, STATE_AGING)

    @property
    def age_minutes(self) -> float | None:
        return None if self.age_seconds is None else round(self.age_seconds / 60.0, 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["last_success"] = self.last_success.isoformat() if self.last_success else None
        payload["age_minutes"] = self.age_minutes
        payload["is_usable"] = self.is_usable
        return payload


def classify(age_seconds: float | None, ttl_seconds: int) -> str:
    if age_seconds is None:
        return STATE_MISSING
    if age_seconds <= ttl_seconds:
        return STATE_FRESH
    if age_seconds <= ttl_seconds * 3:
        return STATE_AGING
    return STATE_STALE


def assess(kind: str, *, source: str | None = None, dataset: str | None = None,
           now: datetime | None = None) -> Freshness:
    """Freshness of a dataset, read from the recorded source status."""
    ttl = FRESHNESS_TTL.get(kind, 3600)
    now = now or datetime.now(timezone.utc)

    clauses, params = [], []
    if source:
        clauses.append("source = ?")
        params.append(source)
    if dataset:
        clauses.append("dataset LIKE ?")
        params.append(f"{dataset}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = query_df(
        f"SELECT source, dataset, last_success, status, message FROM source_status {where} "
        "ORDER BY last_success DESC",
        params,
    )
    if rows.empty:
        return Freshness(dataset or kind, kind, source, None, ttl, None, STATE_MISSING,
                         "no successful fetch recorded")

    top = rows.iloc[0]
    last_success = _parse_ts(top["last_success"])
    age = (now - last_success).total_seconds() if last_success else None
    return Freshness(
        dataset=str(top["dataset"]), kind=kind, source=str(top["source"]),
        last_success=last_success, ttl_seconds=ttl, age_seconds=age,
        state=classify(age, ttl), message=top["message"],
    )


def freshness_report(specs: Iterable[tuple[str, str | None, str | None]] | None = None
                     ) -> list[Freshness]:
    """Freshness across the datasets the platform depends on."""
    specs = specs or (
        ("odds", "odds_api", None),
        ("injuries", "espn_nba", "injuries"),
        ("schedule", "espn_nba", "scoreboard"),
        ("box_scores", "nba_stats", "game_logs"),
        ("soccer_results", "football_data_uk", None),
    )
    return [assess(kind, source=source, dataset=dataset) for kind, source, dataset in specs]


# --------------------------------------------------------------------------
# Prediction-level data quality
# --------------------------------------------------------------------------

@dataclass
class QualityComponent:
    name: str
    score: float          # 0..1
    weight: float
    note: str | None = None


@dataclass
class DataQuality:
    """A transparent 0-100 score: every component and weight is inspectable."""

    score: float
    components: list[QualityComponent]

    @property
    def grade(self) -> str:
        if self.score >= 85:
            return "high"
        if self.score >= 65:
            return "medium"
        return "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "grade": self.grade,
            "components": [
                {"name": c.name, "score": round(c.score, 3), "weight": c.weight, "note": c.note}
                for c in self.components
            ],
        }


def freshness_score(freshness: Freshness | None) -> float:
    """Map a freshness state onto a 0-1 quality contribution."""
    if freshness is None or freshness.state == STATE_MISSING:
        return 0.0
    if freshness.state == STATE_FRESH:
        return 1.0
    if freshness.state == STATE_AGING:
        return 0.6
    return 0.2


def compute_data_quality(components: list[QualityComponent]) -> DataQuality:
    total_weight = sum(c.weight for c in components) or 1.0
    score = sum(max(0.0, min(1.0, c.score)) * c.weight for c in components) / total_weight
    return DataQuality(score=score * 100.0, components=components)
