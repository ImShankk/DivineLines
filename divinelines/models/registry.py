"""Model registry.

Every trained artifact is recorded with the data it saw, the features it used,
its hyperparameters, its validation metrics and a seed, so any result in the
platform can be reproduced or attributed to a specific model version.

The registry also makes "did the change actually help?" answerable: predictions
carry ``model_id``, so realised performance can be sliced by model version.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..config import settings
from ..db.connection import query_df, upsert_rows
from ..db.repository import now_iso
from ..logging_setup import get_logger

log = get_logger(__name__)


def data_version_hash(frames: dict[str, pd.DataFrame] | None = None) -> str:
    """A short, deterministic fingerprint of the data a model was trained on."""
    if frames is None:
        counts = query_df(
            "SELECT sport, COUNT(*) AS n, MAX(game_date) AS last_date, "
            "MAX(retrieved_at) AS last_retrieved FROM games GROUP BY sport"
        )
        payload = counts.to_json(orient="records")
    else:
        payload = json.dumps(
            {
                name: {
                    "rows": int(len(frame)),
                    "columns": sorted(map(str, frame.columns)),
                    "first": str(frame.iloc[0].to_dict()) if len(frame) else "",
                    "last": str(frame.iloc[-1].to_dict()) if len(frame) else "",
                }
                for name, frame in frames.items()
            },
            sort_keys=True,
        )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ModelRecord:
    model_id: str
    sport: str
    kind: str
    model_version: str
    feature_set: list[str]
    feature_set_version: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    league_id: str | None = None
    train_start: str | None = None
    train_end: str | None = None
    valid_start: str | None = None
    valid_end: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    data_version: str | None = None
    random_seed: int = settings.model.random_seed
    artifact_path: str | None = None
    notes: str | None = None
    trained_at: str = field(default_factory=now_iso)

    def to_row(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "sport": self.sport, "league_id": self.league_id,
            "kind": self.kind, "model_version": self.model_version,
            "feature_set": json.dumps(self.feature_set),
            "feature_set_version": self.feature_set_version,
            "hyperparameters": json.dumps(self.hyperparameters, default=str),
            "train_start": self.train_start, "train_end": self.train_end,
            "valid_start": self.valid_start, "valid_end": self.valid_end,
            "metrics": json.dumps(self.metrics, default=str),
            "data_version": self.data_version, "random_seed": self.random_seed,
            "artifact_path": self.artifact_path, "trained_at": self.trained_at,
            "notes": self.notes,
        }


def make_model_id(sport: str, kind: str, version: str, *, league_id: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    parts = [sport, league_id or "all", kind, version, stamp]
    return "-".join(str(p) for p in parts if p)


def save_artifact(model_id: str, payload: Any) -> Path:
    settings.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = settings.paths.artifacts_dir / f"{model_id}.joblib"
    joblib.dump(payload, path)
    return path


def load_artifact(model_id: str) -> Any:
    path = settings.paths.artifacts_dir / f"{model_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no artifact for model '{model_id}' at {path}")
    return joblib.load(path)


def register(record: ModelRecord, payload: Any | None = None) -> ModelRecord:
    if payload is not None:
        record.artifact_path = str(save_artifact(record.model_id, payload))
    upsert_rows("models", [record.to_row()], conflict_columns=["model_id"])
    log.info("registered model", extra={"model_id": record.model_id, "kind": record.kind,
                                        "metrics": record.metrics})
    return record


def list_models(sport: str | None = None, kind: str | None = None,
                limit: int = 50) -> pd.DataFrame:
    clauses, params = [], []
    if sport:
        clauses.append("sport = ?")
        params.append(sport)
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return query_df(
        f"SELECT * FROM models {where} ORDER BY trained_at DESC LIMIT {int(limit)}", params
    )


def latest_model_id(sport: str, kind: str = "ensemble", league_id: str | None = None
                    ) -> str | None:
    clauses = ["sport = ?", "kind = ?"]
    params: list[Any] = [sport, kind]
    if league_id:
        clauses.append("league_id = ?")
        params.append(league_id)
    row = query_df(
        f"SELECT model_id FROM models WHERE {' AND '.join(clauses)} "
        "ORDER BY trained_at DESC LIMIT 1",
        params,
    )
    return None if row.empty else str(row["model_id"].iloc[0])


def load_latest(sport: str, kind: str = "ensemble", league_id: str | None = None) -> Any | None:
    model_id = latest_model_id(sport, kind, league_id)
    if not model_id:
        return None
    try:
        return load_artifact(model_id)
    except FileNotFoundError:
        log.warning("registry references a missing artifact", extra={"model_id": model_id})
        return None


def get_metrics(model_id: str) -> dict[str, Any]:
    row = query_df("SELECT metrics FROM models WHERE model_id = ?", (model_id,))
    if row.empty:
        return {}
    try:
        return json.loads(row["metrics"].iloc[0] or "{}")
    except json.JSONDecodeError:
        return {}


def record_experiment(
    *, name: str, sport: str, variant: str, feature_set: list[str], model_kind: str,
    metrics: dict[str, Any], hyperparameters: dict[str, Any] | None = None,
    train_range: str | None = None, valid_range: str | None = None,
    n_train: int | None = None, n_valid: int | None = None,
    random_seed: int | None = None,
) -> None:
    upsert_rows(
        "experiments",
        [
            {
                "name": name, "sport": sport, "variant": variant,
                "feature_set": json.dumps(feature_set),
                "model_kind": model_kind,
                "hyperparameters": json.dumps(hyperparameters or {}, default=str),
                "train_range": train_range, "valid_range": valid_range,
                "metrics": json.dumps(metrics, default=str),
                "n_train": n_train, "n_valid": n_valid,
                "random_seed": random_seed if random_seed is not None else settings.model.random_seed,
                "created_at": now_iso(),
            }
        ],
    )


def experiment_results(name: str | None = None) -> pd.DataFrame:
    clause = "WHERE name = ?" if name else ""
    params = [name] if name else []
    return query_df(
        f"SELECT * FROM experiments {clause} ORDER BY created_at DESC, experiment_id DESC",
        params,
    )
