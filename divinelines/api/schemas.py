"""Pydantic schemas for the public API.

Response shapes are explicit so the frontend has a contract rather than
whatever a dict happened to contain that day, and so validation errors are
caught at the boundary instead of surfacing as `undefined` in the UI.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Sport = Literal["nba", "soccer"]


class LegacyMatchup(BaseModel):
    """Backwards-compatible request body for the original ``/api/predict``."""

    home: str = Field(..., min_length=2, max_length=40)
    away: str = Field(..., min_length=2, max_length=40)

    @field_validator("home", "away")
    @classmethod
    def strip_upper(cls, value: str) -> str:
        return value.strip()


class ScanRequest(BaseModel):
    sports: list[Sport] = Field(default_factory=lambda: ["nba", "soccer"])
    days_ahead: int = Field(default=3, ge=1, le=14)
    paper_trade: bool = False


class QualityComponentOut(BaseModel):
    name: str
    score: float
    weight: float
    note: str | None = None


class OpportunityOut(BaseModel):
    game_uid: str
    sport: str
    league_id: str
    game_date: str
    kickoff_utc: str | None = None
    home_name: str
    away_name: str
    market: str
    selection: str
    model_probability: float
    market_probability: float | None = None
    price_decimal: float | None = None
    bookmaker: str | None = None
    edge: float | None = None
    ev_per_unit: float | None = None
    kelly_fraction: float = 0.0
    stake: float = 0.0
    confidence: float
    edge_score: float
    data_quality: float
    model_id: str | None = None
    model_version: str | None = None
    components: dict[str, float] = Field(default_factory=dict)
    agreement: float = 1.0
    explanation: list[dict[str, Any]] = Field(default_factory=list)
    availability: dict[str, Any] | None = None
    flags: list[str] = Field(default_factory=list)
    quality_detail: dict[str, Any] = Field(default_factory=dict)
    edge_detail: dict[str, Any] = Field(default_factory=dict)
    n_bookmakers: int = 0
    probability_range: list[float] | None = None
    home_team_uid: str | None = None
    away_team_uid: str | None = None


class ScanResponse(BaseModel):
    generated_at: str
    sport: str
    opportunities: list[OpportunityOut]
    predictions: list[OpportunityOut]
    portfolio: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    cached: bool = False


class GameOut(BaseModel):
    game_uid: str
    sport: str
    league_id: str
    season: str
    game_date: str
    kickoff_utc: str | None = None
    status: str
    home_name: str
    away_name: str
    home_team_uid: str
    away_team_uid: str
    home_score: float | None = None
    away_score: float | None = None
    venue: str | None = None


class SourceHealth(BaseModel):
    source: str
    dataset: str
    status: str | None = None
    last_success: str | None = None
    age_minutes: float | None = None
    state: str
    message: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    mode: str
    checked_at: str
    database: dict[str, Any]
    sources: list[SourceHealth]
    models: list[dict[str, Any]]
    validation: dict[str, Any]


class PerformanceResponse(BaseModel):
    overall: list[dict[str, Any]]
    by_sport: list[dict[str, Any]]
    by_edge_bucket: list[dict[str, Any]]
    by_odds_bucket: list[dict[str, Any]]
    bankroll_curve: list[dict[str, Any]]
    clv: dict[str, Any]
    open_bets: int
    note: str | None = None
