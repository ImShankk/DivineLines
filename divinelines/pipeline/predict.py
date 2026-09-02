"""Game-day prediction and +EV scanning.

The full chain for one slate:

    features -> model -> availability adjustment -> market comparison ->
    EV -> edge quality -> Kelly -> portfolio risk -> stored prediction

Everything a user is shown is produced here, together with the provenance
needed to judge it: which model version, how fresh the inputs were, what the
injury adjustment did, how much the components disagreed, and which risk
constraint bound the stake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..betting.ev import compute_edge_score, expected_value, market_liquidity_proxy
from ..betting.kelly import recommend_stake
from ..betting.ledger import (
    PredictionRecord,
    place_paper_bets,
    record_predictions,
    supersede_predictions,
)
from ..betting.odds_math import build_consensus
from ..betting.portfolio import Candidate, build_portfolio, correlation_warnings
from ..config import nba_season_for_date, settings
from ..data.freshness import (
    QualityComponent,
    assess,
    compute_data_quality,
    freshness_score,
)
from ..db.connection import query_df
from ..db.repository import load_games, load_nba_team_games, load_soccer_matches
from ..logging_setup import get_logger
from ..models.nba_player_impact import (
    build_player_impacts,
    scenario_weighted_probability,
    team_availability,
)
from ..models.registry import latest_model_id, load_artifact
from .refresh import load_cached_player_stats

log = get_logger(__name__)

#: Prediction stages, ordered from least to most informed. The stage a
#: prediction carries is what makes "did the lineup change our view?" answerable.
STAGE_SCHEDULED = "scheduled"
STAGE_PRE_LINEUP = "pre_lineup"
STAGE_PROJECTED_LINEUP = "projected_lineup"
STAGE_CONFIRMED_LINEUP = "confirmed_lineup"

STAGE_ORDER = {
    STAGE_SCHEDULED: 0, STAGE_PRE_LINEUP: 1,
    STAGE_PROJECTED_LINEUP: 2, STAGE_CONFIRMED_LINEUP: 3,
}




@dataclass
class Opportunity:
    """One priced selection, with everything behind the number."""

    game_uid: str
    sport: str
    league_id: str
    game_date: str
    kickoff_utc: str | None
    home_name: str
    away_name: str
    home_team_uid: str
    away_team_uid: str
    market: str
    selection: str
    model_probability: float
    market_probability: float | None
    price_decimal: float | None
    bookmaker: str | None
    edge: float | None
    ev_per_unit: float | None
    kelly_fraction: float
    stake: float
    confidence: float
    edge_score: float
    data_quality: float
    model_id: str | None
    model_version: str | None
    components: dict[str, float] = field(default_factory=dict)
    agreement: float = 1.0
    explanation: list[dict[str, Any]] = field(default_factory=list)
    availability: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)
    quality_detail: dict[str, Any] = field(default_factory=dict)
    edge_detail: dict[str, Any] = field(default_factory=dict)
    n_bookmakers: int = 0
    probability_range: list[float] | None = None
    prediction_stage: str = STAGE_SCHEDULED
    lineup_state: str = "unknown"
    event_start_utc: str | None = None
    seconds_to_event: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items()}
        for key in ("model_probability", "market_probability", "edge", "ev_per_unit",
                    "kelly_fraction", "confidence", "edge_score", "data_quality",
                    "agreement", "price_decimal", "stake"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and value is not None:
                payload[key] = round(float(value), 5)
        return payload


@dataclass
class ScanResult:
    generated_at: str
    sport: str
    opportunities: list[Opportunity]
    all_predictions: list[Opportunity]
    portfolio: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "sport": self.sport,
            "opportunities": [o.to_dict() for o in self.opportunities],
            "predictions": [o.to_dict() for o in self.all_predictions],
            "portfolio": self.portfolio,
            "warnings": self.warnings,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def model_credibility(sport: str) -> dict[str, Any] | None:
    """Does this sport's model actually beat the market in its own backtest?

    The platform refuses to present opportunities as trustworthy when its own
    walk-forward evidence says the model is worse than the closing market.
    That evidence is read here, at the moment of decision, rather than living
    in a document nobody opens.
    """
    import json

    path = settings.paths.artifacts_dir / f"{sport}_backtest_summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # The soccer artifact may hold several configurations; take the best one.
    candidates = [payload] if "probability_metrics" in payload else list(payload.values())
    best: dict[str, Any] | None = None
    for candidate in candidates:
        metrics = (candidate or {}).get("probability_metrics") or {}
        model = metrics.get("model") or metrics.get("ensemble")
        market = metrics.get("market_novig")
        if not model or not market:
            continue
        record = {
            "model_log_loss": model.get("log_loss"),
            "market_log_loss": market.get("log_loss"),
            "n": model.get("n"),
            "beats_market": model.get("log_loss", 9) < market.get("log_loss", 0),
        }
        if best is None or record["model_log_loss"] < best["model_log_loss"]:
            best = record
    return best


def _load_bundle(sport: str) -> tuple[dict[str, Any] | None, str | None]:
    model_id = latest_model_id(sport, "ensemble")
    if not model_id:
        return None, None
    try:
        return load_artifact(model_id), model_id
    except FileNotFoundError:
        log.warning("model artifact missing", extra={"model_id": model_id})
        return None, model_id


def _market_view(game_uid: str, market: str, selections: Sequence[str]
                 ) -> tuple[dict[str, Any] | None, Any]:
    """Consensus + best price for a game, from the latest snapshot per book."""
    from ..db.repository import latest_odds

    odds = latest_odds(game_uid, market)
    if odds.empty:
        return None, None

    quotes: dict[str, dict[str, float]] = {}
    for _, row in odds.iterrows():
        if row["selection"] not in selections:
            continue
        quotes.setdefault(row["bookmaker"], {})[row["selection"]] = float(row["price_decimal"])

    complete = {book: prices for book, prices in quotes.items()
                if all(s in prices for s in selections)}
    if not complete:
        return None, None
    try:
        consensus = build_consensus(complete)
    except ValueError as exc:
        log.warning("consensus failed", extra={"game_uid": game_uid, "error": str(exc)})
        return None, None

    captured = pd.to_datetime(odds["captured_at"], format="mixed", utc=True).max()
    return {"consensus": consensus, "captured_at": captured}, consensus


def _quality(odds_captured: pd.Timestamp | None, injury_state: str | None,
             lineup_known: bool, model_rows: int, *, availability_source: str | None = None,
             extra: list[QualityComponent] | None = None):
    """Assemble the transparent data-quality score for one prediction.

    ``availability_source`` names the injury feed to judge, or ``None`` when the
    sport has no availability feed wired in.  In that case the component is
    dropped rather than scored from an unrelated sport's feed — the remaining
    weights renormalise, and the omission is stated in the returned detail.
    """
    now = datetime.now(timezone.utc)
    odds_age = (now - odds_captured.to_pydatetime()).total_seconds() if odds_captured is not None else None
    odds_component = 1.0 if odds_age is None else float(np.clip(1.0 - odds_age / 3600.0, 0.0, 1.0))

    components = [
        QualityComponent("odds_freshness", odds_component if odds_captured is not None else 0.0,
                         0.35, "no price recorded" if odds_captured is None
                         else f"{(odds_age or 0)/60:.0f} min old"),
        QualityComponent("lineup_certainty", 1.0 if lineup_known else 0.5, 0.15,
                         "confirmed" if lineup_known else "not confirmed"),
        QualityComponent("model_sample", float(np.clip(model_rows / 2000.0, 0.0, 1.0)), 0.25,
                         f"{model_rows} training rows"),
    ]
    if availability_source:
        components.insert(
            1,
            QualityComponent("injury_freshness",
                             freshness_score(assess("injuries", source=availability_source)),
                             0.25, injury_state),
        )
    else:
        components.append(
            QualityComponent("availability_data", 0.0, 0.0,
                             "no availability feed for this sport; component excluded")
        )
    components.extend(extra or [])
    return compute_data_quality(components)


def _event_context(fixture: pd.Series) -> dict[str, Any]:
    """Kick-off time, time remaining, lineup state and the resulting stage.

    Only lineup rows observed before now and not marked ``final`` are consulted:
    a 'final' row was recorded after kick-off, so letting it set the stage would
    backdate knowledge the platform did not have.
    """
    from .ingest_lineups import latest_lineup_state, lineup_state_at

    now = datetime.now(timezone.utc)
    start = pd.to_datetime(fixture.get("kickoff_utc") or fixture.get("game_date"),
                           utc=True, errors="coerce")
    seconds_to_event = (None if pd.isna(start)
                        else int((start.to_pydatetime() - now).total_seconds()))

    observed = lineup_state_at(str(fixture["game_uid"]), now, allow_final=False)
    if observed.empty:
        lineup_state = "unknown"
        stage = STAGE_SCHEDULED if (seconds_to_event or 0) > 6 * 3600 else STAGE_PRE_LINEUP
    else:
        lineup_state = str(observed["lineup_state"].iloc[0])
        stage = (STAGE_CONFIRMED_LINEUP if lineup_state == "confirmed"
                 else STAGE_PROJECTED_LINEUP)

    return {
        "event_start_utc": None if pd.isna(start) else start.isoformat(),
        "seconds_to_event": seconds_to_event,
        "lineup_state": lineup_state,
        "prediction_stage": stage,
        "lineup_players": int(len(observed)),
        "stored_lineup_state": latest_lineup_state(str(fixture["game_uid"])),
    }


def _confidence(agreement: float, data_quality: float, calibration: float,
                availability_uncertainty: float) -> float:
    """Confidence is separate from probability.

    A model can say 67% with high confidence (deep sample, models agree, fresh
    data) or with low confidence (new season, uncertain injuries, components
    disagreeing).  Only confidence scales the stake.
    """
    uncertainty_penalty = float(np.clip(1.0 - availability_uncertainty / 6.0, 0.3, 1.0))
    return float(np.clip(
        0.4 * agreement + 0.3 * (data_quality / 100.0) + 0.2 * calibration
        + 0.1 * uncertainty_penalty, 0.05, 1.0
    )) * uncertainty_penalty


def _flags(*, edge: float | None, market_probability: float | None,
           model_probability: float, data_quality: float, odds_captured: pd.Timestamp | None,
           availability: dict[str, Any] | None, agreement: float,
           early_season: bool, lineup_known: bool) -> list[str]:
    flags: list[str] = []
    if edge is not None and edge >= 0.05:
        flags.append("HIGH_EV")
    if market_probability is not None:
        disagreement = abs(model_probability - market_probability)
        if disagreement >= settings.betting.model_outlier_threshold:
            flags.append("MODEL_OUTLIER")
    if data_quality < 60:
        flags.append("LOW_DATA_QUALITY")
    if odds_captured is None:
        flags.append("NO_MARKET_PRICE")
    else:
        age = (datetime.now(timezone.utc) - odds_captured.to_pydatetime()).total_seconds()
        if age > 3600:
            flags.append("STALE_ODDS")
    if availability and availability.get("uncertainty_margin", 0) > 1.5:
        flags.append("INJURY_UNCERTAINTY")
    if agreement < 0.5:
        flags.append("MODEL_DISAGREEMENT")
    if early_season:
        flags.append("EARLY_SEASON")
    if not lineup_known:
        flags.append("LINEUP_NOT_CONFIRMED")
    return flags


# --------------------------------------------------------------------------
# NBA
# --------------------------------------------------------------------------

def generate_nba_predictions(*, days_ahead: int = 3, include_injuries: bool = True
                             ) -> ScanResult:
    bundle, model_id = _load_bundle("nba")
    notes: list[str] = []
    warnings: list[str] = []
    if bundle is None:
        raise RuntimeError("no trained NBA model — run `divinelines train --sport nba`")

    model = bundle["model"]
    builder = bundle["feature_builder"]
    logits_per_point = bundle.get("logits_per_margin_point", 0.115)
    calibration = model.calibration_quality()

    today = datetime.now(timezone.utc).date()
    horizon = today + timedelta(days=days_ahead)
    fixtures = load_games("nba", status="scheduled", since=str(today), until=str(horizon))
    if fixtures.empty:
        notes.append(f"No NBA fixtures scheduled between {today} and {horizon}.")
        return ScanResult(datetime.now(timezone.utc).isoformat(), "nba", [], [],
                          {"allocations": [], "total_stake": 0.0}, warnings, notes)

    impacts: dict[str, Any] = {}
    statuses: list[Any] = []
    if include_injuries:
        player_stats = load_cached_player_stats()
        if player_stats.empty:
            warnings.append("No cached player stats: injury impact is unavailable "
                            "and predictions are roster-blind.")
        else:
            impacts = build_player_impacts(player_stats)
            notes.append(f"Player impact from {player_stats.attrs.get('season', 'latest')} "
                         f"advanced stats ({len(impacts)} players).")
        statuses = _current_player_statuses()

    season = nba_season_for_date()
    predictions: list[Opportunity] = []

    for _, fixture in fixtures.iterrows():
        game_date = pd.Timestamp(fixture["game_date"])
        context = _event_context(fixture)
        features = builder.upcoming_features(
            fixture["home_team_uid"], fixture["away_team_uid"], game_date, season,
            neutral=bool(fixture.get("neutral_site", 0)),
        )
        frame = pd.DataFrame([features])
        missing = [c for c in model.features if c not in frame.columns]
        for column in missing:
            frame[column] = np.nan

        detail = model.predict_detail(frame)
        base_probability = float(detail["probability"][0])
        agreement = float(detail["agreement"][0])
        components = {
            "xgboost": float(detail["xgboost"][0]),
            "logistic": float(detail["logistic"][0]),
            "elo": float(detail["elo"][0]),
        }
        explanation = model.explain(frame)[0] if model.xgb_model is not None else []

        availability = None
        probability = base_probability
        availability_uncertainty = 0.0
        if impacts and statuses:
            home_adjustment = team_availability(fixture["home_team_uid"], statuses, impacts)
            away_adjustment = team_availability(fixture["away_team_uid"], statuses, impacts)
            availability = scenario_weighted_probability(
                base_probability, home_adjustment, away_adjustment, logits_per_point
            )
            probability = availability["adjusted_probability"]
            availability_uncertainty = availability["uncertainty_margin"]

        market_info, consensus = _market_view(fixture["game_uid"], "h2h", ("home", "away"))
        odds_captured = market_info["captured_at"] if market_info else None

        quality = _quality(
            odds_captured,
            assess("injuries", source="espn_nba").state if include_injuries else None,
            lineup_known=context["lineup_state"] == "confirmed",
            model_rows=model.training_rows,
            availability_source="espn_nba" if include_injuries else None,
        )

        for selection, model_probability in (("home", probability), ("away", 1.0 - probability)):
            market_probability = price = bookmaker = None
            n_books = 0
            if consensus is not None:
                market_probability = consensus.fair_probabilities.get(selection)
                price = consensus.best_price.get(selection)
                bookmaker = consensus.best_bookmaker.get(selection)
                n_books = consensus.n_bookmakers

            ev = (expected_value(model_probability, price, market_probability)
                  if price else None)
            edge = (model_probability - market_probability
                    if market_probability is not None else None)

            confidence = _confidence(agreement, quality.score, calibration,
                                     availability_uncertainty)
            edge_score = compute_edge_score(
                edge=edge or 0.0, model_confidence=confidence,
                data_quality=quality.score, calibration_quality=calibration,
                model_agreement=agreement,
                market_liquidity=market_liquidity_proxy(n_books),
            )
            stake_recommendation = (
                recommend_stake(model_probability=model_probability, price_decimal=price,
                                market_probability=market_probability, confidence=confidence)
                if price else None
            )

            flags = _flags(
                edge=edge, market_probability=market_probability,
                model_probability=model_probability, data_quality=quality.score,
                odds_captured=odds_captured, availability=availability, agreement=agreement,
                early_season=bool(features.get("early_season", 0)),
                lineup_known=context["lineup_state"] == "confirmed",
            )

            predictions.append(
                Opportunity(
                    game_uid=fixture["game_uid"], sport="nba", league_id="NBA",
                    game_date=str(fixture["game_date"]), kickoff_utc=fixture.get("kickoff_utc"),
                    home_name=fixture["home_name"], away_name=fixture["away_name"],
                    home_team_uid=fixture["home_team_uid"],
                    away_team_uid=fixture["away_team_uid"],
                    market="h2h", selection=selection,
                    model_probability=model_probability,
                    market_probability=market_probability,
                    price_decimal=price, bookmaker=bookmaker, edge=edge,
                    ev_per_unit=ev.ev_per_unit if ev else None,
                    kelly_fraction=stake_recommendation.kelly_used if stake_recommendation else 0.0,
                    stake=stake_recommendation.stake if stake_recommendation else 0.0,
                    confidence=confidence, edge_score=edge_score.score,
                    data_quality=quality.score, model_id=model_id,
                    model_version=model.model_version, components=components,
                    agreement=agreement, explanation=explanation, availability=availability,
                    flags=flags, quality_detail=quality.to_dict(),
                    edge_detail=edge_score.to_dict(), n_bookmakers=n_books,
                    probability_range=(availability or {}).get("probability_range"),
                    prediction_stage=context["prediction_stage"],
                    lineup_state=context["lineup_state"],
                    event_start_utc=context["event_start_utc"],
                    seconds_to_event=context["seconds_to_event"],
                )
            )

    return _finalise(predictions, "nba", warnings, notes)


def _current_player_statuses(max_age_hours: int = 48) -> list[Any]:
    """Latest availability record per player, within a freshness window."""
    from ..sources.espn_nba import PlayerAvailability, STATUS_MAP

    rows = query_df(
        """
        SELECT s.player_uid, s.team_uid, s.status, s.detail, s.expected_return,
               s.as_of, s.retrieved_at, p.full_name, p.position
        FROM player_status s
        JOIN (SELECT player_uid, MAX(retrieved_at) AS mx FROM player_status
              WHERE sport='nba' GROUP BY player_uid) latest
          ON latest.player_uid = s.player_uid AND latest.mx = s.retrieved_at
        LEFT JOIN players p ON p.player_uid = s.player_uid
        WHERE s.sport = 'nba'
        """
    )
    if rows.empty:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    statuses: list[PlayerAvailability] = []
    for _, row in rows.iterrows():
        retrieved = pd.to_datetime(row["retrieved_at"], utc=True, format="mixed")
        if pd.isna(retrieved) or retrieved.to_pydatetime() < cutoff:
            continue
        _, play_probability = next(
            ((k, v) for k, (mapped, v) in STATUS_MAP.items() if mapped == row["status"]),
            ("unknown", 0.5),
        )
        statuses.append(
            PlayerAvailability(
                player_name=row.get("full_name") or "", espn_athlete_id=None,
                team_abbr=None, position=row.get("position"), status=row["status"],
                play_probability=play_probability, detail=row.get("detail"),
                expected_return=row.get("expected_return"), as_of=str(row["as_of"]),
            )
        )
        statuses[-1].team_uid = row["team_uid"]  # type: ignore[attr-defined]

    # ``team_availability`` matches on impact.team_uid, so carry the stored team.
    for status in statuses:
        setattr(status, "team_abbr", None)
    return statuses


# --------------------------------------------------------------------------
# Soccer
# --------------------------------------------------------------------------

def generate_soccer_predictions(*, days_ahead: int = 5,
                                league_ids: Sequence[str] | None = None) -> ScanResult:
    bundle, model_id = _load_bundle("soccer")
    notes: list[str] = []
    warnings: list[str] = []
    if bundle is None:
        raise RuntimeError("no trained soccer model — run `divinelines train --sport soccer`")

    model = bundle["model"]
    builder = bundle["feature_builder"]

    today = datetime.now(timezone.utc).date()
    horizon = today + timedelta(days=days_ahead)
    fixtures = load_games("soccer", status="scheduled", since=str(today), until=str(horizon))
    if league_ids is not None and not fixtures.empty:
        fixtures = fixtures[fixtures["league_id"].isin(list(league_ids))]
    if fixtures.empty:
        notes.append(f"No soccer fixtures scheduled between {today} and {horizon}.")
        return ScanResult(datetime.now(timezone.utc).isoformat(), "soccer", [], [],
                          {"allocations": [], "total_stake": 0.0}, warnings, notes)

    predictions: list[Opportunity] = []
    for _, fixture in fixtures.iterrows():
        context = _event_context(fixture)
        features = builder.upcoming_features(fixture)
        frame = pd.DataFrame([features])
        for column in model.features:
            if column not in frame.columns:
                frame[column] = np.nan

        detail = model.predict_detail(frame)
        probabilities = detail["probabilities"][0]
        agreement = float(detail["agreement"][0])
        components = {
            name: float(matrix[0][0]) for name, matrix in detail["components"].items()
        }

        market_info, consensus = _market_view(fixture["game_uid"], "1x2",
                                              ("home", "draw", "away"))
        odds_captured = market_info["captured_at"] if market_info else None
        # Soccer has no injury feed wired in, so that component is omitted
        # rather than borrowed from the NBA feed; the lineup feed does exist and
        # feeds lineup_certainty directly.
        quality = _quality(odds_captured, None,
                           lineup_known=context["lineup_state"] == "confirmed",
                           model_rows=int(model.metrics.get("n_train", 0)),
                           availability_source=None)
        calibration = 0.7 if model.calibrator and model.calibrator.fitted else 0.4

        for index, selection in enumerate(("home", "draw", "away")):
            model_probability = float(probabilities[index])
            market_probability = price = bookmaker = None
            n_books = 0
            if consensus is not None:
                market_probability = consensus.fair_probabilities.get(selection)
                price = consensus.best_price.get(selection)
                bookmaker = consensus.best_bookmaker.get(selection)
                n_books = consensus.n_bookmakers

            ev = (expected_value(model_probability, price, market_probability)
                  if price else None)
            edge = (model_probability - market_probability
                    if market_probability is not None else None)
            confidence = _confidence(agreement, quality.score, calibration, 0.0)
            if features.get("home_is_new_to_league") or features.get("away_is_new_to_league"):
                # A club with almost no history in this division is the single
                # least reliable input the soccer model has.
                confidence *= 0.7
            edge_score = compute_edge_score(
                edge=edge or 0.0, model_confidence=confidence, data_quality=quality.score,
                calibration_quality=calibration, model_agreement=agreement,
                market_liquidity=market_liquidity_proxy(n_books),
            )
            stake_recommendation = (
                recommend_stake(model_probability=model_probability, price_decimal=price,
                                market_probability=market_probability, confidence=confidence)
                if price else None
            )
            flags = _flags(
                edge=edge, market_probability=market_probability,
                model_probability=model_probability, data_quality=quality.score,
                odds_captured=odds_captured, availability=None, agreement=agreement,
                early_season=bool(features.get("home_is_new_to_league", 0)),
                lineup_known=context["lineup_state"] == "confirmed",
            )
            if features.get("home_is_new_to_league") or features.get("away_is_new_to_league"):
                flags.append("PROMOTED_TEAM")

            predictions.append(
                Opportunity(
                    game_uid=fixture["game_uid"], sport="soccer",
                    league_id=fixture["league_id"], game_date=str(fixture["game_date"]),
                    kickoff_utc=fixture.get("kickoff_utc"),
                    home_name=fixture["home_name"], away_name=fixture["away_name"],
                    home_team_uid=fixture["home_team_uid"],
                    away_team_uid=fixture["away_team_uid"],
                    market="1x2", selection=selection,
                    model_probability=model_probability, market_probability=market_probability,
                    price_decimal=price, bookmaker=bookmaker, edge=edge,
                    ev_per_unit=ev.ev_per_unit if ev else None,
                    kelly_fraction=stake_recommendation.kelly_used if stake_recommendation else 0.0,
                    stake=stake_recommendation.stake if stake_recommendation else 0.0,
                    confidence=confidence, edge_score=edge_score.score,
                    data_quality=quality.score, model_id=model_id,
                    model_version=model.model_version, components=components,
                    agreement=agreement, flags=flags, quality_detail=quality.to_dict(),
                    edge_detail=edge_score.to_dict(), n_bookmakers=n_books,
                    prediction_stage=context["prediction_stage"],
                    lineup_state=context["lineup_state"],
                    event_start_utc=context["event_start_utc"],
                    seconds_to_event=context["seconds_to_event"],
                )
            )

    return _finalise(predictions, "soccer", warnings, notes)


# --------------------------------------------------------------------------
# Risk + persistence
# --------------------------------------------------------------------------

def _finalise(predictions: list[Opportunity], sport: str, warnings: list[str],
              notes: list[str]) -> ScanResult:
    """Filter to genuine opportunities and size them under portfolio limits."""
    qualifying = [
        p for p in predictions
        if p.edge is not None and p.edge >= settings.betting.min_edge
        and p.ev_per_unit is not None and p.ev_per_unit > 0
        and p.edge_score >= settings.betting.min_edge_score
        and "MODEL_OUTLIER" not in p.flags
        and p.price_decimal is not None
        and settings.betting.min_decimal_odds <= p.price_decimal <= settings.betting.max_decimal_odds
    ]

    credibility = model_credibility(sport)
    if credibility and not credibility["beats_market"]:
        warnings.insert(
            0,
            f"This {sport} model does NOT beat the market in its own walk-forward "
            f"backtest (model log loss {credibility['model_log_loss']:.4f} vs no-vig "
            f"market {credibility['market_log_loss']:.4f} over {credibility['n']} "
            f"matches). Treat every edge below as unproven: a model that cannot "
            f"out-predict the closing line has no demonstrated betting edge.",
        )
        for prediction in predictions:
            prediction.flags.append("MODEL_UNPROVEN")

    outliers = [p for p in predictions if "MODEL_OUTLIER" in p.flags]
    if outliers:
        warnings.append(
            f"{len(outliers)} selections were rejected as model outliers "
            f"(disagreement with the market above "
            f"{settings.betting.model_outlier_threshold:.0%})."
        )

    candidates = [
        Candidate(
            key=f"{p.game_uid}:{p.market}:{p.selection}", game_uid=p.game_uid, sport=p.sport,
            market=p.market, selection=p.selection,
            teams=(p.home_team_uid, p.away_team_uid), price_decimal=p.price_decimal or 0.0,
            stake=p.stake, model_probability=p.model_probability, edge=p.edge or 0.0,
            edge_score=p.edge_score,
        )
        for p in qualifying
    ]
    portfolio = build_portfolio(candidates)
    allocated = {a.candidate.key: a for a in portfolio.allocations}

    for opportunity in qualifying:
        key = f"{opportunity.game_uid}:{opportunity.market}:{opportunity.selection}"
        allocation = allocated.get(key)
        if allocation is None:
            opportunity.stake = 0.0
            opportunity.flags.append("EXCLUDED_BY_RISK_LIMITS")
        else:
            opportunity.stake = allocation.stake
            opportunity.flags.extend(allocation.binding_constraints)

    warnings.extend(correlation_warnings(candidates))
    return ScanResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sport=sport,
        opportunities=sorted(qualifying, key=lambda o: -o.edge_score),
        all_predictions=predictions,
        portfolio=portfolio.to_dict(),
        warnings=warnings,
        notes=notes,
    )


def persist_scan(result: ScanResult, *, paper_trade: bool = False) -> dict[str, Any]:
    """Store predictions (and optionally open paper bets) for later grading."""
    records = [
        PredictionRecord(
            sport=o.sport, game_uid=o.game_uid, market=o.market, selection=o.selection,
            model_probability=o.model_probability, league_id=o.league_id,
            market_probability=o.market_probability, price_decimal=o.price_decimal,
            bookmaker=o.bookmaker, edge=o.edge, ev_per_unit=o.ev_per_unit,
            kelly_fraction=o.kelly_fraction, stake=o.stake, confidence=o.confidence,
            edge_score=o.edge_score, data_quality=o.data_quality, model_id=o.model_id,
            model_version=o.model_version,
            features={"components": o.components, "agreement": o.agreement},
            explanation={"contributions": o.explanation, "availability": o.availability},
            flags=o.flags, created_at=result.generated_at,
            mode="paper" if paper_trade else settings.mode,
            prediction_stage=o.prediction_stage, lineup_state=o.lineup_state,
            event_start_utc=o.event_start_utc, seconds_to_event=o.seconds_to_event,
            information_snapshot={
                "lineup_state": o.lineup_state,
                "data_quality": o.quality_detail,
                "flags": o.flags,
            },
        )
        for o in result.all_predictions
    ]
    # A new prediction supersedes the previous view of the same fixture rather
    # than replacing it, so pre-lineup and post-lineup opinions both survive and
    # scoring never counts one match twice.
    for game_uid, market in {(o.game_uid, o.market) for o in result.all_predictions}:
        supersede_predictions(game_uid, market, before=result.generated_at,
                              mode="paper" if paper_trade else settings.mode)

    prediction_ids = record_predictions(records)

    staked_ids: list[int] = []
    if paper_trade and prediction_ids:
        opportunity_keys = {(o.game_uid, o.market, o.selection) for o in result.opportunities
                            if o.stake > 0}
        stored = query_df(
            "SELECT prediction_id, game_uid, market, selection FROM predictions "
            "WHERE created_at = ?", (result.generated_at,)
        )
        staked_ids = [
            int(row["prediction_id"]) for _, row in stored.iterrows()
            if (row["game_uid"], row["market"], row["selection"]) in opportunity_keys
        ]
        place_paper_bets(staked_ids)

    return {"predictions_stored": len(prediction_ids), "paper_bets": len(staked_ids)}


def scan(sports: Iterable[str] = ("nba", "soccer"), *, paper_trade: bool = False,
         days_ahead: int = 3, persist: bool = True) -> dict[str, Any]:
    """Run the full scan for each sport, tolerating one sport being unavailable."""
    results: dict[str, Any] = {}
    for sport in sports:
        try:
            if sport == "nba":
                result = generate_nba_predictions(days_ahead=days_ahead)
            else:
                result = generate_soccer_predictions(days_ahead=max(days_ahead, 5))
            persisted = (persist_scan(result, paper_trade=paper_trade) if persist
                         else {"predictions_stored": 0, "paper_bets": 0, "dry_run": True})
            results[sport] = {**result.to_dict(), "persisted": persisted}
        except Exception as exc:
            log.error("scan failed", extra={"sport": sport, "error": str(exc)})
            results[sport] = {"error": str(exc)}
    return results
