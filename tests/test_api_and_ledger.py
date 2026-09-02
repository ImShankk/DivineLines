"""API surface, prediction ledger and settlement.

Settlement is where a mistake costs the most credibility: a bet graded wrong
corrupts every performance number downstream, so each market's grading rules
are tested explicitly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from divinelines.betting.ledger import (
    PredictionRecord,
    _outcome_for_selection,
    bankroll_curve,
    performance_summary,
    place_paper_bets,
    record_predictions,
    settle_open_bets,
)


class TestSettlementGrading:
    def _row(self, market, selection, home, away):
        return pd.Series({"market": market, "selection": selection,
                          "home_score": home, "away_score": away})

    def test_moneyline_home_win(self):
        assert _outcome_for_selection(self._row("h2h", "home", 110, 104)) == "won"
        assert _outcome_for_selection(self._row("h2h", "away", 110, 104)) == "lost"

    def test_moneyline_away_win(self):
        assert _outcome_for_selection(self._row("h2h", "away", 100, 104)) == "won"
        assert _outcome_for_selection(self._row("h2h", "home", 100, 104)) == "lost"

    def test_1x2_draw(self):
        assert _outcome_for_selection(self._row("1x2", "draw", 1, 1)) == "won"
        assert _outcome_for_selection(self._row("1x2", "home", 1, 1)) == "push"
        assert _outcome_for_selection(self._row("1x2", "away", 1, 1)) == "push"

    def test_1x2_home_and_away(self):
        assert _outcome_for_selection(self._row("1x2", "home", 2, 1)) == "won"
        assert _outcome_for_selection(self._row("1x2", "draw", 2, 1)) == "lost"
        assert _outcome_for_selection(self._row("1x2", "away", 1, 2)) == "won"

    def test_totals_over_and_under(self):
        assert _outcome_for_selection(self._row("totals", "over_2.5", 2, 1)) == "won"
        assert _outcome_for_selection(self._row("totals", "under_2.5", 2, 1)) == "lost"
        assert _outcome_for_selection(self._row("totals", "under_2.5", 1, 0)) == "won"

    def test_totals_exact_line_is_a_push(self):
        assert _outcome_for_selection(self._row("totals", "over_3.0", 2, 1)) == "push"

    def test_unfinished_game_is_not_graded(self):
        assert _outcome_for_selection(self._row("h2h", "home", None, None)) is None

    def test_unknown_market_is_not_guessed(self):
        assert _outcome_for_selection(self._row("weird_market", "home", 2, 1)) is None


@pytest.fixture()
def registered_model(seeded_db):
    """Predictions carry a foreign key to a registered model on purpose.

    Provenance is only meaningful if the model that produced a prediction is
    guaranteed to exist, so the ledger tests register one rather than working
    around the constraint.
    """
    from divinelines.models.registry import ModelRecord, register

    register(ModelRecord(
        model_id="test-model", sport="nba", kind="ensemble", model_version="v1",
        feature_set=["diff_elo"], feature_set_version="test", league_id="NBA",
    ))
    return seeded_db


class TestLedger:
    def _prediction(self, game_uid="nba:0022500000", selection="home", stake=25.0,
                    price=2.0, market="h2h", sport="nba"):
        return PredictionRecord(
            sport=sport, game_uid=game_uid, market=market, selection=selection,
            model_probability=0.6, market_probability=0.5, price_decimal=price,
            bookmaker="TestBook", edge=0.1, ev_per_unit=0.2, kelly_fraction=0.05,
            stake=stake, confidence=0.8, edge_score=7.5, data_quality=90.0,
            model_id="test-model", model_version="v1",
        )

    def test_predictions_are_persisted_with_provenance(self, registered_model):
        from divinelines.db.connection import query_df

        ids = record_predictions([self._prediction()])
        assert len(ids) == 1
        stored = query_df("SELECT * FROM predictions").iloc[0]
        assert stored["model_version"] == "v1"
        assert stored["data_quality"] == 90.0
        assert stored["market_prob"] == 0.5

    def test_paper_bets_open_only_where_a_stake_exists(self, registered_model):
        ids = record_predictions([
            self._prediction(stake=25.0),
            self._prediction(selection="away", stake=0.0),
        ])
        assert place_paper_bets(ids) == 1

    def test_settlement_grades_and_pays_correctly(self, registered_model):
        from divinelines.db.connection import query_df

        game = query_df(
            "SELECT game_uid, home_score, away_score FROM games "
            "WHERE sport='nba' AND status='final' LIMIT 1"
        ).iloc[0]
        home_won = game["home_score"] > game["away_score"]

        ids = record_predictions([
            self._prediction(game_uid=game["game_uid"], selection="home", price=2.0)
        ])
        place_paper_bets(ids)
        result = settle_open_bets()
        assert result["settled"] == 1

        bet = query_df("SELECT * FROM bets").iloc[0]
        assert bet["status"] == ("won" if home_won else "lost")
        assert bet["profit"] == pytest.approx(25.0 if home_won else -25.0)

    def test_settlement_is_idempotent(self, registered_model):
        from divinelines.db.connection import query_df

        game = query_df("SELECT game_uid FROM games WHERE status='final' LIMIT 1").iloc[0]
        place_paper_bets(record_predictions([self._prediction(game_uid=game["game_uid"])]))
        assert settle_open_bets()["settled"] == 1
        assert settle_open_bets()["settled"] == 0

    def test_unfinished_games_are_left_open(self, registered_model):
        place_paper_bets(record_predictions([self._prediction(game_uid="nba:upcoming")]))
        assert settle_open_bets()["settled"] == 0

    def test_bankroll_curve_tracks_drawdown(self, registered_model):
        from divinelines.db.connection import query_df

        games = query_df(
            "SELECT game_uid FROM games WHERE status='final' ORDER BY game_date"
        )["game_uid"].tolist()
        ids = record_predictions([
            self._prediction(game_uid=uid, selection="away") for uid in games[:3]
        ])
        place_paper_bets(ids)
        settle_open_bets()
        curve = bankroll_curve()
        assert len(curve) == 3
        assert "drawdown" in curve.columns
        assert (curve["drawdown"] <= 0).all()

    def test_performance_summary_reports_roi(self, registered_model):
        from divinelines.db.connection import query_df

        games = query_df(
            "SELECT game_uid FROM games WHERE status='final' ORDER BY game_date"
        )["game_uid"].tolist()
        place_paper_bets(record_predictions([
            self._prediction(game_uid=uid) for uid in games[:4]
        ]))
        settle_open_bets()
        summary = performance_summary()
        assert not summary.empty
        assert summary.iloc[0]["bets"] == 4
        assert "roi" in summary.columns

    def test_performance_summary_is_empty_without_bets(self, seeded_db):
        assert performance_summary().empty


@pytest.fixture()
def client(seeded_db):
    from divinelines.api import app as app_module

    app_module.invalidate_cache()
    return TestClient(app_module.app)


class TestApi:
    def test_root_reports_the_service(self, client):
        payload = client.get("/").json()
        assert payload["service"] == "DivineLines"
        assert payload["status"] == "online"

    def test_health_returns_a_structured_report(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in ("ok", "degraded", "down")
        assert payload["database"]["connected"] is True
        assert "validation" in payload

    def test_config_never_exposes_the_api_key(self, client):
        payload = client.get("/api/config").json()
        serialised = str(payload)
        assert "odds_api_configured" in payload
        assert "apiKey" not in serialised
        for value in payload.values():
            assert "ODDS_API_KEY" not in str(value)

    def test_games_endpoint_filters(self, client):
        payload = client.get("/api/games?sport=nba&status=final&days=90").json()
        assert payload["count"] == 6
        assert all(game["status"] == "final" for game in payload["games"])

    def test_game_detail_includes_odds_movement(self, client):
        payload = client.get("/api/games/nba:upcoming").json()
        assert payload["game"]["game_uid"] == "nba:upcoming"
        assert "h2h" in payload["odds_movement"]
        assert payload["odds_movement"]["h2h"]["selections"]["home"]["snapshots"] >= 1

    def test_unknown_game_returns_404(self, client):
        assert client.get("/api/games/nba:does-not-exist").status_code == 404

    def test_teams_and_team_detail(self, client):
        teams = client.get("/api/teams?sport=nba").json()
        assert teams["count"] == 30
        detail = client.get("/api/teams/nba:BOS").json()
        assert detail["team"]["canonical_name"] == "Boston Celtics"
        assert len(detail["recent_games"]) > 0

    def test_unknown_team_returns_404(self, client):
        assert client.get("/api/teams/nba:XXX").status_code == 404

    def test_odds_endpoint_returns_prices(self, client):
        payload = client.get("/api/odds/nba:upcoming").json()
        assert len(payload["latest"]) == 4

    def test_odds_endpoint_404s_without_prices(self, client):
        assert client.get("/api/odds/nba:0022500000").status_code == 404

    def test_search_finds_teams_and_games(self, client):
        payload = client.get("/api/search?q=Celtics").json()
        assert any("Celtics" in team["canonical_name"] for team in payload["teams"])

    def test_search_requires_a_real_query(self, client):
        assert client.get("/api/search?q=a").status_code == 422

    def test_performance_endpoint_explains_an_empty_ledger(self, client):
        payload = client.get("/api/performance").json()
        assert payload["overall"] == []
        assert payload["note"]

    def test_models_endpoint(self, client):
        assert "models" in client.get("/api/models").json()

    def test_predictions_without_a_model_returns_503_not_500(self, client):
        response = client.get("/api/predictions?sport=nba")
        assert response.status_code in (200, 503)
        if response.status_code == 503:
            assert "train" in response.json()["detail"]

    def test_legacy_predict_rejects_unknown_teams(self, client):
        response = client.post("/api/predict", json={"home": "XXX", "away": "BOS"})
        assert response.status_code == 422

    def test_legacy_predict_rejects_a_team_playing_itself(self, client):
        response = client.post("/api/predict", json={"home": "BOS", "away": "BOS"})
        assert response.status_code == 422
        assert "itself" in response.json()["detail"]

    def test_legacy_predict_validates_its_body(self, client):
        assert client.post("/api/predict", json={"home": "B"}).status_code == 422

    def test_invalid_sport_is_rejected(self, client):
        assert client.get("/api/predictions?sport=cricket").status_code == 422

    def test_days_ahead_is_bounded(self, client):
        assert client.get("/api/predictions?sport=nba&days_ahead=99").status_code == 422
