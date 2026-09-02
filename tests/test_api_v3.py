"""V3 API surface.

Empty states matter as much as populated ones here: a dashboard that renders
"0.0%" when the honest answer is "not enough data" is worse than one that
renders nothing, so the endpoints are tested for saying so.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(seeded_db):
    from divinelines.api import app as app_module

    app_module.invalidate_cache()
    return TestClient(app_module.app)


@pytest.fixture()
def settled_client(seeded_db):
    """A client backed by one settled prediction with a real close."""
    from divinelines.api import app as app_module
    from divinelines.betting.ledger import PredictionRecord, record_predictions
    from divinelines.betting.settlement import settle
    from divinelines.db.repository import upsert_games, upsert_odds
    from divinelines.models.registry import ModelRecord, register

    start = datetime.now(timezone.utc) - timedelta(days=2)
    upsert_games([{
        "game_uid": "nba:settled", "sport": "nba", "league_id": "NBA", "season": "2025-26",
        "game_date": start.date().isoformat(), "kickoff_utc": start.isoformat(),
        "status": "final", "home_team_uid": "nba:BOS", "away_team_uid": "nba:LAL",
        "home_score": 110.0, "away_score": 104.0, "neutral_site": 0, "venue": None,
        "source": "test", "retrieved_at": start.isoformat(),
    }])
    upsert_odds([
        {"game_uid": "nba:settled", "sport": "nba", "market": "h2h", "selection": selection,
         "bookmaker": "BookA", "price_decimal": price, "captured_at": start.isoformat(),
         "book_updated": None, "is_closing": 1 if phase == "close" else 0,
         "phase": phase, "source": "espn_odds"}
        for phase, prices in (("open", {"home": 2.10, "away": 1.80}),
                              ("close", {"home": 1.90, "away": 2.00}))
        for selection, price in prices.items()
    ])
    register(ModelRecord(model_id="m", sport="nba", kind="ensemble", model_version="v1",
                         feature_set=[], feature_set_version="t"))
    record_predictions([PredictionRecord(
        sport="nba", game_uid="nba:settled", market="h2h", selection="home",
        model_probability=0.6, market_probability=0.52, price_decimal=2.10,
        bookmaker="BookA", edge=0.08, ev_per_unit=0.26, stake=20.0,
        model_id="m", model_version="v1", mode="backtest",
    )])
    settle(sport="nba")

    app_module.invalidate_cache()
    return TestClient(app_module.app)


class TestModelHealthApi:
    def test_empty_ledger_says_so_rather_than_reporting_zero(self, client):
        payload = client.get("/api/model-health?sport=nba").json()
        assert payload["sample_size"] == 0
        assert payload["status"] == "INSUFFICIENT_SAMPLE"
        assert "no graded predictions" in payload["status_reason"]

    def test_predictive_and_betting_health_are_separate(self, client):
        payload = client.get("/api/model-health?sport=nba").json()
        assert "predictive" in payload
        assert "betting" in payload
        assert "market_comparison" in payload
        note = payload["note"].lower()
        assert "predictive health" in note and "betting health" in note

    def test_unknown_window_is_rejected(self, client):
        assert client.get("/api/model-health?sport=nba&window=forever").status_code == 422

    def test_invalid_sport_is_rejected(self, client):
        assert client.get("/api/model-health?sport=cricket").status_code == 422

    def test_all_windows_endpoint(self, client):
        payload = client.get("/api/model-health/all?sport=nba").json()
        assert set(payload["windows"]) >= {"all_time", "last_30d"}
        assert "regression" in payload
        assert "stability" in payload

    def test_populated_health(self, settled_client):
        payload = settled_client.get("/api/model-health?sport=nba").json()
        assert payload["sample_size"] == 1
        assert payload["predictive"]["n"] == 1

    def test_history_endpoint(self, client):
        assert client.get("/api/model-health/history?sport=nba").status_code == 200

    def test_persist_writes_a_snapshot(self, settled_client):
        from divinelines.db.connection import query_df

        settled_client.get("/api/model-health?sport=nba&persist=true")
        assert len(query_df("SELECT * FROM model_health_snapshots")) >= 1


class TestClvApi:
    def test_empty_clv_reports_insufficient_sample(self, client):
        payload = client.get("/api/clv").json()
        assert payload["sample_size"] == 0
        assert payload["sufficient_sample"] is False
        assert "not profit" in payload["disclaimer"]

    def test_clv_and_roi_are_reported_separately(self, settled_client):
        payload = settled_client.get("/api/clv?sport=nba").json()
        assert "overall" in payload
        assert "vs_profit" in payload
        assert payload["vs_profit"]["note"]

    def test_populated_clv_has_cohorts(self, settled_client):
        payload = settled_client.get("/api/clv?sport=nba").json()
        assert payload["sample_size"] == 1
        assert "sport" in payload["cohorts"]

    def test_basis_switch_is_labelled(self, settled_client):
        consensus = settled_client.get("/api/clv?sport=nba&basis=consensus").json()
        same_book = settled_client.get("/api/clv?sport=nba&basis=same_book").json()
        assert consensus["basis_description"] != same_book["basis_description"]

    def test_invalid_basis_rejected(self, client):
        assert client.get("/api/clv?basis=magic").status_code == 422

    def test_skill_endpoint_explains_when_unavailable(self, client):
        payload = client.get("/api/clv/skill?sport=nba").json()
        assert payload["available"] is False
        assert payload["reason"]

    def test_coverage_endpoint(self, client):
        payload = client.get("/api/clv/coverage").json()
        assert "coverage" in payload
        assert "settlement_state" in payload


class TestSettlementApi:
    def test_dry_run_settlement(self, settled_client):
        payload = settled_client.post("/api/settle?dry_run=true").json()
        assert "scanned" in payload
        assert "clv" in payload


class TestTimelineApi:
    def test_unknown_game_returns_404(self, client):
        assert client.get("/api/events/nba:nope/timeline").status_code == 404

    def test_timeline_includes_predictions_and_market(self, settled_client):
        payload = settled_client.get("/api/events/nba:settled/timeline").json()
        assert payload["found"] is True
        kinds = {entry["kind"] for entry in payload["timeline"]}
        assert "prediction" in kinds
        assert "market" in kinds
        assert payload["clv"]

    def test_prediction_versions_endpoint(self, settled_client):
        payload = settled_client.get("/api/events/nba:settled/predictions").json()
        assert payload["count"] == 1


class TestLineupApi:
    def test_missing_lineups_return_an_empty_but_valid_payload(self, client):
        payload = client.get("/api/lineups/nba:upcoming").json()
        assert payload["lineup_state"] == "unknown"
        assert payload["teams"] == {}
        assert payload["freshness"]["state"] == "missing"

    def test_coverage_endpoint_explains_itself(self, client):
        payload = client.get("/api/lineups?sport=soccer").json()
        assert "coverage" in payload
        assert "impact_pairs" in payload
        assert "note" in payload


class TestOperationsApi:
    def test_source_health(self, client):
        payload = client.get("/api/source-health").json()
        assert "sources" in payload
        assert "freshness" in payload
        assert "settlement" in payload

    def test_data_quality_surfaces_gaps(self, client):
        payload = client.get("/api/data-quality").json()
        assert "finished_games_without_price" in payload
        assert "upcoming_without_lineups" in payload
        assert "stale_sources" in payload
        assert "closing_line_coverage" in payload

    def test_v2_routes_still_work(self, client):
        """V3 must not break the contract V2 established."""
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/games?sport=nba").status_code == 200
        assert client.get("/api/performance").status_code == 200
