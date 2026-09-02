"""Soccer Match Centre API.

The point of these is that the *server* is what enforces the replay bound. A
frontend test that checks the UI hides a future goal proves nothing; a request
for minute 32 that comes back carrying a 58th-minute goal is a bug regardless
of what the browser does with it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from divinelines.api.app import app

MATCH = "soccer:ENG_PL:2026-05-24:arsenal-vs-chelsea"


@pytest.fixture()
def client(soccer_match_with_events):
    return TestClient(app)


def get(client, path, **params):
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------- discovery

def test_match_list_reports_coverage_per_fixture(client):
    payload = get(client, "/api/soccer/matches", league_id="ENG_PL")
    row = next(row for row in payload["matches"] if row["game_uid"] == MATCH)
    assert row["events"] > 0
    assert row["starters"] == 22
    assert row["prices"] > 0
    assert row["predictions"] == 0


def test_match_list_can_filter_to_fixtures_with_an_event_feed(client):
    with_events = get(client, "/api/soccer/matches", with_events=True)
    assert all(row["events"] > 0 for row in with_events["matches"])
    # The second fixture in the seed has no events and must be filtered out.
    everything = get(client, "/api/soccer/matches")
    assert everything["count"] > with_events["count"]


def test_standings_are_computed_from_stored_results(client):
    payload = get(client, "/api/soccer/standings", league_id="ENG_PL", season="2526")
    assert payload["available"]
    assert {row["team_name"] for row in payload["table"]}


# ------------------------------------------------------------- the payload

def test_match_centre_returns_one_coherent_payload(client):
    payload = get(client, f"/api/soccer/match/{MATCH}")
    for section in ("match", "state", "events", "statistics", "lineups", "players",
                    "momentum", "shots", "heatmap", "passing", "market", "model",
                    "model_vs_market", "standings", "quality", "provenance", "replay"):
        assert section in payload, f"missing section: {section}"
    assert payload["match"]["home"]["name"]
    assert payload["state"]["state"] == "FINISHED"


def test_an_unknown_match_is_a_404_not_an_empty_page(client):
    response = client.get("/api/soccer/match/soccer:NOPE:1999-01-01:a-vs-b")
    assert response.status_code == 404
    assert "unknown match" in response.json()["detail"]


def test_every_panel_route_answers(client):
    for suffix in ("events", "momentum", "shots", "heatmap", "passes", "stats",
                   "players", "markets", "report"):
        response = client.get(f"/api/soccer/match/{MATCH}/{suffix}")
        assert response.status_code == 200, f"{suffix}: {response.text}"


def test_the_passes_route_reports_no_data_rather_than_404ing(client):
    payload = get(client, f"/api/soccer/match/{MATCH}/passes")
    assert payload["available"] is False
    assert payload["state"] == "NO_DATA"
    assert payload["reason"]
    assert payload["aggregate_totals"], "the aggregate counts that do exist are shown"


def test_the_heatmap_declares_it_is_not_tracking_data(client):
    payload = get(client, f"/api/soccer/match/{MATCH}/heatmap")
    assert payload["not_tracking"] is True
    assert "tracking" in payload["note"]


# ------------------------------------------------------- replay enforcement

def test_the_replay_bound_is_applied_server_side(client):
    early = get(client, f"/api/soccer/match/{MATCH}/events", minute=32)
    assert all(event["minute"] <= 32 for event in early["events"])
    assert not any(event["event_type"] == "goal" for event in early["events"])

    late = get(client, f"/api/soccer/match/{MATCH}/events", minute=90)
    assert any(event["event_type"] == "goal" for event in late["events"])


def test_the_score_in_the_payload_follows_the_replay_bound(client):
    early = get(client, f"/api/soccer/match/{MATCH}", minute=32)
    late = get(client, f"/api/soccer/match/{MATCH}", minute=90)
    assert early["match"]["away"]["score"] == 0
    assert late["match"]["away"]["score"] == 1


def test_momentum_is_truncated_by_the_replay_bound(client):
    early = get(client, f"/api/soccer/match/{MATCH}/momentum", minute=20)
    late = get(client, f"/api/soccer/match/{MATCH}/momentum", minute=90)
    assert len(early["series"]) < len(late["series"])
    assert max(point["minute"] for point in early["series"]) <= 20


def test_the_shot_map_is_truncated_by_the_replay_bound(client):
    early = get(client, f"/api/soccer/match/{MATCH}/shots", minute=20)
    late = get(client, f"/api/soccer/match/{MATCH}/shots", minute=90)
    assert early["located"] < late["located"]
    assert all(point["minute"] <= 20 for point in early["points"])


def test_a_nonsense_replay_position_is_rejected(client):
    assert client.get(f"/api/soccer/match/{MATCH}", params={"minute": -5}).status_code == 422
    assert client.get(f"/api/soccer/match/{MATCH}", params={"minute": 9999}).status_code == 422


def test_the_report_route_carries_its_limitations(client):
    payload = get(client, f"/api/soccer/match/{MATCH}/report")
    assert payload["limitations"]
    assert any("Passing network" in note for note in payload["limitations"])


# ------------------------------------------------------- existing contracts

def test_the_v2_and_v3_routes_still_answer(client):
    for path in ("/api/health", "/api/data-quality", "/api/source-health",
                 f"/api/lineups/{MATCH}", f"/api/events/{MATCH}/timeline"):
        assert client.get(path).status_code == 200, path


def test_coordinates_returned_to_the_client_are_inside_the_pitch(client):
    payload = get(client, f"/api/soccer/match/{MATCH}/shots")
    length = payload["pitch"]["length"]
    width = payload["pitch"]["width"]
    for point in payload["points"]:
        assert 0 <= point["x"] <= length
        assert 0 <= point["y"] <= width
