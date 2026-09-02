"""Match Centre: parsing, spatial normalisation, momentum, assembly, replay.

The load-bearing tests in this file are the adversarial ones at the bottom. It
is easy to write a test that says "the replay shows 13 events at minute 32";
it is much harder to fool a test that injects a goal from the 72nd minute and
asserts the 32nd-minute view is byte-for-byte unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from divinelines.identity import player_name_key, same_player, soccer_player_uid
from divinelines.matchcenter import momentum as momentum_module
from divinelines.matchcenter.momentum import MOMENTUM_VERSION, momentum_series, momentum_summary
from divinelines.matchcenter.quality import match_intelligence
from divinelines.matchcenter.report import match_report
from divinelines.matchcenter.service import (
    MatchNotFound,
    match_center,
    match_events,
    match_momentum,
    match_passes,
    resolve_bounds,
)
from divinelines.matchcenter.spatial import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    distance_to_goal,
    event_density,
    has_position,
    normalise_point,
    shot_map,
)
from divinelines.matchcenter.stats import contributions, player_lines, team_comparison
from divinelines.sources import espn_match

FIXTURE = Path(__file__).parent / "fixtures" / "espn_soccer_summary.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- parsing

def test_context_reads_the_match_identity_from_the_payload(payload):
    context = espn_match.parse_context(payload)
    assert context.match_state == "FINISHED"
    assert context.home_name == "Manchester United"
    assert context.away_name == "Arsenal"
    assert (context.home_score, context.away_score) == (0, 1)
    assert context.venue == "Old Trafford"
    assert context.attendance == 73475
    assert context.officials == ["Simon Hooper"]
    assert context.home_formation and context.away_formation


def test_every_parsed_event_lands_in_the_taxonomy(payload):
    events = espn_match.parse_events(payload)
    assert events
    assert all(event.event_type != "other" for event in events), (
        "an unmapped source type would disappear from every panel that filters by type"
    )


def test_narrative_commentary_without_a_play_is_not_an_event():
    """'Lineups are announced' is prose, not something that happened."""
    events = espn_match.parse_events({
        "commentary": [
            {"sequence": 0, "text": "Lineups are announced."},
            {"sequence": 1, "play": {"id": "1", "type": {"type": "corner-awarded"},
                                     "clock": {"value": 60.0}, "period": {"number": 1}}},
        ]
    })
    assert [event.event_type for event in events] == ["corner"]


def test_events_are_ordered_by_match_clock_not_by_feed_order(payload):
    """keyEvents is appended after the commentary stream.

    Trusting feed order put kick-off at the end of the match, which gave a
    10th-minute replay the full-time score.
    """
    events = espn_match.parse_events(payload)
    clocks = [(event.period or 0, event.clock_seconds or 0.0) for event in events]
    assert clocks == sorted(clocks)
    assert events[0].event_type == "kickoff"


def test_a_play_appearing_in_both_feeds_is_merged_not_duplicated(payload):
    events = espn_match.parse_events(payload)
    ids = [event.external_id for event in events if event.external_id]
    assert len(ids) == len(set(ids))


def test_key_events_fill_in_what_commentary_omits(payload):
    """The goal is in both feeds; the merged copy keeps the richer fields."""
    goals = [event for event in espn_match.parse_events(payload)
             if event.event_type == "goal"]
    assert goals, "the fixture contains a goal"
    assert goals[0].player_name == "Riccardo Calafiori"
    assert goals[0].wallclock_utc


def test_team_and_player_statistics_are_parsed(payload):
    teams = espn_match.parse_team_stats(payload)
    assert {line.home_away for line in teams} == {"home", "away"}
    assert teams[0].stats["possessionPct"][0] is not None

    players = espn_match.parse_players(payload)
    assert players
    keeper = next(p for p in players if p.position_group == "goalkeeper")
    assert keeper.player_uid.startswith("soccer:espn:")
    assert "saves" in keeper.stats


def test_standings_parse_whichever_shape_espn_sends(payload):
    rows = espn_match.parse_standings(payload)
    assert rows and all(row["team_name"] for row in rows)
    # Object-shaped team, as the standings endpoint sends it.
    other = espn_match.parse_standings({
        "standings": {"groups": [{"header": "X", "standings": {"entries": [
            {"team": {"displayName": "Arsenal", "id": "359"},
             "stats": [{"name": "points", "value": 3.0}]},
        ]}}]}
    })
    assert other[0]["team_name"] == "Arsenal"
    assert other[0]["points"] == 3.0


def test_match_states_normalise_including_unknown_names():
    assert espn_match.normalise_state("STATUS_FULL_TIME", "post") == "FINISHED"
    assert espn_match.normalise_state("STATUS_HALFTIME", "in") == "HALFTIME"
    assert espn_match.normalise_state("STATUS_SOMETHING_NEW", "pre") == "SCHEDULED"
    assert espn_match.normalise_state(None, None) == "SCHEDULED"


# ------------------------------------------------------------- identity

def test_one_player_survives_three_spellings():
    assert same_player("Kevin De Bruyne", "De Bruyne, Kevin")
    assert same_player("Gabriel Magalhães", "Gabriel Magalhaes")
    assert player_name_key("Vinícius Júnior") == player_name_key("Vinicius Junior")


def test_a_source_athlete_id_beats_a_name_derived_identity():
    assert soccer_player_uid("Anyone", 274272) == "soccer:espn:274272"
    assert soccer_player_uid("Altay Bayindir") == "soccer:name:altay-bayindir"


def test_different_players_do_not_collide():
    assert not same_player("Gabriel Jesus", "Gabriel Martinelli")


# -------------------------------------------------------------- spatial

def test_the_origin_is_treated_as_a_missing_position():
    """ESPN sends 0.0/0.0 for events it has no position for."""
    assert not has_position(0.0, 0.0)
    assert normalise_point(0.0, 0.0, "home") is None
    assert has_position(0.01, 0.5)


def test_normalised_points_stay_on_the_pitch():
    for x in (0.0, 0.01, 0.25, 0.5, 0.99, 1.0):
        for y in (0.01, 0.5, 0.99):
            for side in ("home", "away"):
                point = normalise_point(x, y, side)
                if point is None:
                    continue
                assert 0.0 <= point[0] <= PITCH_LENGTH
                assert 0.0 <= point[1] <= PITCH_WIDTH


def test_both_sides_attack_their_own_end():
    """The source frame is relative to whoever acted.

    Without the flip, both teams' shots land in the same half — which is the
    single most misleading thing a shot map can do.
    """
    home = normalise_point(0.05, 0.5, "home")
    away = normalise_point(0.05, 0.5, "away")
    assert home is not None and away is not None
    assert home[0] > PITCH_LENGTH / 2, "home attacks right"
    assert away[0] < PITCH_LENGTH / 2, "away attacks left"
    assert distance_to_goal(home, "home") < 10
    assert distance_to_goal(away, "away") < 10


def test_shot_map_drops_unlocated_shots_rather_than_placing_them():
    events = [
        {"event_type": "goal", "home_away": "home", "source_x": 0.05, "source_y": 0.5,
         "event_row_id": 1, "minute": 10.0},
        {"event_type": "shot_on_target", "home_away": "away", "source_x": 0.0,
         "source_y": 0.0, "event_row_id": 2, "minute": 20.0},
        {"event_type": "foul", "home_away": "home", "source_x": 0.6, "source_y": 0.4,
         "event_row_id": 3, "minute": 30.0},
    ]
    points = shot_map(events)
    assert [point.event_row_id for point in points] == [1]
    assert points[0].outcome == "goal"


def test_event_density_reports_its_own_coverage_and_says_it_is_not_tracking():
    events = [
        {"event_type": "foul", "home_away": "home", "source_x": 0.6, "source_y": 0.4},
        {"event_type": "yellow_card", "home_away": "home", "source_x": 0.0, "source_y": 0.0},
    ]
    density = event_density(events)
    assert density["events_considered"] == 2
    assert density["events_located"] == 1
    assert density["not_tracking"] is True
    assert sum(sum(row) for row in density["grid"]) == 1


# ------------------------------------------------------------- momentum

def _event(minute, event_type, side, **extra):
    return {"minute": minute, "event_type": event_type, "home_away": side,
            "clock_display": f"{int(minute)}'", **extra}


def test_momentum_is_empty_without_clocked_events():
    result = momentum_series([{"event_type": "goal", "home_away": "home", "minute": None}])
    assert result["available"] is False
    assert "no clocked events" in result["reason"]


def test_a_goal_pushes_the_curve_toward_the_scoring_side():
    result = momentum_series([_event(10.0, "goal", "away")], until_minute=11)
    assert result["series"][-1]["net"] < 0
    assert result["parameters"]["version"] == MOMENTUM_VERSION


def test_influence_decays_with_time():
    result = momentum_series([_event(0.0, "goal", "home")], until_minute=40)
    at_one = abs(result["series"][1]["net"])
    at_thirty = abs(result["series"][30]["net"])
    assert at_thirty < at_one / 4


def test_the_axis_runs_to_the_requested_replay_minute():
    """Stopping the axis at the last event hid a goal scored 0.9 minutes in."""
    result = momentum_series([_event(32.9, "goal", "away")], until_minute=33)
    assert result["series"][-1]["minute"] == 33.0
    assert result["series"][-1]["net"] < -10


def test_structural_markers_never_move_the_curve():
    with_markers = momentum_series(
        [_event(10.0, "goal", "home"), _event(45.0, "halftime", "home"),
         _event(46.0, "substitution", "home")],
        until_minute=50,
    )
    without = momentum_series([_event(10.0, "goal", "home")], until_minute=50)
    assert with_markers["series"] == without["series"]


def test_swings_name_an_associated_event_never_a_cause():
    result = momentum_series([_event(20.0, "goal", "home")], until_minute=30)
    swings = result["swings"]
    assert swings
    assert all("associated" in swing["note"] for swing in swings)
    assert not any("cause" in swing["note"].replace("not an established cause", "")
                   for swing in swings)


def test_momentum_summary_counts_both_sides():
    result = momentum_series(
        [_event(5.0, "goal", "home"), _event(60.0, "goal", "away"),
         _event(61.0, "goal", "away")],
    )
    summary = momentum_summary(result)
    assert summary["available"]
    assert summary["minutes_home_ahead"] > 0
    assert summary["minutes_away_ahead"] > 0
    assert summary["version"] == MOMENTUM_VERSION


def test_momentum_weights_are_ordered_the_way_the_docstring_claims():
    weights = momentum_module.EVENT_WEIGHTS
    assert weights["goal"] > weights["shot_on_target"] > weights["shot_blocked"] > weights["corner"]
    assert weights["red_card"] < 0 and weights["yellow_card"] < 0


# ---------------------------------------------------------------- stats

def test_team_comparison_normalises_the_two_percentage_conventions():
    """passPct arrives as 0.8, possessionPct as 61.1. Both must read as percent."""
    rows = [
        {"home_away": "home", "stat_name": "passPct", "stat_value": 0.8, "display_value": "0.8"},
        {"home_away": "away", "stat_name": "passPct", "stat_value": 0.7, "display_value": "0.7"},
        {"home_away": "home", "stat_name": "possessionPct", "stat_value": 61.1,
         "display_value": "61.1"},
        {"home_away": "away", "stat_name": "possessionPct", "stat_value": 38.9,
         "display_value": "38.9"},
    ]
    result = team_comparison(rows)
    values = {row["stat"]: (row["home"], row["away"]) for row in result["comparisons"]}
    assert values["passPct"] == (80.0, 70.0)
    assert values["possessionPct"] == (61.1, 38.9)


def test_no_expected_goals_metric_is_manufactured():
    result = team_comparison([
        {"home_away": "home", "stat_name": "totalShots", "stat_value": 12.0,
         "display_value": "12"},
        {"home_away": "away", "stat_name": "totalShots", "stat_value": 9.0,
         "display_value": "9"},
    ])
    assert not any("xg" in row["stat"].lower() for row in result["comparisons"])
    assert "expected-goals" in result["note"] or "expected goals" in result["note"].lower()


def test_player_lines_are_position_aware_and_carry_no_rating():
    roster = [
        {"player_uid": "p1", "player_name": "Keeper", "position_group": "goalkeeper",
         "status": "starter", "home_away": "home", "formation_place": "1"},
        {"player_uid": "p2", "player_name": "Striker", "position_group": "forward",
         "status": "starter", "home_away": "home", "formation_place": "11"},
    ]
    stats = [{"player_uid": "p1", "stat_name": "saves", "stat_value": 3.0,
              "display_value": "3"}]
    result = player_lines(stats, roster)
    keeper, striker = result["players"]
    assert [stat["stat"] for stat in keeper["stats"]][0] == "saves"
    assert "totalGoals" in [stat["stat"] for stat in striker["stats"]]
    assert result["rated"] is False


def test_a_substitute_with_no_touches_still_appears():
    result = player_lines([], [
        {"player_uid": "p9", "player_name": "Unused Sub", "position_group": "midfielder",
         "status": "bench", "home_away": "away", "formation_place": None},
    ])
    assert result["players"][0]["player_name"] == "Unused Sub"
    assert result["players"][0]["has_stats"] is False


def test_contributions_credit_an_own_goal_separately():
    result = contributions([
        {"event_type": "goal", "player_uid": "a", "player_name": "Scorer",
         "team_uid": "t1", "home_away": "home", "assist_player_name": "Helper",
         "clock_display": "10'"},
        {"event_type": "own_goal", "player_uid": "b", "player_name": "Unlucky",
         "team_uid": "t2", "home_away": "away", "clock_display": "70'"},
    ])
    scorer = next(row for row in result["goals"] if row["player_name"] == "Scorer")
    unlucky = next(row for row in result["goals"] if row["player_name"] == "Unlucky")
    assert (scorer["goals"], scorer["own_goals"]) == (1, 0)
    assert (unlucky["goals"], unlucky["own_goals"]) == (0, 1)
    assert result["assists"][0]["player_name"] == "Helper"
    assert len(result["assists"]) == 1, "an own goal has no assist"


# -------------------------------------------------------------- quality

def test_intelligence_reports_components_not_one_score():
    result = match_intelligence(
        events=[{"event_type": "goal"}], team_stats=[], player_stats_count=0,
        lineup_rows=[], odds_snapshots=0, predictions=0, located_events=1, context=None,
    )
    states = {component["name"]: component["state"] for component in result["components"]}
    assert states["events"] == "present"
    assert states["team_stats"] == "absent"
    assert states["passing_network"] == "absent"
    assert states["expected_goals"] == "absent"
    assert result["grade"] == "partial"


def test_permanently_absent_components_do_not_drag_the_grade_down():
    """Passing, tracking and xG are missing for every match in the platform.

    Grading each match down for a platform-wide gap would make the badge
    meaningless.
    """
    result = match_intelligence(
        events=[{"event_type": "goal"}] * 40,
        team_stats=[{"stat_name": "totalShots"}] * 30,
        player_stats_count=100,
        lineup_rows=[{"status": "starter", "lineup_state": "final"}] * 22,
        odds_snapshots=18, predictions=2, located_events=30, context={"venue": "X"},
    )
    assert result["grade"] == "high"
    assert any(component["state"] == "absent" for component in result["components"])


# --------------------------------------------------- assembly and replay

def test_match_center_raises_for_an_unknown_fixture(soccer_match_with_events):
    with pytest.raises(MatchNotFound):
        match_center("soccer:NOPE:1999-01-01:a-vs-b")


def test_match_center_assembles_every_panel(soccer_match_with_events):
    payload = match_center(soccer_match_with_events)
    assert payload["match"]["home"]["name"]
    assert payload["events"]
    assert payload["momentum"]["available"]
    assert payload["shots"]["located"] >= 1
    assert payload["statistics"]["comparisons"]
    assert payload["lineups"]["home"]["starters"]
    assert payload["quality"]["components"]
    assert payload["state"]["mode"] == "POST_MATCH"


def test_the_passing_panel_says_no_data_with_a_reason(soccer_match_with_events):
    result = match_passes(soccer_match_with_events)
    assert result["available"] is False
    assert result["state"] == "NO_DATA"
    assert "pass events" in result["reason"]
    assert result["passes"] == []
    assert result["requires"], "the panel says what a provider would have to supply"


def test_replay_truncates_the_event_stream(soccer_match_with_events):
    early = match_center(soccer_match_with_events, minute=20)
    late = match_center(soccer_match_with_events, minute=90)
    assert len(early["events"]) < len(late["events"])
    assert all(event["minute"] <= 20 for event in early["events"])
    assert early["state"]["mode"] == "REPLAY"


def test_the_score_at_a_replay_position_is_the_score_at_that_minute(
    soccer_match_with_events,
):
    before = match_center(soccer_match_with_events, minute=20)
    after = match_center(soccer_match_with_events, minute=80)
    assert (before["match"]["home"]["score"], before["match"]["away"]["score"]) == (0, 0)
    assert after["match"]["away"]["score"] == 1


def test_full_time_box_statistics_are_withheld_during_a_replay(
    soccer_match_with_events,
):
    """Showing full-time possession next to a 20th-minute score is the most
    convincing way to be wrong."""
    replay = match_center(soccer_match_with_events, minute=20)
    stats = {row["stat"] for row in replay["statistics"]["comparisons"]}
    assert "possessionPct" not in stats
    assert "possessionPct" in replay["statistics"]["unavailable"]
    assert replay["statistics"]["basis"].startswith("recounted")

    full = match_center(soccer_match_with_events)
    assert "possessionPct" in {row["stat"] for row in full["statistics"]["comparisons"]}


def test_a_replay_derives_an_information_cut_off_from_kick_off(
    soccer_match_with_events,
):
    payload = match_center(soccer_match_with_events, minute=30)
    assert payload["replay"]["information_as_of"] is not None
    assert payload["replay"]["replay_minute"] == 30
    assert payload["replay"]["events_basis"].startswith("match clock")


def test_a_retrospectively_ingested_match_says_so(soccer_match_with_events):
    """The event stream was read after full time. Replaying it reconstructs
    what happened, not what we knew."""
    payload = match_center(soccer_match_with_events, minute=30)
    assert payload["replay"]["retrospective_events"] is True
    report = match_report(soccer_match_with_events, minute=30)
    assert any("ingested after full time" in note for note in report["limitations"])


def test_bounds_keep_the_two_clocks_separate():
    game = {"kickoff_utc": "2026-05-24T15:00:00+00:00", "game_date": "2026-05-24"}
    bounds = resolve_bounds(game, None, 30)
    assert bounds.observation is None, "a match-clock replay is not a claim about our history"
    assert bounds.information.startswith("2026-05-24T15:30")

    explicit = resolve_bounds(game, "2026-05-24T15:10:00+00:00", None)
    assert explicit.observation == explicit.information


# ---------------------------------------------------------------- report

def test_the_report_states_its_limitations(soccer_match_with_events):
    report = match_report(soccer_match_with_events)
    assert report["result"]["headline"]
    assert report["limitations"]
    assert any("Passing network" in note for note in report["limitations"])
    assert any("Expected goals" in note for note in report["limitations"])


def test_the_report_never_claims_a_cause(soccer_match_with_events):
    report = match_report(soccer_match_with_events)
    for swing in report["momentum"]["largest_swings"]:
        assert swing["note"] == "associated event, not an established cause"


def test_the_report_prose_reflects_the_replay_position(soccer_match_with_events):
    early = match_report(soccer_match_with_events, minute=20)
    late = match_report(soccer_match_with_events)
    assert len(early["key_events"]) < len(late["key_events"])


# ---------------------------------------------- adversarial leakage tests

def test_a_goal_scored_later_is_invisible_at_an_earlier_minute(
    soccer_match_with_events, insert_event,
):
    """Inject a 72nd-minute goal and assert the 32nd-minute view is unchanged."""
    before = match_center(soccer_match_with_events, minute=32)
    insert_event(minute=72.0, event_type="goal", side="home", sequence=900)
    after = match_center(soccer_match_with_events, minute=32)

    assert after["events"] == before["events"]
    assert after["match"]["home"]["score"] == before["match"]["home"]["score"]
    assert after["momentum"]["series"] == before["momentum"]["series"]
    assert after["shots"]["points"] == before["shots"]["points"]

    # ...and it is there when the bound moves past it.
    full = match_center(soccer_match_with_events, minute=90)
    assert any(event["minute"] == 72.0 for event in full["events"])


def test_an_event_observed_later_is_invisible_at_an_earlier_observation_time(
    soccer_match_with_events, insert_event,
):
    """The other clock: what the platform had *seen*, not what had happened."""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
    insert_event(minute=5.0, event_type="goal", side="home", sequence=901,
                 observed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    seen_before = match_events(soccer_match_with_events, as_of=yesterday)
    assert not any(event["sequence"] == 901 for event in seen_before)

    seen_now = match_events(soccer_match_with_events)
    assert any(event["sequence"] == 901 for event in seen_now)


def test_a_price_captured_after_the_replay_position_is_excluded(
    soccer_match_with_events, insert_price,
):
    early = match_center(soccer_match_with_events, minute=10)
    insert_price(captured_at="2026-05-24T16:59:00+00:00", price=9.99)
    still_early = match_center(soccer_match_with_events, minute=10)

    assert still_early["market"]["snapshots"] == early["market"]["snapshots"]
    later = match_center(soccer_match_with_events, minute=120)
    assert later["market"]["snapshots"] > early["market"]["snapshots"]


def test_a_prediction_created_after_the_replay_position_is_excluded(
    soccer_match_with_events, insert_prediction,
):
    insert_prediction(created_at="2026-05-24T16:59:00+00:00", probability=0.99)
    early = match_center(soccer_match_with_events, minute=10)
    late = match_center(soccer_match_with_events, minute=120)
    assert early["model"]["count"] == 0
    assert late["model"]["count"] == 1


def test_a_lineup_observed_after_the_replay_position_is_excluded(
    soccer_match_with_events, insert_lineup_observation,
):
    insert_lineup_observation(observed_at="2026-05-24T16:59:00+00:00",
                              player_name="Late Arrival")
    early = match_center(soccer_match_with_events, minute=10)
    names = {entry["player_name"]
             for side in ("home", "away")
             for entry in early["lineups"][side]["starters"]}
    assert "Late Arrival" not in names


def test_a_retracted_play_is_stored_but_excluded_from_every_aggregate(
    soccer_match_with_events, insert_event,
):
    """ESPN retracts plays after review. Counting a disallowed goal would be a
    lie with a source attached."""
    insert_event(minute=55.0, event_type="deleted", side="home", sequence=902)
    payload = match_center(soccer_match_with_events)
    assert not any(event["event_type"] == "deleted" for event in payload["events"])
    included = match_events(soccer_match_with_events, include_void=True)
    assert any(event["event_type"] == "deleted" for event in included)


# --------------------------------------------------------------- momentum API

def test_match_momentum_endpoint_function_respects_the_bound(soccer_match_with_events):
    early = match_momentum(soccer_match_with_events, minute=15)
    late = match_momentum(soccer_match_with_events, minute=90)
    assert len(early["series"]) < len(late["series"])
    assert early["summary"]["version"] == MOMENTUM_VERSION
