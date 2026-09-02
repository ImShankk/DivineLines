"""Lineup chronology, lineup features, prediction versioning and model health.

The single most important test in this file is
``test_a_lineup_observed_later_is_invisible_earlier``: if a confirmed XI can
leak backwards into a prediction made before it was seen, every lineup result
the platform produces is worthless.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from divinelines.analytics.model_health import (
    MIN_SAMPLE_FOR_STATUS,
    STATUS_DEGRADED,
    STATUS_INSUFFICIENT,
    STATUS_MARKET_BEATING,
    STATUS_UNPROVEN,
    STATUS_VALIDATED,
    compute_health,
    detect_regression,
    prediction_stability,
    roi_interval,
)
from divinelines.features.lineup_features import (
    LineupFeatureBuilder,
    REGULAR_WINDOW,
    attach_lineup_features,
)
from divinelines.pipeline.ingest_lineups import (
    information_events_for,
    latest_lineup_state,
    lineup_state_at,
)


def _observation(game_uid: str, team_uid: str, players: list[str], observed_at: str,
                 *, state: str = "confirmed", goalkeeper: str | None = None) -> list[dict]:
    rows = []
    for index, player in enumerate(players):
        rows.append({
            "game_uid": game_uid, "team_uid": team_uid, "sport": "soccer",
            "player_uid": None, "player_name": player, "external_player_id": None,
            "status": "starter", "role": "G" if player == goalkeeper else "CM",
            "position_group": "goalkeeper" if player == goalkeeper else "midfielder",
            "formation_place": str(index + 1), "formation": "4-3-3",
            "lineup_state": state, "observed_at": observed_at,
            "source_timestamp": None, "retrieved_at": observed_at, "source": "test",
        })
    return rows


@pytest.fixture()
def lineup_db(seeded_db):
    from divinelines.db.connection import upsert_rows
    from divinelines.db.repository import upsert_games

    kickoff = datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc)
    upsert_games([{
        "game_uid": "soccer:test:1", "sport": "soccer", "league_id": "ENG_PL",
        "season": "2526", "game_date": "2026-05-01", "kickoff_utc": kickoff.isoformat(),
        "status": "final", "home_team_uid": "soccer:arsenal",
        "away_team_uid": "soccer:chelsea", "home_score": 2.0, "away_score": 1.0,
        "neutral_site": 0, "venue": None, "source": "test",
        "retrieved_at": kickoff.isoformat(),
    }])

    early = (kickoff - timedelta(hours=3)).isoformat()
    late = (kickoff - timedelta(minutes=45)).isoformat()

    upsert_rows("lineup_observations", _observation(
        "soccer:test:1", "soccer:arsenal", ["Keeper", "A", "B", "C"], early,
        state="projected", goalkeeper="Keeper"))
    upsert_rows("lineup_observations", _observation(
        "soccer:test:1", "soccer:arsenal", ["Keeper", "A", "B", "D"], late,
        state="confirmed", goalkeeper="Keeper"))
    return {"kickoff": kickoff, "early": early, "late": late}


class TestLineupChronology:
    def test_a_lineup_observed_later_is_invisible_earlier(self, lineup_db):
        """The whole guarantee: an 18:15 XI cannot inform a 16:00 prediction."""
        before = pd.Timestamp(lineup_db["early"]) + pd.Timedelta(minutes=30)
        visible = lineup_state_at("soccer:test:1", before.to_pydatetime())

        players = set(visible["player_name"])
        assert "C" in players, "the earlier projected XI should be visible"
        assert "D" not in players, "the later confirmed XI must not be visible yet"
        assert set(visible["lineup_state"]) == {"projected"}

    def test_latest_observation_wins_after_it_is_seen(self, lineup_db):
        after = pd.Timestamp(lineup_db["late"]) + pd.Timedelta(minutes=5)
        visible = lineup_state_at("soccer:test:1", after.to_pydatetime())
        assert "D" in set(visible["player_name"])
        assert set(visible["lineup_state"]) == {"confirmed"}

    def test_final_rows_are_excluded_from_live_reads(self, seeded_db):
        """Rows observed after kick-off describe the past, not the future."""
        from divinelines.db.connection import upsert_rows
        from divinelines.db.repository import upsert_games

        kickoff = datetime(2026, 5, 2, 19, 0, tzinfo=timezone.utc)
        upsert_games([{
            "game_uid": "soccer:test:2", "sport": "soccer", "league_id": "ENG_PL",
            "season": "2526", "game_date": "2026-05-02", "kickoff_utc": kickoff.isoformat(),
            "status": "final", "home_team_uid": "soccer:arsenal",
            "away_team_uid": "soccer:chelsea", "home_score": 1.0, "away_score": 1.0,
            "neutral_site": 0, "venue": None, "source": "test",
            "retrieved_at": kickoff.isoformat(),
        }])
        upsert_rows("lineup_observations", _observation(
            "soccer:test:2", "soccer:arsenal", ["Keeper", "A"],
            (kickoff + timedelta(hours=2)).isoformat(), state="final", goalkeeper="Keeper"))

        live = lineup_state_at("soccer:test:2", datetime.now(timezone.utc), allow_final=False)
        research = lineup_state_at("soccer:test:2", datetime.now(timezone.utc), allow_final=True)
        assert live.empty
        assert not research.empty

    def test_latest_state_reports_the_newest(self, lineup_db):
        assert latest_lineup_state("soccer:test:1") == "confirmed"

    def test_unknown_game_has_unknown_state(self, seeded_db):
        assert latest_lineup_state("soccer:nope") == "unknown"


class TestLineupFeatures:
    def _xi(self, players, goalkeeper):
        return {
            "starters": players, "goalkeeper": goalkeeper,
            "positions": {p: ("goalkeeper" if p == goalkeeper else "midfielder")
                          for p in players},
        }

    def test_no_history_means_no_features_rather_than_zero(self):
        """A team with no observed history must report absence, not a confident 0."""
        builder = LineupFeatureBuilder()
        features = builder.features_for(
            "home", "away", self._xi(["G", "A", "B"], "G"), self._xi(["G2", "C"], "G2")
        )
        assert np.isnan(features["home_xi_regular_share"])
        assert np.isnan(features["diff_missing_regulars"])

    def test_regulars_emerge_from_prior_matches_only(self):
        builder = LineupFeatureBuilder()
        regulars = ["G", "A", "B", "C"]
        for _ in range(REGULAR_WINDOW):
            builder.apply("home", self._xi(regulars, "G"))

        # Same XI: nothing missing.
        features = builder.features_for("home", "away", self._xi(regulars, "G"), None)
        assert features["home_missing_regulars"] == pytest.approx(0.0)
        assert features["home_xi_regular_share"] == pytest.approx(1.0)
        assert features["home_gk_is_regular"] == pytest.approx(1.0)

    def test_missing_regulars_are_counted(self):
        builder = LineupFeatureBuilder()
        for _ in range(REGULAR_WINDOW):
            builder.apply("home", self._xi(["G", "A", "B", "C"], "G"))
        features = builder.features_for("home", "away", self._xi(["G", "A", "X", "Y"], "G"), None)
        assert features["home_missing_regulars"] == pytest.approx(2.0)

    def test_goalkeeper_change_is_detected_and_weighted(self):
        builder = LineupFeatureBuilder()
        for _ in range(REGULAR_WINDOW):
            builder.apply("home", self._xi(["G", "A", "B", "C"], "G"))

        same = builder.features_for("home", "away", self._xi(["G", "A", "B", "C"], "G"), None)
        changed = builder.features_for("home", "away", self._xi(["G2", "A", "B", "C"], "G2"), None)

        assert same["home_gk_is_regular"] == pytest.approx(1.0)
        assert changed["home_gk_is_regular"] == pytest.approx(0.0)
        # The keeper carries a heavier weight than an outfield absence.
        assert changed["home_weighted_missing"] > changed["home_missing_regulars"]

    def test_coverage_flag_reflects_what_was_available(self):
        builder = LineupFeatureBuilder()
        assert builder.features_for("h", "a", None, None)["lineup_coverage"] == 0.0
        assert builder.features_for("h", "a", self._xi(["G"], "G"), None)["lineup_coverage"] == 0.5
        both = builder.features_for("h", "a", self._xi(["G"], "G"), self._xi(["K"], "K"))
        assert both["lineup_coverage"] == 1.0

    def test_attach_reports_coverage_statistics(self, seeded_db):
        dataset = pd.DataFrame([{
            "game_uid": "soccer:test:1", "game_date": pd.Timestamp("2026-05-01"),
            "home_team_uid": "soccer:arsenal", "away_team_uid": "soccer:chelsea",
        }])
        augmented, stats = attach_lineup_features(dataset)
        assert stats["rows"] == 1
        assert "coverage" in stats
        assert "diff_xi_regular_share" in augmented.columns


class TestInformationEvents:
    def _observation(self, players, goalkeeper, state, observed_at):
        from divinelines.sources.espn_lineups import LineupEntry, LineupObservation, TeamLineup

        entries = [
            LineupEntry(player_name=name, external_player_id=None, status="starter",
                        role="G" if name == goalkeeper else "CM",
                        position_group="goalkeeper" if name == goalkeeper else "midfielder",
                        formation_place=str(index + 1))
            for index, name in enumerate(players)
        ]
        return LineupObservation(
            espn_event_id="1", sport="soccer", league_id="ENG_PL",
            teams=[TeamLineup(team_name="Arsenal", home_away="home",
                              formation="4-3-3", entries=entries)],
            lineup_state=state, observed_at=observed_at, retrieved_at=observed_at,
            from_cache=False,
        )

    def test_first_observation_emits_a_lineup_event(self, lineup_db):
        from divinelines.pipeline.ingest_lineups import _diff_information_events

        moment = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
        written = _diff_information_events(
            "soccer:test:1",
            self._observation(["Keeper", "A", "B"], "Keeper", "confirmed", moment),
        )
        assert written >= 1
        events = information_events_for("soccer:test:1")
        assert (events["kind"].str.startswith("LINEUP")).any()

    def test_changed_starters_emit_player_in_and_out(self, seeded_db):
        """A swap between observations should surface as attributable news.

        The differ compares against what is already stored, so the test drives
        the same persist-then-diff order the ingestion pipeline uses.
        """
        from divinelines.db.repository import upsert_games
        from divinelines.pipeline.ingest_lineups import _diff_information_events, _persist

        kickoff = datetime(2026, 5, 5, 19, 0, tzinfo=timezone.utc)
        upsert_games([{
            "game_uid": "soccer:diff:1", "sport": "soccer", "league_id": "ENG_PL",
            "season": "2526", "game_date": "2026-05-05", "kickoff_utc": kickoff.isoformat(),
            "status": "scheduled", "home_team_uid": "soccer:arsenal",
            "away_team_uid": "soccer:chelsea", "home_score": None, "away_score": None,
            "neutral_site": 0, "venue": None, "source": "test",
            "retrieved_at": kickoff.isoformat(),
        }])

        base = kickoff - timedelta(hours=3)
        first = self._observation(["Keeper", "A", "B"], "Keeper", "projected", base)
        _persist("soccer:diff:1", first)
        _diff_information_events("soccer:diff:1", first)

        second = self._observation(["Keeper", "A", "Z"], "Keeper", "confirmed",
                                   base + timedelta(hours=1))
        _persist("soccer:diff:1", second)
        _diff_information_events("soccer:diff:1", second)

        kinds = set(information_events_for("soccer:diff:1")["kind"])
        assert "PLAYER_IN" in kinds
        assert "PLAYER_OUT" in kinds


class TestRoiInterval:
    def test_small_sample_refuses_an_interval(self):
        result = roi_interval([10] * 5, [1] * 5)
        assert result["ci_low"] is None
        assert "too small" in result["interpretation"]

    def test_break_even_is_reported_as_break_even(self):
        rng = np.random.default_rng(0)
        stakes = [10.0] * 400
        profits = rng.choice([9.0, -10.0], size=400, p=[0.526, 0.474]).tolist()
        result = roi_interval(stakes, profits)
        assert result["significant"] is False
        assert "break-even" in result["interpretation"]

    def test_clear_edge_is_detected(self):
        stakes = [10.0] * 500
        profits = [1.0] * 500          # a riskless 10% return
        result = roi_interval(stakes, profits)
        assert result["significant"] is True
        assert result["ci_low"] > 0

    def test_no_stakes(self):
        assert roi_interval([], [])["roi"] is None


@pytest.fixture()
def graded_ledger(seeded_db):
    """Predictions on finished games, with market probabilities attached."""
    from divinelines.betting.ledger import PredictionRecord, record_predictions
    from divinelines.db.connection import query_df
    from divinelines.models.registry import ModelRecord, register

    register(ModelRecord(model_id="m1", sport="nba", kind="ensemble", model_version="v1",
                         feature_set=[], feature_set_version="t"))
    games = query_df(
        "SELECT game_uid, home_score, away_score FROM games "
        "WHERE sport='nba' AND status='final' ORDER BY game_date"
    )

    def build(probability_for_home, market_probability):
        records = []
        for _, row in games.iterrows():
            records.append(PredictionRecord(
                sport="nba", game_uid=row["game_uid"], market="h2h", selection="home",
                model_probability=probability_for_home(row),
                market_probability=market_probability, price_decimal=2.0,
                model_id="m1", model_version="v1", mode="backtest",
            ))
        return records

    return {"games": games, "build": build, "record": record_predictions}


class TestModelHealth:
    def test_small_sample_is_insufficient(self, graded_ledger):
        graded_ledger["record"](graded_ledger["build"](lambda row: 0.6, 0.5))
        result = compute_health("nba")
        assert result.status == STATUS_INSUFFICIENT
        assert str(MIN_SAMPLE_FOR_STATUS) in result.status_reason

    def test_no_predictions_reports_no_data(self, seeded_db):
        result = compute_health("nba")
        assert result.sample_size == 0
        assert "no graded predictions" in result.status_reason

    def test_market_comparison_uses_identical_rows(self, graded_ledger):
        graded_ledger["record"](graded_ledger["build"](lambda row: 0.6, 0.55))
        result = compute_health("nba")
        assert result.market_comparison["n"] == result.sample_size

    def test_status_thresholds_are_documented_numbers(self):
        """Statuses must come from stated thresholds, not adjustable taste."""
        from divinelines.analytics.model_health import (
            MIN_BRIER_SKILL,
            MIN_MARKET_EDGE,
            MIN_SAMPLE_FOR_MARKET_CLAIM,
        )

        assert MIN_BRIER_SKILL > 0
        assert MIN_MARKET_EDGE > 0
        assert MIN_SAMPLE_FOR_MARKET_CLAIM >= 100

    def test_regression_needs_a_real_sample(self, seeded_db):
        result = detect_regression("nba")
        assert result["regression"] is False
        assert "too small" in result["reason"] or "log loss" in result["reason"]

    def test_stability_needs_multiple_versions(self, seeded_db):
        result = prediction_stability("nba")
        assert result["n_series"] == 0


class TestPredictionVersioning:
    def test_superseding_marks_older_rows(self, seeded_db):
        from divinelines.betting.ledger import (
            PredictionRecord,
            record_predictions,
            supersede_predictions,
        )
        from divinelines.db.connection import query_df
        from divinelines.models.registry import ModelRecord, register

        register(ModelRecord(model_id="m", sport="nba", kind="ensemble", model_version="v1",
                             feature_set=[], feature_set_version="t"))
        record_predictions([PredictionRecord(
            sport="nba", game_uid="nba:upcoming", market="h2h", selection="home",
            model_probability=0.52, model_id="m", model_version="v1",
            created_at="2026-08-20T10:00:00+00:00", prediction_stage="pre_lineup",
        )])
        record_predictions([PredictionRecord(
            sport="nba", game_uid="nba:upcoming", market="h2h", selection="home",
            model_probability=0.47, model_id="m", model_version="v1",
            created_at="2026-08-20T18:40:00+00:00", prediction_stage="confirmed_lineup",
            lineup_state="confirmed",
        )])
        superseded = supersede_predictions("nba:upcoming", "h2h",
                                           before="2026-08-20T18:40:00+00:00")

        rows = query_df("SELECT * FROM predictions ORDER BY created_at")
        assert superseded == 1
        assert len(rows) == 2, "the earlier prediction must survive, not be overwritten"
        assert pd.notna(rows.iloc[0]["superseded_at"])
        assert pd.isna(rows.iloc[1]["superseded_at"])

    def test_both_versions_remain_queryable(self, seeded_db):
        from divinelines.analytics.timeline import prediction_versions
        from divinelines.betting.ledger import PredictionRecord, record_predictions
        from divinelines.models.registry import ModelRecord, register

        register(ModelRecord(model_id="m", sport="nba", kind="ensemble", model_version="v1",
                             feature_set=[], feature_set_version="t"))
        for stamp, probability, state in (
            ("2026-08-20T10:00:00+00:00", 0.52, "unknown"),
            ("2026-08-20T18:40:00+00:00", 0.47, "confirmed"),
        ):
            record_predictions([PredictionRecord(
                sport="nba", game_uid="nba:upcoming", market="h2h", selection="home",
                model_probability=probability, model_id="m", model_version="v1",
                created_at=stamp, lineup_state=state,
            )])
        versions = prediction_versions("nba:upcoming")
        assert len(versions) == 2
        assert list(versions["model_prob"]) == pytest.approx([0.52, 0.47])
