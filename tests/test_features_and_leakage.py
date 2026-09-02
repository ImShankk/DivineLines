"""Feature engineering, with leakage as the headline concern.

The decisive test here is ``test_features_are_identical_when_the_future_is_removed``:
if a row's features change when later games are deleted, the builder is
reading the future.  That single property is worth more than any number of
spot checks on individual columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from divinelines.features.nba_features import (
    NbaFeatureBuilder,
    NbaFeatureConfig,
    build_nba_dataset,
    compute_game_metrics,
    haversine_km,
)
from divinelines.features.ratings import (
    AdjustedEfficiency,
    EloConfig,
    EloRatings,
    shrink_to_prior,
)
from divinelines.features.soccer_features import build_soccer_dataset, match_metrics


class TestNbaFeatureLeakage:
    def test_features_are_identical_when_the_future_is_removed(self, nba_team_games):
        """The strongest guarantee: no row may depend on a later game."""
        full, _ = build_nba_dataset(nba_team_games, NbaFeatureConfig(min_prior_games=0))
        cutoff = full["game_date"].iloc[len(full) // 2]
        truncated_input = nba_team_games[nba_team_games["game_date"] <= cutoff]
        truncated, _ = build_nba_dataset(truncated_input, NbaFeatureConfig(min_prior_games=0))

        shared = full[full["game_uid"].isin(truncated["game_uid"])].reset_index(drop=True)
        truncated = truncated.reset_index(drop=True)
        feature_columns = [
            column for column in shared.columns
            if column.startswith(("diff_", "elo_", "adj_", "home_", "away_", "h2h_"))
            and column not in ("home_score", "away_score")
        ]
        pd.testing.assert_frame_equal(
            shared[feature_columns], truncated[feature_columns], check_exact=False, rtol=1e-9
        )

    def test_target_is_never_a_feature(self, nba_team_games):
        dataset, _ = build_nba_dataset(nba_team_games, NbaFeatureConfig(min_prior_games=0))
        feature_columns = [c for c in dataset.columns if c.startswith("diff_")]
        checked = 0
        for column in feature_columns:
            values = dataset[column].dropna()
            if len(values) < 5 or values.nunique() < 2:
                continue  # a constant column cannot correlate with anything
            correlation = dataset[column].corr(dataset["home_win"])
            if pd.isna(correlation):
                continue
            checked += 1
            assert abs(correlation) < 0.999, f"{column} is a copy of the target"
        assert checked > 0, "no usable feature columns were checked"

    def test_first_game_has_no_prior_form(self, nba_team_games):
        builder = NbaFeatureBuilder(NbaFeatureConfig(min_prior_games=0))
        dataset = builder.build(nba_team_games)
        first = dataset.iloc[0]
        assert pd.isna(first["diff_net_rating_r5"])
        assert first["h2h_games"] == 0

    def test_h2h_is_absent_rather_than_invented(self, nba_team_games):
        """The v1 pipeline hard-coded H2H to 0.50; absence must read as NaN."""
        builder = NbaFeatureBuilder(NbaFeatureConfig(min_prior_games=0))
        dataset = builder.build(nba_team_games)
        first = dataset.iloc[0]
        assert np.isnan(first["h2h_home_win_pct"])
        assert (dataset["h2h_home_win_pct"].dropna() != 0.5).any()

    def test_season_rollover_clears_current_season_form(self, nba_team_games):
        builder = NbaFeatureBuilder(NbaFeatureConfig(min_prior_games=0))
        dataset = builder.build(nba_team_games)
        new_season = dataset[dataset["season"] == "2025-26"]
        assert not new_season.empty
        first_of_season = new_season.iloc[0]
        assert first_of_season["home_season_games"] == 0
        assert first_of_season["away_season_games"] == 0

    def test_upcoming_features_match_the_training_path(self, nba_team_games):
        builder = NbaFeatureBuilder(NbaFeatureConfig(min_prior_games=0))
        builder.build(nba_team_games)
        features = builder.upcoming_features(
            "nba:BOS", "nba:LAL", pd.Timestamp("2026-01-05"), "2025-26"
        )
        assert "diff_elo" in features
        assert features["home_win"] is None
        assert 0.0 < features["elo_home_prob"] < 1.0


class TestGameMetrics:
    def test_possessions_and_ratings(self):
        row = pd.Series({"fga": 90.0, "fta": 20.0, "oreb": 10.0, "tov": 12.0, "fgm": 40.0,
                         "fg3m": 10.0, "fg3a": 30.0, "dreb": 33.0, "pts": 110.0, "min": 240.0})
        opponent = row.copy()
        opponent["pts"] = 100.0
        metrics = compute_game_metrics(row, opponent)
        expected_possessions = 90 - 10 + 12 + 0.44 * 20
        assert metrics["poss"] == pytest.approx(expected_possessions)
        assert metrics["ortg"] == pytest.approx(100 * 110 / expected_possessions)
        assert metrics["drtg"] == pytest.approx(100 * 100 / expected_possessions)
        assert metrics["net_rating"] == pytest.approx(metrics["ortg"] - metrics["drtg"])
        assert metrics["win"] == 1.0

    def test_efg_and_ts_are_bounded(self):
        row = pd.Series({"fga": 88.0, "fta": 22.0, "oreb": 10.0, "tov": 13.0, "fgm": 40.0,
                         "fg3m": 12.0, "fg3a": 34.0, "dreb": 33.0, "pts": 110.0, "min": 240.0})
        metrics = compute_game_metrics(row, row)
        assert 0.3 < metrics["efg"] < 0.8
        assert 0.4 < metrics["ts"] < 0.8

    def test_zero_attempts_do_not_explode(self):
        row = pd.Series({"fga": 0.0, "fta": 0.0, "oreb": 0.0, "tov": 0.0, "fgm": 0.0,
                         "fg3m": 0.0, "fg3a": 0.0, "dreb": 0.0, "pts": 0.0, "min": 240.0})
        metrics = compute_game_metrics(row, row)
        assert metrics["efg"] == 0.0 and metrics["ts"] == 0.0


class TestElo:
    def test_ratings_are_zero_sum(self):
        elo = EloRatings(EloConfig(use_mov=False))
        elo.update("A", "B", 110, 100)
        assert elo.rating("A") + elo.rating("B") == pytest.approx(3000.0)

    def test_winning_raises_and_losing_lowers(self):
        elo = EloRatings(EloConfig(use_mov=False))
        before = elo.rating("A")
        elo.update("A", "B", 110, 100)
        assert elo.rating("A") > before
        assert elo.rating("B") < before

    def test_home_advantage_moves_the_expected_score(self):
        elo = EloRatings(EloConfig(home_advantage=60))
        assert elo.expected_home_score("A", "B") > 0.5
        assert elo.expected_home_score("A", "B", neutral=True) == pytest.approx(0.5)

    def test_season_regression_pulls_toward_the_mean(self):
        elo = EloRatings(EloConfig(season_carry=0.5, use_mov=False))
        elo.start_season("2024")
        for _ in range(30):
            elo.update("A", "B", 120, 100)
        elevated = elo.rating("A")
        elo.start_season("2025")
        assert elo.rating("A") < elevated
        assert elo.rating("A") == pytest.approx(1500 + 0.5 * (elevated - 1500))

    def test_three_way_probabilities_sum_to_one(self):
        elo = EloRatings(EloConfig(draws_possible=True))
        probabilities = elo.win_probabilities("A", "B")
        assert set(probabilities) == {"home", "draw", "away"}
        assert sum(probabilities.values()) == pytest.approx(1.0)

    def test_margin_of_victory_dampens_blowouts(self):
        plain = EloRatings(EloConfig(use_mov=False))
        mov = EloRatings(EloConfig(use_mov=True))
        plain.update("A", "B", 130, 100)
        mov.update("A", "B", 130, 100)
        assert mov.rating("A") != plain.rating("A")


class TestAdjustedEfficiency:
    def test_learns_that_one_team_is_stronger(self):
        adjusted = AdjustedEfficiency()
        for _ in range(60):
            adjusted.update_game("strong", "weak", 125.0, 100.0)
        assert adjusted.get("strong").offense > adjusted.get("weak").offense
        assert adjusted.get("strong").net > adjusted.get("weak").net

    def test_ignores_non_finite_values(self):
        adjusted = AdjustedEfficiency()
        adjusted.update("A", "B", float("nan"))
        assert adjusted.get("A").games == 0


class TestShrinkage:
    def test_prior_dominates_with_no_evidence(self):
        assert shrink_to_prior(130.0, 0, 110.0, prior_games=10) == pytest.approx(110.0)

    def test_evidence_takes_over_as_games_accumulate(self):
        early = shrink_to_prior(130.0, 2, 110.0, prior_games=10)
        late = shrink_to_prior(130.0, 40, 110.0, prior_games=10)
        assert 110.0 < early < late < 130.0

    def test_handles_missing_values(self):
        assert shrink_to_prior(None, 5, 110.0) == 110.0
        assert shrink_to_prior(120.0, 5, None) == 120.0
        assert shrink_to_prior(None, 0, None) is None


class TestTravel:
    def test_known_distance(self):
        # Boston -> Los Angeles is roughly 4,170 km.
        distance = haversine_km(42.3662, -71.0621, 34.0430, -118.2673)
        assert 4000 < distance < 4300

    def test_same_venue_is_zero(self):
        assert haversine_km(42.0, -71.0, 42.0, -71.0) == pytest.approx(0.0)


class TestSoccerFeatures:
    def test_builds_three_way_outcomes(self, soccer_matches):
        dataset, _ = build_soccer_dataset(soccer_matches)
        assert set(dataset["outcome_selection"].dropna()) <= {"home", "draw", "away"}
        assert dataset["outcome"].isin([0, 1, 2]).all()

    def test_no_leakage_when_the_future_is_removed(self, soccer_matches):
        full, _ = build_soccer_dataset(soccer_matches)
        cutoff = full["game_date"].iloc[len(full) // 2]
        truncated, _ = build_soccer_dataset(
            soccer_matches[soccer_matches["game_date"] <= cutoff]
        )
        shared = full[full["game_uid"].isin(truncated["game_uid"])].reset_index(drop=True)
        columns = [c for c in shared.columns if c.startswith(("diff_", "elo_", "h2h_"))]
        pd.testing.assert_frame_equal(
            shared[columns], truncated.reset_index(drop=True)[columns],
            check_exact=False, rtol=1e-9,
        )

    def test_new_clubs_are_flagged(self, soccer_matches):
        dataset, _ = build_soccer_dataset(soccer_matches)
        assert dataset["home_is_new_to_league"].iloc[0] == 1
        assert dataset["home_is_new_to_league"].iloc[-1] == 0

    def test_league_home_advantage_is_learned_not_fixed(self, soccer_matches):
        dataset, _ = build_soccer_dataset(soccer_matches)
        values = dataset["league_home_advantage"].dropna().unique()
        assert len(values) > 1

    def test_match_metrics_award_points_correctly(self):
        row = pd.Series({"home_shots": 12, "away_shots": 8, "home_sot": 5, "away_sot": 3,
                         "home_corners": 6, "away_corners": 3, "home_yellow": 1,
                         "away_yellow": 2, "home_red": 0, "away_red": 1})
        assert match_metrics(2, 1, row, "home")["points"] == 3.0
        assert match_metrics(1, 1, row, "home")["points"] == 1.0
        assert match_metrics(0, 1, row, "home")["points"] == 0.0

    def test_shot_accuracy_is_a_ratio_not_xg(self):
        row = pd.Series({"home_shots": 10, "away_shots": 8, "home_sot": 4, "away_sot": 3})
        metrics = match_metrics(2, 1, row, "home")
        assert metrics["shot_accuracy"] == pytest.approx(0.4)
        assert metrics["conversion"] == pytest.approx(0.5)
