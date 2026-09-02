"""Models, calibration, backtesting and settlement.

The backtest tests exist to prove two properties that are easy to break and
expensive to get wrong: a model never trains on the period it is scored on,
and a simulated bet is never struck at the closing price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from divinelines.backtest.metrics import BetResult, bucket_analysis, summarise
from divinelines.backtest.walkforward import BettingPolicy, simulate_market, walk_forward
from divinelines.models.calibration import (
    Calibrator,
    brier_score,
    calibration_quality,
    evaluate_probabilities,
    expected_calibration_error,
    log_loss_score,
    multiclass_brier,
    multiclass_log_loss,
    reliability_curve,
)
from divinelines.models.nba_model import (
    DEFAULT_VARIANT,
    FEATURE_VARIANTS,
    chronological_split,
    resolve_features,
)
from divinelines.models.nba_player_impact import (
    AvailabilityAdjustment,
    apply_margin_adjustment,
    build_player_impacts,
    scenario_weighted_probability,
    team_availability,
)
from divinelines.models.soccer_model import DixonColesConfig, DixonColesModel, TemperatureCalibrator


class TestCalibrationMetrics:
    def test_brier_is_mean_squared_error(self):
        assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
        assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)

    def test_log_loss_rewards_confident_correct_calls(self):
        assert log_loss_score([1], [0.9]) < log_loss_score([1], [0.6])

    def test_log_loss_punishes_confident_wrong_calls(self):
        assert log_loss_score([1], [0.01]) > log_loss_score([1], [0.4])

    def test_perfect_calibration_has_no_error(self):
        y_true = [1] * 70 + [0] * 30
        y_prob = [0.7] * 100
        assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)

    def test_overconfidence_is_detected(self):
        y_true = [1] * 50 + [0] * 50
        y_prob = [0.9] * 100
        assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.4, abs=1e-6)

    def test_brier_skill_is_zero_for_the_base_rate(self):
        y_true = np.array([1] * 55 + [0] * 45)
        metrics = evaluate_probabilities(y_true, np.full(100, 0.55))
        assert metrics.brier_skill == pytest.approx(0.0, abs=1e-9)

    def test_reliability_curve_bins_carry_counts(self):
        curve = reliability_curve([1, 0, 1, 1], [0.9, 0.1, 0.8, 0.85])
        assert sum(point["count"] for point in curve) == 4

    def test_multiclass_metrics(self):
        probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        assert multiclass_log_loss([0, 1], probabilities) < 0.4
        assert multiclass_brier([0, 1], probabilities) < 0.2


class TestCalibrator:
    def test_isotonic_fixes_systematic_overconfidence(self):
        rng = np.random.default_rng(0)
        truth = rng.random(2000)
        y_true = (rng.random(2000) < truth).astype(int)
        # Push probabilities toward the extremes: classic overconfidence.
        raw = np.clip(truth * 1.4 - 0.2, 0.01, 0.99)

        calibrator = Calibrator("isotonic").fit(raw[:1000], y_true[:1000])
        calibrated = calibrator.transform(raw[1000:])
        before = expected_calibration_error(y_true[1000:], raw[1000:])
        after = expected_calibration_error(y_true[1000:], calibrated)
        assert after < before

    def test_platt_also_works(self):
        rng = np.random.default_rng(1)
        truth = rng.random(1500)
        y_true = (rng.random(1500) < truth).astype(int)
        raw = np.clip(truth * 1.3 - 0.15, 0.01, 0.99)
        calibrator = Calibrator("platt").fit(raw[:800], y_true[:800])
        assert calibrator.transform(raw[800:]).max() <= 0.995

    def test_small_samples_pass_through_rather_than_overfit(self):
        calibrator = Calibrator("isotonic").fit([0.4, 0.6], [0, 1])
        assert calibrator.method == "none"
        assert calibrator.transform([0.4])[0] == pytest.approx(0.4)

    def test_transform_before_fit_is_an_error(self):
        with pytest.raises(RuntimeError):
            Calibrator().transform([0.5])

    def test_quality_score_falls_as_error_rises(self):
        good = evaluate_probabilities([1] * 70 + [0] * 30, [0.7] * 100)
        bad = evaluate_probabilities([1] * 50 + [0] * 50, [0.95] * 100)
        assert calibration_quality(good) > calibration_quality(bad)


class TestSplitting:
    def test_split_is_strictly_chronological(self):
        frame = pd.DataFrame({
            "game_date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "home_win": [0, 1] * 50,
        })
        split = chronological_split(frame, valid_fraction=0.2, test_fraction=0.2)
        assert split.train["game_date"].max() <= split.valid["game_date"].min()
        assert split.valid["game_date"].max() <= split.test["game_date"].min()
        assert len(split.train) + len(split.valid) + len(split.test) == 100

    def test_feature_variants_resolve_to_known_columns(self):
        for variant in FEATURE_VARIANTS:
            assert resolve_features(variant)

    def test_default_variant_is_a_real_variant(self):
        assert DEFAULT_VARIANT in FEATURE_VARIANTS

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            resolve_features("does_not_exist")

    def test_missing_columns_are_dropped_not_faked(self):
        resolved = resolve_features("ratings" and ["ratings"], available=["diff_elo"])
        assert resolved == ["diff_elo"]


class TestDixonColes:
    def test_recovers_relative_team_strength(self, soccer_matches):
        model = DixonColesModel(DixonColesConfig(xi=0.0, l2=0.005)).fit(soccer_matches)
        strengths = model.team_strengths().set_index("team_uid")
        # club0 was generated strongest, club7 weakest.
        assert strengths.loc["soccer:club0", "attack"] > strengths.loc["soccer:club7", "attack"]

    def test_probabilities_sum_to_one(self, soccer_matches):
        model = DixonColesModel().fit(soccer_matches)
        prediction = model.predict_match("soccer:club0", "soccer:club7", "ENG_PL")
        total = prediction["home"] + prediction["draw"] + prediction["away"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_draw_probability_is_material(self, soccer_matches):
        model = DixonColesModel().fit(soccer_matches)
        prediction = model.predict_match("soccer:club3", "soccer:club4", "ENG_PL")
        assert 0.12 < prediction["draw"] < 0.45

    def test_recovers_known_parameters(self, dixon_coles_ground_truth):
        """Fit a league generated from known parameters and get them back."""
        matches, truth = dixon_coles_ground_truth
        model = DixonColesModel(DixonColesConfig(xi=0.0, l2=0.0)).fit(matches)
        fit = model.fit_result

        assert fit.home_advantage == pytest.approx(truth["home_advantage"], abs=0.05)
        teams = sorted(truth["attack"])
        attack_correlation = np.corrcoef(
            [truth["attack"][t] for t in teams], [fit.attack[t] for t in teams]
        )[0, 1]
        defence_correlation = np.corrcoef(
            [truth["defence"][t] for t in teams], [fit.defence[t] for t in teams]
        )[0, 1]
        assert attack_correlation > 0.95
        assert defence_correlation > 0.95

    def test_home_advantage_is_positive(self, dixon_coles_ground_truth):
        matches, _ = dixon_coles_ground_truth
        assert DixonColesModel().fit(matches).fit_result.home_advantage > 0

    def test_stronger_team_is_favoured(self, soccer_matches):
        model = DixonColesModel().fit(soccer_matches)
        strong_home = model.predict_match("soccer:club0", "soccer:club7", "ENG_PL")
        weak_home = model.predict_match("soccer:club7", "soccer:club0", "ENG_PL")
        assert strong_home["home"] > weak_home["home"]

    def test_totals_are_consistent_with_the_score_matrix(self, soccer_matches):
        model = DixonColesModel().fit(soccer_matches)
        prediction = model.predict_match("soccer:club2", "soccer:club5", "ENG_PL")
        totals = prediction["totals"]
        assert totals["over_2.5"] + totals["under_2.5"] == pytest.approx(1.0, abs=1e-6)
        assert totals["over_1.5"] > totals["over_2.5"] > totals["over_3.5"]

    def test_unknown_club_is_treated_as_league_average(self, soccer_matches):
        model = DixonColesModel().fit(soccer_matches)
        prediction = model.predict_match("soccer:promoted-club", "soccer:club7", "ENG_PL")
        assert 0.0 < prediction["home"] < 1.0
        assert prediction["expected_home_goals"] > 0

    def test_requires_results(self):
        with pytest.raises(ValueError):
            DixonColesModel().fit(pd.DataFrame({
                "home_team_uid": ["a"], "away_team_uid": ["b"],
                "home_score": [np.nan], "away_score": [np.nan],
                "game_date": [pd.Timestamp("2024-01-01")],
            }))

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError):
            DixonColesModel().fit(pd.DataFrame({"home_team_uid": ["a"]}))

    def test_temperature_calibrator_keeps_rows_normalised(self):
        rng = np.random.default_rng(3)
        probabilities = rng.dirichlet([2, 2, 2], size=500)
        outcomes = np.array([rng.choice(3, p=row) for row in probabilities])
        calibrated = TemperatureCalibrator().fit(probabilities, outcomes).transform(probabilities)
        assert np.allclose(calibrated.sum(axis=1), 1.0)


class TestWalkForward:
    def test_a_model_never_sees_the_period_it_predicts(self):
        frame = pd.DataFrame({
            "game_date": pd.date_range("2022-01-01", periods=900, freq="D"),
            "season": ["A"] * 300 + ["B"] * 300 + ["C"] * 300,
            "value": range(900),
        })
        seen: list[tuple[str, pd.Timestamp]] = []

        def fit_predict(train, valid, test):
            seen.append((test["season"].iloc[0], train["game_date"].max()))
            assert train["game_date"].max() < test["game_date"].min()
            assert valid["game_date"].max() < test["game_date"].min()
            return pd.DataFrame({"model_probability": [0.5] * len(test)})

        result = walk_forward(frame, fit_predict, min_train_rows=100)
        assert [season for season, _ in seen] == ["B", "C"]
        assert len(result.folds) == 2

    def test_periods_without_enough_history_are_skipped(self):
        frame = pd.DataFrame({
            "game_date": pd.date_range("2022-01-01", periods=20, freq="D"),
            "season": ["A"] * 10 + ["B"] * 10,
        })
        result = walk_forward(
            frame, lambda train, valid, test: pd.DataFrame({"model_probability": [0.5] * len(test)}),
            min_train_rows=500,
        )
        assert result.predictions.empty


class TestMarketSimulation:
    def _frame(self):
        return pd.DataFrame([
            {
                "game_uid": "g1", "game_date": pd.Timestamp("2024-05-01"), "league_id": "ENG_PL",
                "outcome_selection": "home",
                "odds_open_home": 2.60, "odds_open_draw": 3.40, "odds_open_away": 2.80,
                "odds_close_home": 2.20, "odds_close_draw": 3.40, "odds_close_away": 3.20,
                "prob_home": 0.55, "prob_draw": 0.25, "prob_away": 0.20,
            }
        ])

    def _columns(self, phase):
        return {s: f"odds_{phase}_{s}" for s in ("home", "draw", "away")}

    def test_bets_are_struck_at_the_decision_time_price(self):
        bets, _ = simulate_market(
            self._frame(), price_columns=self._columns("open"),
            closing_columns=self._columns("close"), policy=BettingPolicy(min_edge=0.02),
        )
        assert len(bets) == 1
        assert bets[0].price_decimal == 2.60          # opening, not closing
        assert bets[0].closing_price == 2.20          # closing used only for CLV

    def test_clv_reflects_a_shortening_price(self):
        bets, _ = simulate_market(
            self._frame(), price_columns=self._columns("open"),
            closing_columns=self._columns("close"), policy=BettingPolicy(min_edge=0.02),
        )
        assert bets[0].clv_price_pct > 0

    def test_winning_bet_profit_matches_the_price(self):
        bets, metrics = simulate_market(
            self._frame(), price_columns=self._columns("open"),
            policy=BettingPolicy(min_edge=0.02),
        )
        bet = bets[0]
        assert bet.won is True
        assert bet.profit == pytest.approx(bet.stake * (bet.price_decimal - 1))
        assert metrics.bets == 1

    def test_edge_threshold_is_respected(self):
        bets, _ = simulate_market(
            self._frame(), price_columns=self._columns("open"),
            policy=BettingPolicy(min_edge=0.90),
        )
        assert bets == []

    def test_extreme_disagreement_is_rejected_as_a_model_outlier(self):
        frame = self._frame()
        frame.loc[0, "prob_home"] = 0.99
        frame.loc[0, "prob_draw"] = 0.005
        frame.loc[0, "prob_away"] = 0.005
        bets, _ = simulate_market(
            frame, price_columns=self._columns("open"),
            policy=BettingPolicy(min_edge=0.02, max_disagreement=0.25),
        )
        assert bets == []

    def test_consensus_book_can_differ_from_the_price_taken(self):
        frame = self._frame()
        frame["best_home"] = 3.10
        frame["best_draw"] = 3.40
        frame["best_away"] = 2.80
        bets, _ = simulate_market(
            frame,
            price_columns={"home": "best_home", "draw": "best_draw", "away": "best_away"},
            consensus_columns=self._columns("open"),
            policy=BettingPolicy(min_edge=0.02),
        )
        assert bets[0].price_decimal == 3.10
        # Fair probability still comes from the consensus book.
        assert bets[0].market_probability == pytest.approx(
            simulate_market(self._frame(), price_columns=self._columns("open"),
                            policy=BettingPolicy(min_edge=0.02))[0][0].market_probability
        )


class TestBacktestMetrics:
    def _bet(self, won, stake=10.0, price=2.0, edge=0.05):
        return BetResult(
            game_uid="g", date=pd.Timestamp("2024-01-01"), league_id="NBA", market="h2h",
            selection="home", price_decimal=price, stake=stake, model_probability=0.55,
            market_probability=0.50, edge=edge, won=won,
        )

    def test_roi_and_profit(self):
        metrics = summarise([self._bet(True), self._bet(False)], starting_bankroll=1000)
        assert metrics.profit == pytest.approx(0.0)
        assert metrics.roi == pytest.approx(0.0)
        assert metrics.hit_rate == pytest.approx(0.5)

    def test_push_returns_the_stake(self):
        bet = self._bet(False)
        bet.push = True
        assert bet.profit == 0.0

    def test_drawdown_is_negative_after_a_loss(self):
        metrics = summarise([self._bet(False), self._bet(True)], starting_bankroll=1000)
        assert metrics.max_drawdown < 0

    def test_empty_backtest_is_reported_not_crashed(self):
        metrics = summarise([], starting_bankroll=500)
        assert metrics.bets == 0
        assert metrics.final_bankroll == 500

    def test_bucket_analysis_groups_by_edge(self):
        buckets = bucket_analysis(
            [self._bet(True, edge=0.01), self._bet(False, edge=0.08)], by="edge"
        )
        assert len(buckets) == 2
        assert "roi" in buckets.columns


class TestPlayerImpact:
    def _stats(self):
        return pd.DataFrame([
            {"PLAYER_NAME": "Star Player", "TEAM_ABBREVIATION": "BOS", "MIN": 36.0,
             "PIE": 0.20, "GP": 60, "USG_PCT": 0.31, "NET_RATING": 8.0},
            {"PLAYER_NAME": "Bench Guy", "TEAM_ABBREVIATION": "BOS", "MIN": 9.0,
             "PIE": 0.06, "GP": 40, "USG_PCT": 0.15, "NET_RATING": -2.0},
            {"PLAYER_NAME": "Small Sample", "TEAM_ABBREVIATION": "BOS", "MIN": 30.0,
             "PIE": 0.30, "GP": 2, "USG_PCT": 0.30, "NET_RATING": 20.0},
        ])

    class _Status:
        def __init__(self, name, status, probability):
            self.player_name = name
            self.status = status
            self.play_probability = probability

    def test_star_is_worth_far_more_than_a_bench_player(self):
        impacts = build_player_impacts(self._stats())
        assert impacts["star player"].margin_impact > 4 * impacts["bench guy"].margin_impact

    def test_tiny_samples_are_excluded(self):
        impacts = build_player_impacts(self._stats())
        assert "small sample" not in impacts

    def test_missing_star_lowers_expected_margin(self):
        impacts = build_player_impacts(self._stats())
        adjustment = team_availability(
            "nba:BOS", [self._Status("Star Player", "out", 0.0)], impacts
        )
        assert adjustment.expected_margin_delta < -3
        assert adjustment.missing_players[0]["player"] == "Star Player"
        assert adjustment.uncertainty == 0.0

    def test_questionable_players_produce_partial_impact_and_uncertainty(self):
        impacts = build_player_impacts(self._stats())
        adjustment = team_availability(
            "nba:BOS", [self._Status("Star Player", "questionable", 0.5)], impacts
        )
        certain = team_availability(
            "nba:BOS", [self._Status("Star Player", "out", 0.0)], impacts
        )
        assert certain.expected_margin_delta < adjustment.expected_margin_delta < 0
        assert adjustment.uncertainty > 0

    def test_margin_adjustment_moves_probability_the_right_way(self):
        assert apply_margin_adjustment(0.5, -3.0, 0.1) < 0.5
        assert apply_margin_adjustment(0.5, 3.0, 0.1) > 0.5
        assert apply_margin_adjustment(0.5, 0.0, 0.1) == pytest.approx(0.5)

    def test_probability_stays_in_range_for_extreme_adjustments(self):
        assert 0.0 < apply_margin_adjustment(0.5, -500.0, 0.1) < 1.0
        assert 0.0 < apply_margin_adjustment(0.5, 500.0, 0.1) < 1.0

    def test_scenario_weighting_reports_a_range(self):
        impacts = build_player_impacts(self._stats())
        home = team_availability("nba:BOS", [self._Status("Star Player", "questionable", 0.5)],
                                 impacts)
        away = AvailabilityAdjustment("nba:LAL", 0.0, 0.0, 0.0)
        result = scenario_weighted_probability(0.60, home, away, 0.1)
        low, high = result["probability_range"]
        assert low < result["adjusted_probability"] < high
        assert result["adjusted_probability"] < 0.60
