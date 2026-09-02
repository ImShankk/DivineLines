"""Odds conversion, margin removal, EV, Kelly and portfolio risk.

These are the calculations that decide how much money a recommendation
implies, so they are tested against closed-form expectations rather than
against whatever the code currently returns.
"""

from __future__ import annotations

import math

import pytest

from divinelines.betting.clv import closing_line_value, summarise_clv
from divinelines.betting.ev import compute_edge_score, expected_value, market_liquidity_proxy
from divinelines.betting.kelly import kelly_fraction, recommend_stake, shrink_probability
from divinelines.betting.odds_math import (
    american_to_decimal,
    build_consensus,
    decimal_to_american,
    implied_probability,
    overround,
    probability_to_decimal,
    remove_vig,
    remove_vig_multiplicative,
    remove_vig_power,
)
from divinelines.betting.portfolio import Candidate, build_portfolio, correlation_warnings


class TestOddsConversion:
    @pytest.mark.parametrize(
        "american,decimal",
        [(100, 2.0), (150, 2.5), (-200, 1.5), (-110, 1.909090909), (250, 3.5)],
    )
    def test_american_to_decimal(self, american, decimal):
        assert american_to_decimal(american) == pytest.approx(decimal, abs=1e-6)

    @pytest.mark.parametrize("american", [-500, -200, -110, 100, 150, 900])
    def test_round_trip(self, american):
        assert decimal_to_american(american_to_decimal(american)) == pytest.approx(american)

    def test_implied_probability(self):
        assert implied_probability(2.0) == pytest.approx(0.5)
        assert implied_probability(4.0) == pytest.approx(0.25)

    def test_probability_to_decimal_is_the_inverse(self):
        assert probability_to_decimal(implied_probability(3.2)) == pytest.approx(3.2)

    def test_rejects_impossible_prices(self):
        with pytest.raises(ValueError):
            implied_probability(1.0)
        with pytest.raises(ValueError):
            american_to_decimal(0)
        with pytest.raises(ValueError):
            probability_to_decimal(0.0)

    def test_overround_is_positive_for_a_real_book(self):
        assert overround([1.90, 1.90]) == pytest.approx(0.0526, abs=1e-3)


class TestMarginRemoval:
    def test_fair_probabilities_sum_to_one(self):
        for method in ("multiplicative", "power"):
            fair = remove_vig([2.10, 3.50, 3.60], method)
            assert sum(fair) == pytest.approx(1.0, abs=1e-9)

    def test_devig_reduces_every_probability(self):
        prices = [1.90, 1.90]
        fair = remove_vig(prices, "multiplicative")
        for price, probability in zip(prices, fair):
            assert probability < implied_probability(price)

    def test_power_method_shrinks_longshots_more(self):
        # Needs a real overround: with prices summing under 100% there is no
        # margin to remove and the power method defers to proportional scaling.
        prices = [1.10, 8.0]
        multiplicative = remove_vig_multiplicative(prices)
        power = remove_vig_power(prices)
        # The favourite keeps more of its probability under the power method.
        assert power[0] > multiplicative[0]
        assert power[1] < multiplicative[1]

    def test_a_fair_book_is_left_alone(self):
        fair = remove_vig([2.0, 2.0], "power")
        assert fair == pytest.approx([0.5, 0.5])

    def test_needs_the_whole_market(self):
        with pytest.raises(ValueError):
            remove_vig([1.9])


class TestConsensus:
    def test_devigs_each_book_before_aggregating(self):
        consensus = build_consensus(
            {
                "A": {"home": 1.90, "away": 1.95},
                "B": {"home": 1.85, "away": 2.05},
                "C": {"home": 1.95, "away": 1.90},
            }
        )
        assert sum(consensus.fair_probabilities.values()) == pytest.approx(1.0)
        assert consensus.n_bookmakers == 3
        assert consensus.overround > 0

    def test_best_price_is_the_highest_available(self):
        consensus = build_consensus(
            {"A": {"home": 1.90, "away": 1.95}, "B": {"home": 2.05, "away": 1.80}}
        )
        assert consensus.best_price["home"] == 2.05
        assert consensus.best_bookmaker["home"] == "B"

    def test_partial_books_are_ignored(self):
        consensus = build_consensus(
            {"A": {"home": 1.90, "away": 1.95}, "Partial": {"home": 5.0}}
        )
        # The 5.0 price would wreck a naive average; it cannot be de-vigged.
        assert consensus.n_bookmakers == 1
        assert consensus.best_price["home"] == 5.0  # still shown as available

    def test_rejects_a_market_no_book_quotes_completely(self):
        with pytest.raises(ValueError):
            build_consensus({"A": {"home": 1.9}, "B": {"away": 2.0}})


class TestExpectedValue:
    def test_ev_is_zero_at_the_fair_price(self):
        result = expected_value(0.5, 2.0)
        assert result.ev_per_unit == pytest.approx(0.0)
        assert result.breakeven_probability == pytest.approx(0.5)

    def test_positive_ev_when_model_beats_the_price(self):
        result = expected_value(0.60, 2.0, 0.50)
        assert result.ev_per_unit == pytest.approx(0.20)
        assert result.edge == pytest.approx(0.10)

    def test_negative_ev_is_reported_not_hidden(self):
        assert expected_value(0.40, 2.0, 0.5).ev_per_unit == pytest.approx(-0.20)

    def test_fair_price_matches_the_model(self):
        assert expected_value(0.25, 5.0).fair_price == pytest.approx(4.0)

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            expected_value(1.4, 2.0)
        with pytest.raises(ValueError):
            expected_value(0.5, 0.9)


class TestKelly:
    def test_full_kelly_matches_the_closed_form(self):
        # p=0.6, d=2.0 -> (0.6*2 - 1)/(2-1) = 0.20
        assert kelly_fraction(0.6, 2.0) == pytest.approx(0.20)

    def test_no_stake_without_an_edge(self):
        assert kelly_fraction(0.5, 2.0) == 0.0
        assert kelly_fraction(0.4, 2.0) == 0.0

    def test_fractional_kelly_scales_linearly(self):
        full = recommend_stake(model_probability=0.6, price_decimal=2.0, bankroll=1000,
                               kelly_multiplier=1.0, max_stake_pct=1.0)
        quarter = recommend_stake(model_probability=0.6, price_decimal=2.0, bankroll=1000,
                                  kelly_multiplier=0.25, max_stake_pct=1.0)
        assert quarter.stake == pytest.approx(full.stake * 0.25)

    def test_cap_binds_and_is_reported(self):
        result = recommend_stake(model_probability=0.9, price_decimal=3.0, bankroll=1000,
                                 kelly_multiplier=1.0, max_stake_pct=0.02)
        assert result.stake == pytest.approx(20.0)
        assert result.capped_by == "max_stake_pct"

    def test_low_confidence_shrinks_toward_the_market(self):
        assert shrink_probability(0.70, 0.50, 1.0) == pytest.approx(0.70)
        assert shrink_probability(0.70, 0.50, 0.0) == pytest.approx(0.50)
        assert shrink_probability(0.70, 0.50, 0.5) == pytest.approx(0.60)

    def test_uncertainty_reduces_the_stake(self):
        confident = recommend_stake(model_probability=0.60, price_decimal=2.0,
                                    market_probability=0.50, confidence=1.0, bankroll=1000,
                                    max_stake_pct=1.0)
        unsure = recommend_stake(model_probability=0.60, price_decimal=2.0,
                                 market_probability=0.50, confidence=0.4, bankroll=1000,
                                 max_stake_pct=1.0)
        assert unsure.stake < confident.stake


class TestEdgeScore:
    def test_score_is_bounded_and_weighted(self):
        score = compute_edge_score(edge=0.06, model_confidence=1.0, data_quality=100,
                                   calibration_quality=1.0, model_agreement=1.0,
                                   market_liquidity=1.0)
        assert score.score == pytest.approx(10.0, abs=1e-6)
        assert sum(c.weight for c in score.components) == pytest.approx(1.0)

    def test_poor_inputs_drag_a_big_edge_down(self):
        good = compute_edge_score(edge=0.06, model_confidence=0.9, data_quality=95,
                                  calibration_quality=0.9, model_agreement=0.9,
                                  market_liquidity=1.0)
        stale = compute_edge_score(edge=0.06, model_confidence=0.3, data_quality=30,
                                   calibration_quality=0.2, model_agreement=0.3,
                                   market_liquidity=0.2)
        assert stale.score < good.score / 1.5

    def test_huge_edges_do_not_keep_scoring_higher(self):
        moderate = compute_edge_score(edge=0.06, model_confidence=0.8, data_quality=80,
                                      calibration_quality=0.8, model_agreement=0.8,
                                      market_liquidity=0.8)
        absurd = compute_edge_score(edge=0.40, model_confidence=0.8, data_quality=80,
                                    calibration_quality=0.8, model_agreement=0.8,
                                    market_liquidity=0.8)
        assert absurd.score == pytest.approx(moderate.score)

    def test_liquidity_proxy_saturates(self):
        assert market_liquidity_proxy(0) == 0.0
        assert market_liquidity_proxy(4) == pytest.approx(0.5)
        assert market_liquidity_proxy(50) == 1.0


class TestPortfolio:
    def _candidate(self, key, game, teams, stake, score=8.0):
        return Candidate(key=key, game_uid=game, sport="nba", market="h2h",
                         selection="home", teams=teams, price_decimal=2.0, stake=stake,
                         model_probability=0.6, edge=0.05, edge_score=score)

    def test_per_game_cap_is_enforced(self):
        candidates = [
            self._candidate("a", "g1", ("t1", "t2"), 40),
            self._candidate("b", "g1", ("t1", "t2"), 40),
        ]
        result = build_portfolio(candidates, bankroll=1000, max_game_pct=0.03,
                                 max_slate_pct=1.0, max_team_pct=1.0, max_sport_pct=1.0)
        assert sum(a.stake for a in result.allocations) == pytest.approx(30.0, abs=0.05)

    def test_slate_cap_is_enforced(self):
        candidates = [
            self._candidate(f"k{i}", f"g{i}", (f"t{i}", f"u{i}"), 40) for i in range(10)
        ]
        result = build_portfolio(candidates, bankroll=1000, max_slate_pct=0.10,
                                 max_game_pct=1.0, max_team_pct=1.0, max_sport_pct=1.0)
        assert result.total_stake <= 100.0 + 0.05

    def test_team_cap_limits_correlated_exposure(self):
        candidates = [
            self._candidate("a", "g1", ("shared", "x"), 40),
            self._candidate("b", "g2", ("shared", "y"), 40),
        ]
        result = build_portfolio(candidates, bankroll=1000, max_team_pct=0.04,
                                 max_slate_pct=1.0, max_game_pct=1.0, max_sport_pct=1.0)
        assert sum(a.stake for a in result.allocations) == pytest.approx(40.0, abs=0.05)

    def test_binding_constraint_is_reported(self):
        result = build_portfolio(
            [self._candidate("a", "g1", ("t1", "t2"), 500)],
            bankroll=1000, max_game_pct=0.03, max_slate_pct=1.0,
            max_team_pct=1.0, max_sport_pct=1.0,
        )
        assert any("game_cap" in flag for flag in result.allocations[0].binding_constraints)

    def test_best_opportunities_are_processed_first(self):
        candidates = [
            self._candidate("weak", "g1", ("a", "b"), 50, score=2.0),
            self._candidate("strong", "g2", ("c", "d"), 50, score=9.0),
        ]
        result = build_portfolio(candidates, bankroll=1000, max_slate_pct=0.05,
                                 max_game_pct=1.0, max_team_pct=1.0, max_sport_pct=1.0)
        assert result.allocations[0].candidate.key == "strong"

    def test_correlation_warnings_surface_same_game_bets(self):
        warnings = correlation_warnings([
            self._candidate("a", "g1", ("t1", "t2"), 10),
            self._candidate("b", "g1", ("t1", "t2"), 10),
        ])
        assert any("g1" in warning for warning in warnings)


class TestClv:
    def test_beating_the_close_is_positive_clv(self):
        result = closing_line_value(2.20, {"home": 2.00, "away": 1.95}, "home")
        assert result.clv_price_pct > 0
        assert result.beat_close is True

    def test_taking_a_worse_price_is_negative_clv(self):
        result = closing_line_value(1.80, {"home": 2.00, "away": 1.95}, "home")
        assert result.clv_price_pct < 0
        assert result.beat_close is False

    def test_clv_is_measured_against_the_no_vig_close(self):
        result = closing_line_value(2.00, {"home": 2.00, "away": 2.00}, "home")
        # A 100% book: the fair closing probability is 0.5 while the raw
        # implied probability is also 0.5, so there is no CLV.
        assert result.closing_fair_probability == pytest.approx(0.5)
        assert result.clv_prob_points == pytest.approx(0.0, abs=1e-9)

    def test_summary_aggregates(self):
        results = [
            closing_line_value(2.20, {"home": 2.00, "away": 1.95}, "home"),
            closing_line_value(1.80, {"home": 2.00, "away": 1.95}, "home"),
        ]
        summary = summarise_clv(results)
        assert summary.n == 2
        assert summary.beat_close_rate == pytest.approx(0.5)

    def test_requires_the_selection(self):
        with pytest.raises(ValueError):
            closing_line_value(2.0, {"away": 1.9}, "home")
