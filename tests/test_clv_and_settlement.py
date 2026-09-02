"""CLV, closing-line policy and settlement.

The properties tested here are the ones that, if broken, would make every
downstream number wrong without anything visibly failing:

* one CLV sign convention everywhere;
* a close is never taken from after the event started;
* settlement is idempotent and incremental;
* a pre-event fixture has no close, so it cannot be "settled" early.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from divinelines.betting.clv import (
    MIN_SAMPLE_FOR_INFERENCE,
    ClvResult,
    clv_against_close,
    closing_line_value,
    compute_clv,
    summarise_clv,
)
from divinelines.betting.closing_line import (
    STATUS_NO_CLOSE,
    STATUS_PENDING,
    ClosingLine,
    ClosingLinePolicy,
    NoClosingLine,
    resolve_closing_line,
)
from divinelines.betting.settlement import _grade, settle, settlement_state


class TestClvConvention:
    def test_longer_entry_price_is_positive_clv(self):
        """Taking 2.50 on something that closes 2.20 means beating the close."""
        result = compute_clv(2.50, 2.20)
        assert result.clv_price_pct > 0
        assert result.beat_close is True

    def test_shorter_entry_price_is_negative_clv(self):
        result = compute_clv(2.00, 2.30)
        assert result.clv_price_pct < 0
        assert result.beat_close is False

    def test_magnitude_matches_the_formula(self):
        # entry / close - 1 = 2.50/2.00 - 1 = 25%
        assert compute_clv(2.50, 2.00).clv_price_pct == pytest.approx(25.0)

    def test_no_movement_is_zero(self):
        assert compute_clv(2.0, 2.0).clv_price_pct == pytest.approx(0.0)

    def test_probability_form_uses_the_novig_close(self):
        result = compute_clv(2.50, 2.00, closing_fair_probability=0.48)
        # entry implied 40%, fair close 48% -> +8 probability points
        assert result.clv_prob_points == pytest.approx(8.0, abs=1e-6)

    def test_all_entry_points_agree(self):
        """CLI, settlement and analysis must not each derive their own number."""
        prices = {"home": 2.00, "away": 2.00}
        direct = compute_clv(2.50, 2.00, closing_fair_probability=0.5)
        via_market = closing_line_value(2.50, prices, "home")
        close = ClosingLine(
            game_uid="g", market="h2h", prices=prices,
            novig_probabilities={"home": 0.5, "away": 0.5},
            implied_probabilities={"home": 0.5, "away": 0.5},
            book_prices={"BookA": prices}, bookmaker="BookA", source="test",
            timestamp=None, policy="median", phase="declared_close", n_bookmakers=1,
        )
        via_close = clv_against_close(2.50, "home", close)

        assert direct.clv_price_pct == pytest.approx(via_market.clv_price_pct)
        assert direct.clv_price_pct == pytest.approx(via_close.clv_price_pct)

    def test_rejects_impossible_prices(self):
        with pytest.raises(ValueError):
            compute_clv(1.0, 2.0)
        with pytest.raises(ValueError):
            compute_clv(2.0, 0.9)


class TestClvSummary:
    def test_small_samples_refuse_to_claim_anything(self):
        summary = summarise_clv([5.0] * 10)
        assert summary.significant is False
        assert "insufficient sample" in summary.interpretation
        assert summary.ci_low is None

    def test_large_consistent_sample_is_significant(self):
        summary = summarise_clv([3.0, 4.0, 5.0] * 40)
        assert summary.n == 120
        assert summary.significant is True
        assert summary.ci_low > 0

    def test_noisy_sample_is_not_significant(self):
        values = [10.0, -10.0] * 50
        summary = summarise_clv(values)
        assert summary.mean_clv_price_pct == pytest.approx(0.0)
        assert summary.significant is False
        assert "consistent with zero" in summary.interpretation

    def test_median_exposes_a_mean_dragged_by_outliers(self):
        """Nine small losses and one huge win must not read as an edge."""
        summary = summarise_clv([-1.0] * 9 + [100.0])
        assert summary.mean_clv_price_pct > 0
        assert summary.median_clv_price_pct < 0

    def test_percentiles_are_reported(self):
        summary = summarise_clv(list(range(100)))
        assert summary.percentiles is not None
        assert summary.percentiles["p05"] < summary.percentiles["p95"]

    def test_empty_input(self):
        assert summarise_clv([]).n == 0


@pytest.fixture()
def priced_game(seeded_db):
    """A finished game with declared open/close prices from two books."""
    from divinelines.db.repository import upsert_games, upsert_odds

    start = datetime.now(timezone.utc) - timedelta(days=2)
    upsert_games([{
        "game_uid": "nba:closed", "sport": "nba", "league_id": "NBA", "season": "2025-26",
        "game_date": start.date().isoformat(), "kickoff_utc": start.isoformat(),
        "status": "final", "home_team_uid": "nba:BOS", "away_team_uid": "nba:LAL",
        "home_score": 110.0, "away_score": 104.0, "neutral_site": 0, "venue": None,
        "source": "test", "retrieved_at": start.isoformat(),
    }])
    rows = []
    for book, open_home, close_home in (("BookA", 2.10, 1.90), ("BookB", 2.00, 1.95)):
        for phase, home_price in (("open", open_home), ("close", close_home)):
            for selection, price in (("home", home_price), ("away", 3.0)):
                rows.append({
                    "game_uid": "nba:closed", "sport": "nba", "market": "h2h",
                    "selection": selection, "bookmaker": book, "price_decimal": price,
                    "captured_at": start.isoformat(), "book_updated": None,
                    "is_closing": 1 if phase == "close" else 0, "phase": phase,
                    "source": "espn_odds",
                })
    upsert_odds(rows)
    return seeded_db


class TestClosingLinePolicy:
    def test_resolves_a_declared_close(self, priced_game):
        close = resolve_closing_line("nba:closed", "h2h", ("home", "away"))
        assert isinstance(close, ClosingLine)
        assert close.phase == "declared_close"
        # median of 1.90 and 1.95
        assert close.price_for("home") == pytest.approx(1.925)

    def test_novig_probabilities_sum_to_one(self, priced_game):
        close = resolve_closing_line("nba:closed", "h2h", ("home", "away"))
        assert sum(close.novig_probabilities.values()) == pytest.approx(1.0)

    def test_per_book_prices_are_retained_for_same_book_comparison(self, priced_game):
        close = resolve_closing_line("nba:closed", "h2h", ("home", "away"))
        assert set(close.book_prices) == {"BookA", "BookB"}
        assert close.book_prices["BookA"]["home"] == pytest.approx(1.90)

    def test_best_aggregation_differs_from_median(self, priced_game):
        best = resolve_closing_line("nba:closed", "h2h", ("home", "away"),
                                    policy=ClosingLinePolicy(aggregation="best"))
        assert best.price_for("home") == pytest.approx(1.95)
        assert "best" in best.bookmaker

    def test_named_book_aggregation(self, priced_game):
        close = resolve_closing_line(
            "nba:closed", "h2h", ("home", "away"),
            policy=ClosingLinePolicy(aggregation="book", bookmaker="BookA"),
        )
        assert close.price_for("home") == pytest.approx(1.90)
        assert close.bookmaker == "BookA"

    def test_policy_label_is_recorded(self, priced_game):
        close = resolve_closing_line("nba:closed", "h2h", ("home", "away"))
        assert "median" in close.policy

    def test_future_event_has_no_close(self, seeded_db):
        """A fixture that has not kicked off cannot have a closing line."""
        result = resolve_closing_line("nba:upcoming", "h2h", ("home", "away"))
        assert isinstance(result, NoClosingLine)
        assert result.status == STATUS_PENDING
        assert "not started" in result.reason

    def test_snapshots_after_kickoff_are_rejected(self, seeded_db):
        """In-play prices must never become the close."""
        from divinelines.db.repository import upsert_games, upsert_odds

        start = datetime.now(timezone.utc) - timedelta(hours=3)
        upsert_games([{
            "game_uid": "nba:inplay", "sport": "nba", "league_id": "NBA", "season": "2025-26",
            "game_date": start.date().isoformat(), "kickoff_utc": start.isoformat(),
            "status": "final", "home_team_uid": "nba:BOS", "away_team_uid": "nba:LAL",
            "home_score": 100.0, "away_score": 99.0, "neutral_site": 0, "venue": None,
            "source": "test", "retrieved_at": start.isoformat(),
        }])
        after = (start + timedelta(hours=1)).isoformat()
        upsert_odds([
            {"game_uid": "nba:inplay", "sport": "nba", "market": "h2h", "selection": selection,
             "bookmaker": "BookA", "price_decimal": price, "captured_at": after,
             "book_updated": None, "is_closing": 0, "phase": "snapshot", "source": "test"}
            for selection, price in (("home", 1.10), ("away", 8.0))
        ])
        result = resolve_closing_line("nba:inplay", "h2h", ("home", "away"))
        assert isinstance(result, NoClosingLine)
        assert result.status == STATUS_NO_CLOSE

    def test_missing_prices_explain_themselves(self, seeded_db):
        result = resolve_closing_line("nba:0022500000", "h2h", ("home", "away"))
        assert isinstance(result, NoClosingLine)
        assert result.reason


@pytest.fixture()
def replayed_prediction(priced_game):
    from divinelines.betting.ledger import PredictionRecord, record_predictions
    from divinelines.models.registry import ModelRecord, register

    register(ModelRecord(model_id="test-model", sport="nba", kind="ensemble",
                         model_version="v1", feature_set=["diff_elo"],
                         feature_set_version="test", league_id="NBA"))
    record_predictions([PredictionRecord(
        sport="nba", game_uid="nba:closed", market="h2h", selection="home",
        model_probability=0.62, market_probability=0.52, price_decimal=2.10,
        bookmaker="BookA", edge=0.10, ev_per_unit=0.30, stake=25.0,
        model_id="test-model", model_version="v1", mode="backtest",
        prediction_stage="pre_event",
    )])
    return priced_game


class TestSettlement:
    def test_creates_a_clv_record_with_the_close(self, replayed_prediction):
        from divinelines.db.connection import query_df

        report = settle(sport="nba")
        assert report.close_found == 1
        record = query_df("SELECT * FROM clv_records").iloc[0]
        assert record["closing_odds"] == pytest.approx(1.925)
        assert record["clv_price_pct"] == pytest.approx((2.10 / 1.925 - 1) * 100)
        assert record["status"] == "SETTLED"

    def test_same_book_clv_uses_the_entry_book(self, replayed_prediction):
        from divinelines.db.connection import query_df

        settle(sport="nba")
        record = query_df("SELECT * FROM clv_records").iloc[0]
        # entry at BookA 2.10, BookA closed 1.90
        assert record["closing_same_book_odds"] == pytest.approx(1.90)
        assert record["clv_same_book_pct"] == pytest.approx((2.10 / 1.90 - 1) * 100)

    def test_grades_the_result_and_profit(self, replayed_prediction):
        from divinelines.db.connection import query_df

        settle(sport="nba")
        record = query_df("SELECT * FROM clv_records").iloc[0]
        assert record["result"] == "won"          # 110-104 home win
        assert record["profit"] == pytest.approx(25.0 * (2.10 - 1))

    def test_settlement_is_idempotent(self, replayed_prediction):
        from divinelines.db.connection import query_df

        settle(sport="nba")
        first = query_df("SELECT * FROM clv_records")
        settle(sport="nba")
        second = query_df("SELECT * FROM clv_records")

        assert len(first) == len(second) == 1
        assert first["profit"].iloc[0] == pytest.approx(second["profit"].iloc[0])
        assert first["clv_id"].iloc[0] == second["clv_id"].iloc[0]

    def test_second_run_is_incremental(self, replayed_prediction):
        settle(sport="nba")
        second = settle(sport="nba")
        assert second.scanned == 0

    def test_full_reconcile_reprocesses(self, replayed_prediction):
        settle(sport="nba")
        reconciled = settle(sport="nba", full_reconcile=True)
        assert reconciled.scanned == 1

    def test_dry_run_writes_nothing(self, replayed_prediction):
        from divinelines.db.connection import query_df

        report = settle(sport="nba", dry_run=True)
        assert report.close_found == 1
        assert query_df("SELECT * FROM clv_records").empty

    def test_pending_when_the_event_has_not_started(self, seeded_db):
        from divinelines.betting.ledger import PredictionRecord, record_predictions
        from divinelines.db.connection import query_df
        from divinelines.models.registry import ModelRecord, register

        register(ModelRecord(model_id="m", sport="nba", kind="ensemble", model_version="v1",
                             feature_set=[], feature_set_version="t"))
        record_predictions([PredictionRecord(
            sport="nba", game_uid="nba:upcoming", market="h2h", selection="home",
            model_probability=0.6, price_decimal=2.0, model_id="m", model_version="v1",
        )])
        report = settle(sport="nba")
        assert report.awaiting_close == 1
        assert query_df("SELECT status FROM clv_records").iloc[0]["status"] == "PENDING"

    def test_predictions_without_a_price_are_not_counted(self, seeded_db):
        from divinelines.betting.ledger import PredictionRecord, record_predictions
        from divinelines.models.registry import ModelRecord, register

        register(ModelRecord(model_id="m", sport="nba", kind="ensemble", model_version="v1",
                             feature_set=[], feature_set_version="t"))
        record_predictions([PredictionRecord(
            sport="nba", game_uid="nba:upcoming", market="h2h", selection="home",
            model_probability=0.6, price_decimal=None, model_id="m", model_version="v1",
        )])
        assert settle(sport="nba").scanned == 0

    def test_settlement_state_counts(self, replayed_prediction):
        settle(sport="nba")
        state = settlement_state()
        assert state.get("SETTLED") == 1


class TestGrading:
    @pytest.mark.parametrize(
        "market,selection,home,away,expected",
        [
            ("h2h", "home", 110, 104, "won"),
            ("h2h", "away", 110, 104, "lost"),
            ("1x2", "draw", 1, 1, "won"),
            ("1x2", "home", 1, 1, "lost"),
            ("1x2", "away", 0, 2, "won"),
            ("h2h", "home", 100, 100, "push"),
            ("totals", "over_2.5", 2, 1, "won"),
            ("totals", "under_2.5", 2, 1, "lost"),
            ("totals", "over_3.0", 2, 1, "push"),
        ],
    )
    def test_grades(self, market, selection, home, away, expected):
        assert _grade(market, selection, home, away) == expected

    def test_ungraded_when_no_score(self):
        assert _grade("h2h", "home", None, None) is None

    def test_unknown_market_is_not_guessed(self):
        assert _grade("weird", "home", 2, 1) is None
