"""Identity resolution, migration, validation and freshness.

The migration tests encode the two real corruptions found in the v1 database,
so a regression would be caught rather than rediscovered months later.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from divinelines.data.freshness import (
    QualityComponent,
    classify,
    compute_data_quality,
    freshness_score,
)
from divinelines.db.migrate import normalise_legacy_game_logs, season_label_from_season_id
from divinelines.db.validation import (
    ValidationError,
    validate_games,
    validate_nba_box,
    validate_odds,
    validate_probabilities,
)
from divinelines.identity import (
    canonical_club_name,
    club_id,
    normalize_key,
    resolve_nba_team,
    same_club,
)


class TestNbaIdentity:
    @pytest.mark.parametrize(
        "value",
        ["LAL", "lal", "Los Angeles Lakers", "LA Lakers", "L.A. Lakers", "Lakers",
         1610612747, "1610612747"],
    )
    def test_every_spelling_resolves_to_one_team(self, value):
        assert resolve_nba_team(value) == "LAL"

    def test_relocated_franchises_keep_one_identity(self):
        assert resolve_nba_team("Seattle SuperSonics") == "OKC"
        assert resolve_nba_team("New Jersey Nets") == "BKN"
        assert resolve_nba_team("Charlotte Bobcats") == "CHA"

    def test_the_two_la_teams_stay_separate(self):
        assert resolve_nba_team("LA Clippers") == "LAC"
        assert resolve_nba_team("Los Angeles Lakers") == "LAL"

    def test_espn_style_abbreviations(self):
        assert resolve_nba_team("GS") == "GSW"
        assert resolve_nba_team("NY") == "NYK"
        assert resolve_nba_team("SA") == "SAS"

    def test_unknown_names_return_none_rather_than_guessing(self):
        assert resolve_nba_team("Notta Team") is None
        assert resolve_nba_team(None) is None
        assert resolve_nba_team("") is None


class TestSoccerIdentity:
    @pytest.mark.parametrize(
        "left,right",
        [
            ("Man United", "Manchester United"),
            ("Man City", "Manchester City"),
            ("Tottenham", "Tottenham Hotspur"),
            ("Ath Madrid", "Atletico Madrid"),
            ("Inter", "Inter Milan"),
            ("Paris SG", "Paris Saint Germain"),
            ("M'gladbach", "Borussia Monchengladbach"),
            ("Sp Lisbon", "Sporting CP"),
        ],
    )
    def test_source_spellings_merge(self, left, right):
        assert same_club(left, right), f"{left} should resolve to {right}"

    def test_different_clubs_do_not_merge(self):
        assert not same_club("Manchester United", "Manchester City")
        assert not same_club("Real Madrid", "Atletico Madrid")

    def test_accents_and_punctuation_are_normalised(self):
        assert normalize_key("Atlético Madrid") == normalize_key("Atletico Madrid")
        assert club_id("Bayern München") == club_id("Bayern Munchen")

    def test_club_id_is_stable_and_slug_like(self):
        assert club_id("Manchester United") == "manchester-united"

    def test_unknown_club_still_gets_a_stable_id(self):
        assert club_id("Some New FC") == club_id("some new")


class TestLegacyMigration:
    def _legacy_frame(self):
        """Reproduces the real defects found in the v1 database."""
        rows = []
        for padded in ("0022500001", "22500001"):  # same game, two id formats
            for team_id, abbr, matchup, wl, points in (
                (1610612738, "BOS", "BOS vs. LAL", "W", 110),
                (1610612747, "LAL", "LAL @ BOS", "L", 104),
            ):
                rows.append({
                    "SEASON_ID": "22025", "TEAM_ID": team_id, "TEAM_ABBREVIATION": abbr,
                    "GAME_ID": padded, "GAME_DATE": "2025-11-02", "MATCHUP": matchup,
                    "WL": wl, "MIN": 240, "PTS": points, "FGM": 40, "FGA": 88, "FG3M": 12,
                    "FG3A": 34, "FTM": 18, "FTA": 22, "OREB": 10, "DREB": 33, "REB": 43,
                    "AST": 25, "STL": 7, "BLK": 5, "TOV": 13, "PF": 19, "PLUS_MINUS": 6,
                })
        # A game captured while still in progress: no result, partial minutes.
        for team_id, abbr, matchup in (
            (1610612738, "BOS", "BOS vs. LAL"), (1610612747, "LAL", "LAL @ BOS")
        ):
            rows.append({
                "SEASON_ID": "22025", "TEAM_ID": team_id, "TEAM_ABBREVIATION": abbr,
                "GAME_ID": "0022500099", "GAME_DATE": "2025-11-09", "MATCHUP": matchup,
                "WL": None, "MIN": 0, "PTS": 18, "FGM": 6, "FGA": 13, "FG3M": 2,
                "FG3A": 5, "FTM": 4, "FTA": 5, "OREB": 2, "DREB": 4, "REB": 6,
                "AST": 3, "STL": 1, "BLK": 0, "TOV": 2, "PF": 3, "PLUS_MINUS": 0,
            })
        return pd.DataFrame(rows)

    def test_padded_and_unpadded_ids_collapse_to_one_game(self):
        cleaned, stats = normalise_legacy_game_logs(self._legacy_frame())
        assert stats["duplicate_team_games_removed"] == 2
        assert cleaned[cleaned["is_final"]]["GAME_ID"].nunique() == 1

    def test_ids_are_canonicalised_to_ten_characters(self):
        cleaned, _ = normalise_legacy_game_logs(self._legacy_frame())
        assert (cleaned["GAME_ID"].str.len() == 10).all()

    def test_in_progress_games_are_not_treated_as_results(self):
        cleaned, stats = normalise_legacy_game_logs(self._legacy_frame())
        assert stats["partial_games_excluded"] == 1
        assert not cleaned[cleaned["GAME_ID"] == "0022500099"]["is_final"].any()

    def test_home_and_away_are_derived_from_the_matchup_string(self):
        cleaned, _ = normalise_legacy_game_logs(self._legacy_frame())
        finished = cleaned[cleaned["is_final"]]
        assert finished["is_home"].sum() == 1

    def test_season_label_from_season_id(self):
        assert season_label_from_season_id("22025", pd.Timestamp("2025-11-02")) == "2025-26"
        assert season_label_from_season_id("22021", pd.Timestamp("2021-11-02")) == "2021-22"

    def test_season_label_falls_back_to_the_date(self):
        assert season_label_from_season_id(None, pd.Timestamp("2025-11-02")) == "2025-26"
        assert season_label_from_season_id("junk", pd.Timestamp("2025-03-02")) == "2024-25"


class TestValidation:
    def _games(self, **overrides):
        row = {
            "game_uid": "nba:1", "game_date": "2025-11-02", "status": "final",
            "home_team_uid": "nba:BOS", "away_team_uid": "nba:LAL",
            "home_score": 110.0, "away_score": 104.0,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_valid_games_pass(self):
        assert validate_games(self._games()).ok

    def test_duplicate_game_uid_is_critical(self):
        frame = pd.concat([self._games(), self._games()], ignore_index=True)
        report = validate_games(frame)
        assert not report.ok
        assert any(issue.code == "duplicate_game_uid" for issue in report.critical)

    def test_team_playing_itself_is_critical(self):
        report = validate_games(self._games(away_team_uid="nba:BOS"))
        assert any(issue.code == "self_matchup" for issue in report.critical)

    def test_final_game_without_a_score_is_critical(self):
        report = validate_games(self._games(home_score=None))
        assert any(issue.code == "final_without_score" for issue in report.critical)

    def test_unparseable_date_is_critical(self):
        report = validate_games(self._games(game_date="not-a-date"))
        assert any(issue.code == "invalid_date" for issue in report.critical)

    def test_empty_dataset_is_critical(self):
        assert not validate_games(pd.DataFrame()).ok

    def test_raise_if_critical(self):
        with pytest.raises(ValidationError):
            validate_games(self._games(home_score=None)).raise_if_critical()

    def test_box_scores_need_exactly_two_teams(self):
        frame = pd.DataFrame([
            {"game_uid": "g1", "team_uid": "a", "pts": 110, "fgm": 40, "fga": 88,
             "fta": 22, "reb": 43, "tov": 13},
        ])
        report = validate_nba_box(frame)
        assert any(issue.code == "bad_team_count" for issue in report.critical)

    def test_impossible_scores_are_rejected(self):
        frame = pd.DataFrame([
            {"game_uid": "g1", "team_uid": "a", "pts": 250, "fgm": 40, "fga": 88,
             "fta": 22, "reb": 43, "tov": 13},
            {"game_uid": "g1", "team_uid": "b", "pts": 104, "fgm": 40, "fga": 88,
             "fta": 22, "reb": 43, "tov": 13},
        ])
        assert any(i.code == "implausible_points" for i in validate_nba_box(frame).critical)

    def test_makes_cannot_exceed_attempts(self):
        frame = pd.DataFrame([
            {"game_uid": "g1", "team_uid": "a", "pts": 110, "fgm": 90, "fga": 88,
             "fta": 22, "reb": 43, "tov": 13},
            {"game_uid": "g1", "team_uid": "b", "pts": 104, "fgm": 40, "fga": 88,
             "fta": 22, "reb": 43, "tov": 13},
        ])
        assert any(i.code == "fgm_gt_fga" for i in validate_nba_box(frame).critical)

    def test_impossible_odds_are_rejected(self):
        report = validate_odds([
            {"game_uid": "g", "selection": "home", "price_decimal": 0.5},
            {"game_uid": "g", "selection": "away", "price_decimal": 2.0},
        ])
        assert any(i.code == "implausible_price" for i in report.critical)

    def test_probabilities_must_be_valid_and_sum_to_one(self):
        assert validate_probabilities({"home": 0.6, "away": 0.4}).ok
        assert not validate_probabilities({"home": 0.6, "away": 0.6}).ok
        assert not validate_probabilities({"home": 1.4, "away": -0.4}).ok


class TestFreshness:
    def test_states(self):
        assert classify(60, 600) == "fresh"
        assert classify(1200, 600) == "aging"
        assert classify(100000, 600) == "stale"
        assert classify(None, 600) == "missing"

    def test_scores_decline_with_staleness(self):
        from divinelines.data.freshness import Freshness

        def build(state):
            return Freshness("d", "odds", "s", None, 600, 100.0, state)

        assert freshness_score(build("fresh")) == 1.0
        assert freshness_score(build("aging")) < 1.0
        assert freshness_score(build("stale")) < freshness_score(build("aging"))
        assert freshness_score(None) == 0.0

    def test_quality_score_is_a_transparent_weighted_mean(self):
        quality = compute_data_quality([
            QualityComponent("a", 1.0, 0.5),
            QualityComponent("b", 0.0, 0.5),
        ])
        assert quality.score == pytest.approx(50.0)
        assert quality.grade == "low"
        assert len(quality.to_dict()["components"]) == 2

    def test_perfect_inputs_score_high(self):
        quality = compute_data_quality([QualityComponent("a", 1.0, 1.0)])
        assert quality.score == pytest.approx(100.0)
        assert quality.grade == "high"


class TestRepository:
    def test_upsert_is_idempotent(self, seeded_db):
        from divinelines.db.connection import query_df
        from divinelines.db.repository import ensure_nba_teams

        before = query_df("SELECT COUNT(*) AS n FROM teams")["n"].iloc[0]
        ensure_nba_teams()
        after = query_df("SELECT COUNT(*) AS n FROM teams")["n"].iloc[0]
        assert before == after

    def test_load_games_filters_by_status(self, seeded_db):
        from divinelines.db.repository import load_games

        assert len(load_games("nba", status="final")) == 6
        assert len(load_games("nba", status="scheduled")) == 1

    def test_latest_odds_returns_one_row_per_book_and_selection(self, seeded_db):
        from divinelines.db.repository import latest_odds

        odds = latest_odds("nba:upcoming", "h2h")
        assert len(odds) == 4
        assert set(odds["selection"]) == {"home", "away"}

    def test_source_status_records_failures(self, seeded_db):
        from divinelines.db.repository import record_source_status, source_status_table

        record_source_status("test_source", "dataset", status="error", message="boom")
        table = source_status_table()
        row = table[table["source"] == "test_source"].iloc[0]
        assert row["status"] == "error"
        assert row["last_success"] is None
