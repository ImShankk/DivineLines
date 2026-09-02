"""Source contract tests.

These run against stored payloads, not the live internet. If ESPN changes a
field name the parser depends on, these fail immediately and deterministically
instead of the platform quietly ingesting nothing at 3am.

The fixtures are real captured responses, trimmed to the fields the parsers
read.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from divinelines.sources.espn_lineups import (
    STATE_CONFIRMED,
    STATE_FINAL,
    EspnLineupSource,
    position_group_for,
)
from divinelines.sources.base import SourceError
from divinelines.sources.football_data import FootballDataSource
from divinelines.sources.espn_match import EspnMatchSource
from divinelines.sources.espn_odds import EspnOddsSource

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class _StubResult:
    """Stands in for a FetchResult without touching the network."""

    def __init__(self, data):
        self.data = data
        self.retrieved_at = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
        self.from_cache = False


class TestEspnOddsParser:
    @pytest.fixture()
    def source(self, monkeypatch):
        source = EspnOddsSource()
        payload = load("espn_nba_odds.json")
        monkeypatch.setattr(source, "fetch_json",
                            lambda *args, **kwargs: _StubResult(payload))
        return source

    def test_parses_open_and_close_for_both_sides(self, source):
        quotes = source.fetch_event_odds("nba", "401585589")
        phases = {(q.selection, q.phase) for q in quotes}
        assert ("home", "open") in phases
        assert ("home", "close") in phases
        assert ("away", "open") in phases
        assert ("away", "close") in phases

    def test_in_play_providers_are_excluded(self, source):
        """A live feed quotes prices during the game; ingesting it would leak."""
        quotes = source.fetch_event_odds("nba", "401585589")
        books = {q.bookmaker for q in quotes}
        assert books, "fixture should yield at least one book"
        assert not any("live" in book.lower() for book in books)

    def test_multiple_books_survive(self, source):
        quotes = source.fetch_event_odds("nba", "401585589")
        assert len({q.bookmaker for q in quotes}) >= 2

    def test_prices_are_decimal_and_plausible(self, source):
        for quote in source.fetch_event_odds("nba", "401585589"):
            assert quote.price_decimal > 1.0
            assert quote.price_decimal < 100.0

    def test_market_is_moneyline(self, source):
        assert {q.market for q in source.fetch_event_odds("nba", "401585589")} == {"h2h"}

    def test_missing_moneyline_block_is_skipped_not_guessed(self):
        assert EspnOddsSource._moneyline_decimal(None) is None
        assert EspnOddsSource._moneyline_decimal({}) is None
        assert EspnOddsSource._moneyline_decimal({"moneyLine": {"decimal": 1.0}}) is None
        assert EspnOddsSource._moneyline_decimal({"moneyLine": {"decimal": 2.5}}) == 2.5

    def test_unknown_sport_raises(self):
        with pytest.raises(SourceError):
            EspnOddsSource().fetch_event_odds("cricket", "1")


class TestEspnLineupParser:
    @pytest.fixture()
    def source(self, monkeypatch):
        source = EspnLineupSource()
        payload = load("espn_soccer_lineup.json")
        monkeypatch.setattr(source, "fetch_json",
                            lambda *args, **kwargs: _StubResult(payload))
        return source

    def test_extracts_two_teams_with_eleven_starters(self, source):
        observation = source.fetch_lineup("ENG_PL", "740947", event_started=True)
        assert len(observation.teams) == 2
        for team in observation.teams:
            assert len(team.starters) == 11

    def test_goalkeeper_is_identified(self, source):
        observation = source.fetch_lineup("ENG_PL", "740947", event_started=True)
        for team in observation.teams:
            assert team.has_goalkeeper, f"{team.team_name} has no goalkeeper in its XI"

    def test_formation_is_captured(self, source):
        observation = source.fetch_lineup("ENG_PL", "740947", event_started=True)
        assert any(team.formation for team in observation.teams)

    def test_started_event_is_marked_final_not_confirmed(self, source):
        """A lineup seen after kick-off is a record, not advance information."""
        observation = source.fetch_lineup("ENG_PL", "740947", event_started=True)
        assert observation.lineup_state == STATE_FINAL
        assert observation.is_usable_live is False

    def test_pre_event_lineup_is_usable_live(self, source):
        observation = source.fetch_lineup("ENG_PL", "740947", event_started=False)
        assert observation.lineup_state == STATE_CONFIRMED
        assert observation.is_usable_live is True

    def test_empty_rosters_raise_rather_than_return_nothing(self, monkeypatch):
        source = EspnLineupSource()
        monkeypatch.setattr(source, "fetch_json",
                            lambda *args, **kwargs: _StubResult({"rosters": []}))
        with pytest.raises(SourceError):
            source.fetch_lineup("ENG_PL", "1")


class TestPositionMapping:
    @pytest.mark.parametrize(
        "slot,expected",
        [("G", "goalkeeper"), ("CD-L", "defender"), ("CD-R", "defender"),
         ("LB", "defender"), ("RB", "defender"), ("LM", "midfielder"),
         ("AM-R", "midfielder"), ("CM-L", "midfielder"), ("DM", "midfielder"),
         ("CF-L", "forward"), ("LF", "forward"), ("F", "forward")],
    )
    def test_formation_slots_map_to_groups(self, slot, expected):
        """ESPN reports formation slots, not position names, for a starting XI."""
        assert position_group_for(None, slot) == expected

    def test_display_names_still_work(self):
        assert position_group_for("Goalkeeper", None) == "goalkeeper"
        assert position_group_for("Midfielder", None) == "midfielder"

    def test_unknown_slot_returns_none_rather_than_guessing(self):
        assert position_group_for(None, "ZZ") is None
        assert position_group_for(None, None) is None


class TestEspnMatchDetailParser:
    """The match-detail feed, against a stored summary payload.

    If ESPN renames ``fieldPositionX`` or restructures ``commentary``, the
    Match Centre would quietly lose its shot map and its momentum curve. These
    fail loudly instead.
    """

    @pytest.fixture()
    def source(self, monkeypatch):
        source = EspnMatchSource()
        payload = load("espn_soccer_summary.json")
        monkeypatch.setattr(source, "fetch_json",
                            lambda *args, **kwargs: _StubResult(payload))
        return source

    def test_shares_the_lineup_adapter_cache(self):
        """Two adapters, one upstream document, one download."""
        assert EspnMatchSource.cache_namespace == EspnLineupSource.name

    def test_fetch_detail_returns_every_section(self, source):
        detail = source.fetch_detail("ENG_PL", "740604")
        assert detail.espn_event_id == "740604"
        assert detail.events and detail.team_stats and detail.players
        assert detail.context.match_state == "FINISHED"
        assert detail.standings

    def test_an_unknown_league_is_refused_rather_than_guessed(self, source):
        with pytest.raises(SourceError):
            source.fetch_detail("NOT_A_LEAGUE", "1")

    def test_shots_carry_field_positions(self, source):
        detail = source.fetch_detail("ENG_PL", "740604")
        shots = [event for event in detail.events if event.is_shot]
        assert shots, "the fixture contains shots"
        located = [shot for shot in shots
                   if shot.source_x is not None and (shot.source_x or shot.source_y)]
        assert located, "shot coordinates are the whole basis of the shot map"

    def test_cards_and_substitutions_carry_no_position(self, source):
        """ESPN sends 0.0/0.0 rather than omitting the field.

        Treating that as a real coordinate would pile phantom events into one
        corner of the pitch.
        """
        detail = source.fetch_detail("ENG_PL", "740604")
        for event in detail.events:
            if event.event_type in ("yellow_card", "red_card", "substitution"):
                assert not (event.source_x or 0) and not (event.source_y or 0)

    def test_clock_is_cumulative_from_kick_off(self, source):
        detail = source.fetch_detail("ENG_PL", "740604")
        for event in detail.events:
            if event.clock_seconds and event.clock_display and "+" not in event.clock_display:
                stated = int(event.clock_display.strip("'"))
                assert abs(stated - event.clock_seconds / 60) <= 1.5

    def test_player_stats_are_keyed_by_a_canonical_uid(self, source):
        detail = source.fetch_detail("ENG_PL", "740604")
        uids = [player.player_uid for player in detail.players]
        assert all(uid for uid in uids)
        assert len(uids) == len(set(uids))

    def test_a_missing_rosters_block_does_not_crash_the_parse(self, source, monkeypatch):
        payload = load("espn_soccer_summary.json")
        payload.pop("rosters")
        monkeypatch.setattr(source, "fetch_json",
                            lambda *args, **kwargs: _StubResult(payload))
        detail = source.fetch_detail("ENG_PL", "740604")
        assert detail.players == []
        assert detail.events, "an optional section failing must not take the rest with it"

    def test_a_malformed_play_is_skipped_not_fatal(self, source, monkeypatch):
        payload = load("espn_soccer_summary.json")
        payload["commentary"].append({"sequence": 999, "play": {"id": "bad"}})
        payload["commentary"].append({"sequence": 1000})
        monkeypatch.setattr(source, "fetch_json",
                            lambda *args, **kwargs: _StubResult(payload))
        detail = source.fetch_detail("ENG_PL", "740604")
        assert not any(event.external_id == "bad" for event in detail.events)

    def test_an_empty_payload_is_an_error_not_an_empty_match(self, source, monkeypatch):
        monkeypatch.setattr(source, "fetch_json", lambda *args, **kwargs: _StubResult({}))
        with pytest.raises(SourceError):
            source.fetch_detail("ENG_PL", "740604")

    def test_every_source_type_in_the_fixture_maps_to_the_taxonomy(self, source):
        detail = source.fetch_detail("ENG_PL", "740604")
        unmapped = {event.source_type for event in detail.events
                    if event.event_type == "other"}
        assert not unmapped, f"unmapped ESPN play types: {unmapped}"


class TestFootballDataDivisionGuard:
    """football-data.co.uk serves a *different* division's file when the one you
    asked for has not been published yet.

    At the start of 2026-27 the request for ``E0.csv`` came back carrying
    ``Div=EC`` — the National League — and 12 National League fixtures went into
    the store labelled Premier League. Same for ``SP1.csv``, which returned the
    Portuguese Primeira Liga. The adapter now believes the file, not the URL.
    """

    HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR"

    def _csv(self, division: str, *rows: str) -> bytes:
        lines = [self.HEADER] + [f"{division},{row}" for row in rows]
        return "\n".join(lines).encode("utf-8")

    def test_a_file_declaring_another_division_is_refused(self):
        source = FootballDataSource()
        payload = self._csv("EC", "08/08/2026,15:00,Altrincham,Southend,1,3,A")
        with pytest.raises(SourceError) as excinfo:
            source._parse_csv(payload, "ENG_PL", "2627")
        assert "expected division 'E0'" in str(excinfo.value)

    def test_the_right_division_parses_normally(self):
        source = FootballDataSource()
        payload = self._csv("E0", "08/08/2026,15:00,Arsenal,Chelsea,1,0,H")
        frame = source._parse_csv(payload, "ENG_PL", "2627")
        assert len(frame) == 1
        assert frame.iloc[0]["home_name"] == "Arsenal"

    def test_a_mixed_file_keeps_only_the_requested_division(self):
        source = FootballDataSource()
        payload = self._csv("E0", "08/08/2026,15:00,Arsenal,Chelsea,1,0,H")
        payload += b"\nEC,08/08/2026,15:00,Altrincham,Southend,1,3,A"
        frame = source._parse_csv(payload, "ENG_PL", "2627")
        assert frame["home_name"].tolist() == ["Arsenal"]

    def test_a_file_with_no_division_column_is_still_accepted(self):
        """Some very old season files omit Div; refusing them would delete a
        decade of history to fix a 2026 problem."""
        source = FootballDataSource()
        payload = b"Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n08/08/2020,15:00,Arsenal,Chelsea,1,0,H"
        frame = source._parse_csv(payload, "ENG_PL", "2021")
        assert len(frame) == 1
