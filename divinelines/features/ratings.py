"""Streaming rating systems: Elo and opponent-adjusted efficiency.

Both are deliberately **online**: they consume games in chronological order
and expose only the state that existed *before* each game.  That makes them
structurally incapable of leaking future information, which a batch
regression fitted over a whole season cannot promise.

Elo is not a replacement for the gradient-boosted models — it is a feature, a
baseline to beat, and a sanity check when the main model says something
strange.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..config import settings


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------

@dataclass
class EloConfig:
    k: float = 20.0
    home_advantage: float = 60.0     # in Elo points
    initial: float = 1500.0
    #: Weight carried across a season boundary (1.0 = no regression to mean).
    season_carry: float = 0.75
    #: Scale factor of the logistic curve (400 is the Elo convention).
    scale: float = 400.0
    #: Margin-of-victory multiplier, off by default so it can be ablated.
    use_mov: bool = True
    mov_shrink: float = 2.2
    #: Draw handling for sports that have them (soccer).
    draws_possible: bool = False
    draw_width: float = 0.28


@dataclass
class EloState:
    rating: float
    games: int = 0


class EloRatings:
    """Elo with optional margin-of-victory scaling and season regression."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self.ratings: dict[str, EloState] = {}
        self._season: str | None = None

    # -------------------------------------------------------------- accessors

    def rating(self, team: str) -> float:
        return self.ratings.get(team, EloState(self.config.initial)).rating

    def games_played(self, team: str) -> int:
        return self.ratings.get(team, EloState(self.config.initial)).games

    def snapshot(self) -> dict[str, float]:
        return {team: state.rating for team, state in self.ratings.items()}

    # ---------------------------------------------------------------- seasons

    def start_season(self, season: str) -> None:
        """Regress ratings toward the mean when a new season begins.

        A roster in October is not the roster of the previous April; carrying
        a full rating across the break makes the model far too confident about
        teams that changed substantially.
        """
        if self._season is None:
            self._season = season
            return
        if season == self._season:
            return
        carry = self.config.season_carry
        mean = self.config.initial
        for state in self.ratings.values():
            state.rating = mean + carry * (state.rating - mean)
            state.games = 0
        self._season = season

    # ------------------------------------------------------------ prediction

    def expected_home_score(self, home: str, away: str, *, neutral: bool = False) -> float:
        """Expected score for the home team (win + half a draw)."""
        advantage = 0.0 if neutral else self.config.home_advantage
        diff = self.rating(home) + advantage - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / self.config.scale))

    def win_probabilities(self, home: str, away: str, *, neutral: bool = False
                          ) -> dict[str, float]:
        """Outcome probabilities; three-way when draws are possible.

        The draw model is the standard Elo adaptation: the expected score is
        split around a draw band whose width is a fitted constant.  It is a
        baseline — the Dixon-Coles goal model is the primary soccer engine.
        """
        expected = self.expected_home_score(home, away, neutral=neutral)
        if not self.config.draws_possible:
            return {"home": expected, "away": 1.0 - expected}

        width = self.config.draw_width
        draw = width * (1.0 - 2.0 * abs(expected - 0.5))
        draw = min(max(draw, 0.02), 0.45)
        home_prob = expected - draw / 2.0
        away_prob = 1.0 - home_prob - draw
        floor = 1e-4
        home_prob, away_prob = max(home_prob, floor), max(away_prob, floor)
        total = home_prob + draw + away_prob
        return {"home": home_prob / total, "draw": draw / total, "away": away_prob / total}

    # --------------------------------------------------------------- updating

    def update(self, home: str, away: str, home_score: float, away_score: float,
               *, neutral: bool = False) -> tuple[float, float]:
        """Apply one result.  Returns the pre-game ratings actually used."""
        home_rating = self.rating(home)
        away_rating = self.rating(away)

        expected = self.expected_home_score(home, away, neutral=neutral)
        if home_score > away_score:
            actual = 1.0
        elif home_score < away_score:
            actual = 0.0
        else:
            actual = 0.5

        k = self.config.k
        if self.config.use_mov:
            margin = abs(home_score - away_score)
            elo_diff = (home_rating + (0 if neutral else self.config.home_advantage)
                        - away_rating)
            winner_diff = elo_diff if actual == 1.0 else -elo_diff
            # 538's multiplier: damps runaway ratings from blowouts by strong
            # favourites, which would otherwise be double counted.
            k = k * math.log(margin + 1.0) * (self.config.mov_shrink /
                                              ((winner_diff * 0.001) + self.config.mov_shrink))

        change = k * (actual - expected)
        self.ratings.setdefault(home, EloState(self.config.initial)).rating = home_rating + change
        self.ratings.setdefault(away, EloState(self.config.initial)).rating = away_rating - change
        self.ratings[home].games += 1
        self.ratings[away].games += 1
        return home_rating, away_rating


# --------------------------------------------------------------------------
# Opponent-adjusted efficiency
# --------------------------------------------------------------------------

@dataclass
class AdjustedRatingConfig:
    learning_rate: float = 0.08
    league_mean: float = 112.0        # points per 100 possessions, NBA-scale
    #: Shrink toward the league mean at each season boundary.
    season_carry: float = 0.70
    #: Ignore the first games of a team's season for stability of the update.
    warmup_games: int = 0


@dataclass
class TeamEfficiency:
    offense: float = 0.0      # points per 100 above league average
    defense: float = 0.0      # points allowed per 100 above league average (lower is better)
    games: int = 0

    @property
    def net(self) -> float:
        return self.offense - self.defense


class AdjustedEfficiency:
    """Online opponent adjustment of offensive/defensive efficiency.

    A 120 offensive rating against the league's worst defence is not the same
    achievement as 120 against its best.  This fits the additive model

        observed_ortg ≈ league_mean + offense[team] + defense[opponent]

    by stochastic gradient descent over games in time order, so the rating
    available before a game only ever reflects earlier games.
    """

    def __init__(self, config: AdjustedRatingConfig | None = None) -> None:
        self.config = config or AdjustedRatingConfig()
        self.teams: dict[str, TeamEfficiency] = {}
        self._season: str | None = None

    def get(self, team: str) -> TeamEfficiency:
        return self.teams.setdefault(team, TeamEfficiency())

    def start_season(self, season: str) -> None:
        if self._season is None:
            self._season = season
            return
        if season == self._season:
            return
        carry = self.config.season_carry
        for efficiency in self.teams.values():
            efficiency.offense *= carry
            efficiency.defense *= carry
            efficiency.games = 0
        self._season = season

    def expected_ortg(self, team: str, opponent: str) -> float:
        return (self.config.league_mean + self.get(team).offense
                + self.get(opponent).defense)

    def update(self, team: str, opponent: str, ortg: float) -> None:
        """One observation of ``team``'s offence against ``opponent``'s defence."""
        if ortg is None or not math.isfinite(ortg):
            return
        team_state = self.get(team)
        opponent_state = self.get(opponent)
        error = ortg - self.expected_ortg(team, opponent)
        step = self.config.learning_rate * error
        team_state.offense += step
        opponent_state.defense += step
        team_state.games += 1

    def update_game(self, home: str, away: str, home_ortg: float, away_ortg: float) -> None:
        self.update(home, away, home_ortg)
        self.update(away, home, away_ortg)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            team: {"offense": e.offense, "defense": e.defense, "net": e.net, "games": e.games}
            for team, e in self.teams.items()
        }


# --------------------------------------------------------------------------
# Bayesian shrinkage
# --------------------------------------------------------------------------

def shrink_to_prior(current_value: float | None, current_games: int,
                    prior_value: float | None, *, prior_games: float | None = None) -> float | None:
    """Blend current-season evidence with a prior-season estimate.

    ``weight = n / (n + prior_games)`` is the standard conjugate-normal
    shrinkage: with no games played the prior dominates entirely, and its
    influence decays as evidence accumulates.  ``prior_games`` is the number of
    games at which the two carry equal weight, and is configurable rather than
    an arbitrary constant.
    """
    prior_games = settings.model.shrinkage_prior_games if prior_games is None else prior_games
    if current_value is None and prior_value is None:
        return None
    if current_value is None:
        return prior_value
    if prior_value is None:
        return current_value
    weight = current_games / (current_games + prior_games) if current_games >= 0 else 0.0
    return weight * current_value + (1.0 - weight) * prior_value
