"""Soccer feature engineering.

Soccer is not basketball with fewer points, so this is not a port of the NBA
builder.  The differences that drive the design:

* three outcomes, and the draw is a real, modellable state;
* low scoring, so goals are a noisy signal and shot volume/accuracy carry
  information that goals alone do not;
* promotion and relegation mean a club can appear in a division with **no**
  history in it — treating such a club as an established side is one of the
  fastest ways to lose money in August;
* clubs play in several competitions, so identity must be league-independent.

As with the NBA builder this is a single chronological pass: every feature is
emitted from state that existed before kick-off.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import SOCCER_LEAGUES, settings
from ..logging_setup import get_logger
from .ratings import EloConfig, EloRatings, shrink_to_prior

log = get_logger(__name__)

FEATURE_SET_VERSION = "soccer_v2.0"

METRICS = (
    "goals_for", "goals_against", "shots_for", "shots_against",
    "sot_for", "sot_against", "corners_for", "cards_for",
    "points", "shot_accuracy", "conversion", "goal_diff",
)


@dataclass
class SoccerFeatureConfig:
    windows: tuple[int, ...] = (5, 10, 20)
    ewma_halflife: float = 8.0
    min_prior_matches: int = 5
    prior_season_weight_matches: float | None = 8.0
    elo: EloConfig = field(default_factory=lambda: EloConfig(
        k=settings.model.elo_k_soccer,
        home_advantage=60.0,
        season_carry=settings.model.elo_season_regression,
        use_mov=True,
        draws_possible=True,
    ))
    #: Matches in a division below which a club is treated as new to it.
    promotion_threshold: int = 15
    version: str = FEATURE_SET_VERSION


@dataclass
class ClubState:
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    season: str | None = None
    season_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    season_matches: int = 0
    prior_season: dict[str, float] = field(default_factory=dict)
    ewma: dict[str, float] = field(default_factory=dict)
    last_match_date: pd.Timestamp | None = None
    recent_dates: deque = field(default_factory=lambda: deque(maxlen=12))
    matches_by_league: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    career_matches: int = 0

    def rollover(self, season: str) -> None:
        if self.season is not None and self.season != season:
            if self.season_matches:
                self.prior_season = {
                    k: v / self.season_matches for k, v in self.season_totals.items()
                }
            self.season_totals = defaultdict(float)
            self.season_matches = 0
            self.ewma = {}
        self.season = season

    def rolling_mean(self, metric: str, window: int) -> float | None:
        if not self.history:
            return None
        values = [m[metric] for m in list(self.history)[-window:] if metric in m]
        return float(np.mean(values)) if values else None

    def season_mean(self, metric: str) -> float | None:
        if not self.season_matches:
            return None
        return self.season_totals[metric] / self.season_matches

    def apply(self, metrics: dict[str, float], league_id: str, halflife: float) -> None:
        self.history.append(metrics)
        for key, value in metrics.items():
            self.season_totals[key] += value
        self.season_matches += 1
        self.career_matches += 1
        self.matches_by_league[league_id] += 1
        alpha = 1.0 - 0.5 ** (1.0 / halflife)
        for key, value in metrics.items():
            self.ewma[key] = value if key not in self.ewma else (
                alpha * value + (1 - alpha) * self.ewma[key]
            )


def match_metrics(goals_for: float, goals_against: float, row: pd.Series,
                  side: str) -> dict[str, float]:
    """Per-match metrics for one club."""
    other = "away" if side == "home" else "home"
    shots_for = _numeric(row.get(f"{side}_shots"))
    shots_against = _numeric(row.get(f"{other}_shots"))
    sot_for = _numeric(row.get(f"{side}_sot"))
    sot_against = _numeric(row.get(f"{other}_sot"))

    points = 3.0 if goals_for > goals_against else (1.0 if goals_for == goals_against else 0.0)
    return {
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "goal_diff": float(goals_for - goals_against),
        "shots_for": shots_for if shots_for is not None else np.nan,
        "shots_against": shots_against if shots_against is not None else np.nan,
        "sot_for": sot_for if sot_for is not None else np.nan,
        "sot_against": sot_against if sot_against is not None else np.nan,
        "corners_for": _numeric(row.get(f"{side}_corners"), default=np.nan),
        "cards_for": (_numeric(row.get(f"{side}_yellow"), default=0.0) or 0.0)
        + 2.0 * (_numeric(row.get(f"{side}_red"), default=0.0) or 0.0),
        "points": points,
        # Shot accuracy and conversion are the closest honest proxies for shot
        # quality available in this source.  They are NOT xG: football-data
        # publishes no expected-goals column, and the platform never presents
        # these as such.
        "shot_accuracy": (sot_for / shots_for) if shots_for else np.nan,
        "conversion": (goals_for / sot_for) if sot_for else np.nan,
    }


def _numeric(value: Any, default: float | None = None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _league_window() -> deque:
    """Rolling window of one league-season's worth of matches.

    A module-level factory rather than a lambda: builder state is pickled into
    the model registry, and lambdas are not picklable.
    """
    return deque(maxlen=380)


class SoccerFeatureBuilder:
    """Chronological feature builder covering every configured competition."""

    def __init__(self, config: SoccerFeatureConfig | None = None) -> None:
        self.config = config or SoccerFeatureConfig()
        self.elo = EloRatings(self.config.elo)
        self.states: dict[str, ClubState] = defaultdict(ClubState)
        self.h2h: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        #: Rolling home-advantage estimate per league, learned rather than fixed.
        self.league_home_points: dict[str, deque] = defaultdict(_league_window)
        self.league_goals: dict[str, deque] = defaultdict(_league_window)

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        if matches.empty:
            return pd.DataFrame()

        ordered = matches.sort_values(["game_date", "game_uid"]).reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        current_season: str | None = None

        for _, row in ordered.iterrows():
            season = str(row["season"])
            if season != current_season:
                self.elo.start_season(season)
                current_season = season
            home_uid, away_uid = row["home_team_uid"], row["away_team_uid"]
            for uid in (home_uid, away_uid):
                self.states[uid].rollover(season)

            features = self._emit(row)
            if features is not None:
                rows.append(features)
            self._apply(row)

        dataset = pd.DataFrame(rows)
        if dataset.empty:
            return dataset
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        return dataset.sort_values("game_date").reset_index(drop=True)

    # -------------------------------------------------------------- features

    def _emit(self, row: pd.Series) -> dict[str, Any] | None:
        home_uid, away_uid = row["home_team_uid"], row["away_team_uid"]
        home, away = self.states[home_uid], self.states[away_uid]
        league_id = str(row["league_id"])
        match_date = pd.Timestamp(row["game_date"])

        record: dict[str, Any] = {
            "game_uid": row["game_uid"],
            "game_date": match_date,
            "league_id": league_id,
            "season": str(row["season"]),
            "home_team_uid": home_uid,
            "away_team_uid": away_uid,
            "home_name": row.get("home_name"),
            "away_name": row.get("away_name"),
        }

        for metric in ("goals_for", "goals_against", "goal_diff", "shots_for",
                       "shots_against", "sot_for", "sot_against", "points",
                       "shot_accuracy", "conversion", "corners_for", "cards_for"):
            for window in self.config.windows:
                record[f"home_{metric}_r{window}"] = home.rolling_mean(metric, window)
                record[f"away_{metric}_r{window}"] = away.rolling_mean(metric, window)
                record[f"diff_{metric}_r{window}"] = _diff(
                    record[f"home_{metric}_r{window}"], record[f"away_{metric}_r{window}"]
                )
            record[f"diff_{metric}_ewma"] = _diff(home.ewma.get(metric), away.ewma.get(metric))
            home_shrunk = shrink_to_prior(
                home.season_mean(metric), home.season_matches, home.prior_season.get(metric),
                prior_games=self.config.prior_season_weight_matches,
            )
            away_shrunk = shrink_to_prior(
                away.season_mean(metric), away.season_matches, away.prior_season.get(metric),
                prior_games=self.config.prior_season_weight_matches,
            )
            record[f"diff_{metric}_shrunk"] = _diff(home_shrunk, away_shrunk)

        # Attack vs defence matchup: a team's scoring against this opponent's
        # concession rate, which is what actually determines goal expectation.
        record["home_attack_vs_defence"] = _product(
            home.rolling_mean("goals_for", 10), away.rolling_mean("goals_against", 10)
        )
        record["away_attack_vs_defence"] = _product(
            away.rolling_mean("goals_for", 10), home.rolling_mean("goals_against", 10)
        )

        record["elo_home"] = self.elo.rating(home_uid)
        record["elo_away"] = self.elo.rating(away_uid)
        record["diff_elo"] = record["elo_home"] - record["elo_away"]
        elo_probs = self.elo.win_probabilities(home_uid, away_uid)
        record["elo_prob_home"] = elo_probs["home"]
        record["elo_prob_draw"] = elo_probs.get("draw", 0.0)
        record["elo_prob_away"] = elo_probs["away"]

        record["home_rest_days"] = _rest_days(home, match_date)
        record["away_rest_days"] = _rest_days(away, match_date)
        record["diff_rest_days"] = record["home_rest_days"] - record["away_rest_days"]
        record["home_matches_14d"] = _matches_within(home, match_date, 14)
        record["away_matches_14d"] = _matches_within(away, match_date, 14)
        record["diff_congestion"] = record["home_matches_14d"] - record["away_matches_14d"]

        # Promotion / relegation and league context.
        home_league_matches = home.matches_by_league.get(league_id, 0)
        away_league_matches = away.matches_by_league.get(league_id, 0)
        record["home_league_matches"] = home_league_matches
        record["away_league_matches"] = away_league_matches
        record["home_is_new_to_league"] = int(home_league_matches < self.config.promotion_threshold)
        record["away_is_new_to_league"] = int(away_league_matches < self.config.promotion_threshold)
        record["any_new_to_league"] = int(
            record["home_is_new_to_league"] or record["away_is_new_to_league"]
        )
        record["league_strength"] = SOCCER_LEAGUES.get(league_id, {}).get("strength", 0.8)

        home_points = self.league_home_points[league_id]
        record["league_home_advantage"] = float(np.mean(home_points)) if home_points else 1.5
        league_goals = self.league_goals[league_id]
        record["league_avg_goals"] = float(np.mean(league_goals)) if league_goals else 2.7

        record.update(self._h2h(home_uid, away_uid, match_date))

        home_goals, away_goals = row.get("home_score"), row.get("away_score")
        if pd.isna(home_goals) or pd.isna(away_goals):
            record["home_goals"] = record["away_goals"] = None
            record["outcome"] = None
            record["outcome_selection"] = None
        else:
            record["home_goals"] = float(home_goals)
            record["away_goals"] = float(away_goals)
            record["total_goals"] = float(home_goals + away_goals)
            if home_goals > away_goals:
                record["outcome"], record["outcome_selection"] = 0, "home"
            elif home_goals == away_goals:
                record["outcome"], record["outcome_selection"] = 1, "draw"
            else:
                record["outcome"], record["outcome_selection"] = 2, "away"

        record["eligible"] = int(
            min(home.career_matches, away.career_matches) >= self.config.min_prior_matches
        )
        return record

    def _h2h(self, home_uid: str, away_uid: str, match_date: pd.Timestamp) -> dict[str, float]:
        key = tuple(sorted((home_uid, away_uid)))
        meetings = [m for m in self.h2h.get(key, [])
                    if (match_date - m["date"]).days <= 1500]
        if not meetings:
            return {"h2h_matches": 0.0, "h2h_home_points": np.nan, "h2h_goal_diff": np.nan}
        points = []
        diffs = []
        for meeting in meetings:
            if meeting["home"] == home_uid:
                diff = meeting["home_goals"] - meeting["away_goals"]
            else:
                diff = meeting["away_goals"] - meeting["home_goals"]
            diffs.append(diff)
            points.append(3.0 if diff > 0 else (1.0 if diff == 0 else 0.0))
        return {
            "h2h_matches": float(len(meetings)),
            "h2h_home_points": float(np.mean(points)),
            "h2h_goal_diff": float(np.mean(diffs)),
        }

    # ---------------------------------------------------------------- update

    def _apply(self, row: pd.Series) -> None:
        home_goals, away_goals = row.get("home_score"), row.get("away_score")
        if pd.isna(home_goals) or pd.isna(away_goals):
            return

        home_uid, away_uid = row["home_team_uid"], row["away_team_uid"]
        league_id = str(row["league_id"])
        match_date = pd.Timestamp(row["game_date"])

        self.elo.update(home_uid, away_uid, float(home_goals), float(away_goals))

        for uid, side, goals_for, goals_against in (
            (home_uid, "home", home_goals, away_goals),
            (away_uid, "away", away_goals, home_goals),
        ):
            metrics = match_metrics(goals_for, goals_against, row, side)
            state = self.states[uid]
            state.apply(metrics, league_id, self.config.ewma_halflife)
            state.last_match_date = match_date
            state.recent_dates.append(match_date)

        home_points = 3.0 if home_goals > away_goals else (1.0 if home_goals == away_goals else 0.0)
        self.league_home_points[league_id].append(home_points)
        self.league_goals[league_id].append(float(home_goals + away_goals))

        key = tuple(sorted((home_uid, away_uid)))
        self.h2h[key].append(
            {"date": match_date, "home": home_uid,
             "home_goals": float(home_goals), "away_goals": float(away_goals)}
        )

    # ------------------------------------------------------------ prediction

    def upcoming_features(self, row: pd.Series) -> dict[str, Any]:
        """Features for an unplayed fixture, via the training code path."""
        season = str(row["season"])
        self.elo.start_season(season)
        for uid in (row["home_team_uid"], row["away_team_uid"]):
            self.states[uid].rollover(season)
        return self._emit(row)


def _diff(home_value: float | None, away_value: float | None) -> float:
    if home_value is None or away_value is None or pd.isna(home_value) or pd.isna(away_value):
        return np.nan
    return float(home_value) - float(away_value)


def _product(a: float | None, b: float | None) -> float:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) * float(b)


def _rest_days(state: ClubState, match_date: pd.Timestamp) -> float:
    if state.last_match_date is None:
        return 7.0
    return float(min(max((match_date - state.last_match_date).days, 0), 21))


def _matches_within(state: ClubState, match_date: pd.Timestamp, days: int) -> float:
    cutoff = match_date - pd.Timedelta(days=days)
    return float(sum(1 for d in state.recent_dates if d > cutoff))


def build_soccer_dataset(matches: pd.DataFrame, config: SoccerFeatureConfig | None = None
                         ) -> tuple[pd.DataFrame, SoccerFeatureBuilder]:
    builder = SoccerFeatureBuilder(config)
    dataset = builder.build(matches)
    log.info("built soccer feature dataset",
             extra={"rows": len(dataset), "columns": len(dataset.columns)})
    return dataset, builder
