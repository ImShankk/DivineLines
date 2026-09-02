"""NBA feature engineering.

Written as a single chronological pass over team-games that maintains per-team
state.  Each row's features are emitted from the state *before* the game is
applied, so leakage is structurally impossible rather than something to be
audited after the fact — the previous implementation relied on remembering to
call ``.shift(1)`` in the right places, which is exactly the kind of thing that
breaks silently.

Also fixed here:

* rolling windows respected only game order, never season boundaries, so a
  team's first game of October was described by the previous April's form;
* opponent columns were built with ``groupby.transform(lambda x: x[::-1])``,
  which silently produces nonsense if a game ever has other than two rows;
* ``H2H_WIN_PCT`` was hard-coded to 0.50 — a dead feature the model was still
  being asked to split on.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import settings
from ..identity import NBA_BY_ABBR
from ..logging_setup import get_logger
from .ratings import (
    AdjustedEfficiency,
    AdjustedRatingConfig,
    EloConfig,
    EloRatings,
    shrink_to_prior,
)

log = get_logger(__name__)

FEATURE_SET_VERSION = "nba_v2.0"

#: Per-game efficiency metrics tracked for every team.
METRICS = (
    "ortg", "drtg", "net_rating", "pace", "efg", "ts", "tov_rate", "oreb_pct",
    "ft_rate", "fg3a_rate", "opp_efg", "opp_fg3_pct", "margin", "win",
)


@dataclass
class NbaFeatureConfig:
    windows: tuple[int, ...] = (5, 10, 20)
    ewma_halflife: float = 8.0
    #: Rows with fewer prior games than this are dropped from training.
    min_prior_games: int = 5
    prior_season_weight_games: float | None = None
    elo: EloConfig = field(default_factory=lambda: EloConfig(
        k=settings.model.elo_k_nba,
        home_advantage=60.0,
        season_carry=settings.model.elo_season_regression,
        use_mov=True,
    ))
    adjusted: AdjustedRatingConfig = field(default_factory=AdjustedRatingConfig)
    version: str = FEATURE_SET_VERSION


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, used for travel fatigue features."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


TZ_OFFSET = {"ET": 0, "CT": -1, "MT": -2, "PT": -3}


def compute_game_metrics(row: pd.Series, opp: pd.Series) -> dict[str, float]:
    """Advanced metrics for one team in one game."""
    fga, fta, oreb, tov = row["fga"], row["fta"], row["oreb"], row["tov"]
    poss = fga - oreb + tov + 0.44 * fta
    opp_poss = opp["fga"] - opp["oreb"] + opp["tov"] + 0.44 * opp["fta"]
    poss = max(poss, 1.0)
    opp_poss = max(opp_poss, 1.0)

    minutes = row.get("min") or 240.0
    periods = max(minutes / 5.0, 1.0)   # team minutes / 5 players = game minutes

    metrics = {
        "poss": poss,
        "ortg": 100.0 * row["pts"] / poss,
        "drtg": 100.0 * opp["pts"] / opp_poss,
        "pace": 48.0 * ((poss + opp_poss) / 2.0) / periods,
        "efg": (row["fgm"] + 0.5 * row["fg3m"]) / fga if fga else 0.0,
        "ts": row["pts"] / (2.0 * (fga + 0.44 * fta)) if (fga + fta) else 0.0,
        "tov_rate": tov / poss,
        "oreb_pct": oreb / (oreb + opp["dreb"]) if (oreb + opp["dreb"]) else 0.0,
        "ft_rate": fta / fga if fga else 0.0,
        "fg3a_rate": row["fg3a"] / fga if fga else 0.0,
        "opp_efg": (opp["fgm"] + 0.5 * opp["fg3m"]) / opp["fga"] if opp["fga"] else 0.0,
        "opp_fg3_pct": opp["fg3m"] / opp["fg3a"] if opp["fg3a"] else 0.0,
        "margin": row["pts"] - opp["pts"],
        "win": 1.0 if row["pts"] > opp["pts"] else 0.0,
    }
    metrics["net_rating"] = metrics["ortg"] - metrics["drtg"]
    return metrics


@dataclass
class TeamState:
    """Everything known about a team *before* its next game."""

    history: deque = field(default_factory=lambda: deque(maxlen=40))
    season: str | None = None
    season_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    season_games: int = 0
    prior_season: dict[str, float] = field(default_factory=dict)
    prior_season_games: int = 0
    ewma: dict[str, float] = field(default_factory=dict)
    last_game_date: pd.Timestamp | None = None
    last_venue: tuple[float, float] | None = None
    last_tz: str | None = None
    last_was_home: bool | None = None
    recent_dates: deque = field(default_factory=lambda: deque(maxlen=12))
    career_games: int = 0

    def rollover(self, season: str) -> None:
        """Archive the finished season and reset current-season accumulators."""
        if self.season is not None and self.season != season:
            if self.season_games:
                self.prior_season = {
                    k: v / self.season_games for k, v in self.season_totals.items()
                }
                self.prior_season_games = self.season_games
            self.season_totals = defaultdict(float)
            self.season_games = 0
            self.history.clear()
            self.ewma = {}
        self.season = season

    def rolling_mean(self, metric: str, window: int) -> float | None:
        if not self.history:
            return None
        values = [g[metric] for g in list(self.history)[-window:] if metric in g]
        return float(np.mean(values)) if values else None

    def season_mean(self, metric: str) -> float | None:
        if not self.season_games:
            return None
        return self.season_totals[metric] / self.season_games

    def apply(self, metrics: dict[str, float], halflife: float) -> None:
        self.history.append(metrics)
        for key, value in metrics.items():
            self.season_totals[key] += value
        self.season_games += 1
        self.career_games += 1
        alpha = 1.0 - 0.5 ** (1.0 / halflife)
        for key, value in metrics.items():
            self.ewma[key] = value if key not in self.ewma else (
                alpha * value + (1 - alpha) * self.ewma[key]
            )


class NbaFeatureBuilder:
    """Builds a leak-free, season-aware matchup dataset."""

    def __init__(self, config: NbaFeatureConfig | None = None) -> None:
        self.config = config or NbaFeatureConfig()
        self.elo = EloRatings(self.config.elo)
        self.adjusted = AdjustedEfficiency(self.config.adjusted)
        self.states: dict[str, TeamState] = defaultdict(TeamState)
        self.h2h: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    # ------------------------------------------------------------------ main

    def build(self, team_games: pd.DataFrame, *, include_incomplete: bool = False
              ) -> pd.DataFrame:
        """One row per game, chronologically ordered, with the target attached."""
        if team_games.empty:
            return pd.DataFrame()

        frame = team_games.sort_values(["game_date", "game_uid", "is_home"],
                                       ascending=[True, True, False])
        rows: list[dict[str, Any]] = []

        for game_uid, group in frame.groupby("game_uid", sort=False):
            if len(group) != 2:
                log.warning("skipping game without two team rows",
                            extra={"game_uid": game_uid, "rows": len(group)})
                continue
            home = group[group["is_home"] == 1]
            away = group[group["is_home"] == 0]
            if len(home) != 1 or len(away) != 1:
                continue
            home_row, away_row = home.iloc[0], away.iloc[0]

            season = str(home_row["season"])
            self.elo.start_season(season)
            self.adjusted.start_season(season)
            for uid in (home_row["team_uid"], away_row["team_uid"]):
                self.states[uid].rollover(season)

            features = self._emit_features(home_row, away_row, season)
            if features is not None and (include_incomplete or features["_eligible"]):
                rows.append(features)

            self._apply_result(home_row, away_row)

        dataset = pd.DataFrame(rows)
        if dataset.empty:
            return dataset
        dataset = dataset.drop(columns=["_eligible"])
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        return dataset.sort_values("game_date").reset_index(drop=True)

    # -------------------------------------------------------------- features

    def _emit_features(self, home_row: pd.Series, away_row: pd.Series,
                       season: str) -> dict[str, Any] | None:
        home_uid, away_uid = home_row["team_uid"], away_row["team_uid"]
        home_state, away_state = self.states[home_uid], self.states[away_uid]
        game_date = pd.Timestamp(home_row["game_date"])
        neutral = bool(home_row.get("neutral_site", 0))

        record: dict[str, Any] = {
            "game_uid": home_row["game_uid"],
            "game_date": game_date,
            "season": season,
            "home_team_uid": home_uid,
            "away_team_uid": away_uid,
            "neutral_site": int(neutral),
        }

        # --- form: rolling windows, EWMA and shrunk season form ------------
        for metric in ("ortg", "drtg", "net_rating", "pace", "efg", "ts",
                       "tov_rate", "oreb_pct", "ft_rate", "fg3a_rate",
                       "opp_efg", "opp_fg3_pct", "win"):
            for window in self.config.windows:
                home_value = home_state.rolling_mean(metric, window)
                away_value = away_state.rolling_mean(metric, window)
                record[f"diff_{metric}_r{window}"] = _diff(home_value, away_value)
            record[f"diff_{metric}_ewma"] = _diff(
                home_state.ewma.get(metric), away_state.ewma.get(metric)
            )
            home_shrunk = shrink_to_prior(
                home_state.season_mean(metric), home_state.season_games,
                home_state.prior_season.get(metric),
                prior_games=self.config.prior_season_weight_games,
            )
            away_shrunk = shrink_to_prior(
                away_state.season_mean(metric), away_state.season_games,
                away_state.prior_season.get(metric),
                prior_games=self.config.prior_season_weight_games,
            )
            record[f"diff_{metric}_shrunk"] = _diff(home_shrunk, away_shrunk)

        # --- ratings -------------------------------------------------------
        record["elo_home"] = self.elo.rating(home_uid)
        record["elo_away"] = self.elo.rating(away_uid)
        record["diff_elo"] = record["elo_home"] - record["elo_away"]
        record["elo_home_prob"] = self.elo.expected_home_score(
            home_uid, away_uid, neutral=neutral
        )
        home_eff = self.adjusted.get(home_uid)
        away_eff = self.adjusted.get(away_uid)
        record["diff_adj_offense"] = home_eff.offense - away_eff.offense
        record["diff_adj_defense"] = home_eff.defense - away_eff.defense
        record["diff_adj_net"] = home_eff.net - away_eff.net
        record["adj_matchup_home"] = home_eff.offense + away_eff.defense
        record["adj_matchup_away"] = away_eff.offense + home_eff.defense

        # --- schedule / fatigue / travel -----------------------------------
        home_schedule = self._schedule_features(home_state, game_date, home_uid, True, neutral)
        away_schedule = self._schedule_features(away_state, game_date, home_uid, False, neutral)
        for key in home_schedule:
            record[f"home_{key}"] = home_schedule[key]
            record[f"away_{key}"] = away_schedule[key]
            record[f"diff_{key}"] = _diff(home_schedule[key], away_schedule[key])

        # --- season context ------------------------------------------------
        record["home_season_games"] = home_state.season_games
        record["away_season_games"] = away_state.season_games
        record["min_season_games"] = min(home_state.season_games, away_state.season_games)
        record["diff_season_games"] = home_state.season_games - away_state.season_games
        record["early_season"] = int(record["min_season_games"] < 10)

        # --- head to head (real, not a placeholder) ------------------------
        record.update(self._h2h_features(home_uid, away_uid, game_date))

        # --- target --------------------------------------------------------
        home_score = home_row.get("pts")
        away_score = away_row.get("pts")
        record["home_score"] = float(home_score) if pd.notna(home_score) else None
        record["away_score"] = float(away_score) if pd.notna(away_score) else None
        record["home_win"] = (
            int(home_score > away_score) if pd.notna(home_score) and pd.notna(away_score) else None
        )
        record["margin"] = (
            float(home_score - away_score) if pd.notna(home_score) and pd.notna(away_score) else None
        )
        record["total_points"] = (
            float(home_score + away_score) if pd.notna(home_score) and pd.notna(away_score) else None
        )

        record["_eligible"] = (
            min(home_state.career_games, away_state.career_games) >= self.config.min_prior_games
            and record["home_win"] is not None
        )
        return record

    def _schedule_features(self, state: TeamState, game_date: pd.Timestamp,
                           home_uid: str, is_home: bool, neutral: bool) -> dict[str, float]:
        venue_team = NBA_BY_ABBR.get(home_uid.split(":")[-1])
        venue = (venue_team.lat, venue_team.lon) if venue_team else None
        venue_tz = venue_team.tz if venue_team else None

        if state.last_game_date is None:
            rest_days = 3.0    # season opener: treated as fully rested
        else:
            rest_days = float((game_date - state.last_game_date).days)

        travel = 0.0
        if state.last_venue and venue:
            travel = haversine_km(*state.last_venue, *venue)

        tz_shift = 0.0
        if state.last_tz and venue_tz:
            tz_shift = abs(TZ_OFFSET.get(venue_tz, 0) - TZ_OFFSET.get(state.last_tz, 0))

        window_start = game_date - pd.Timedelta(days=7)
        games_last_7 = sum(1 for d in state.recent_dates if d > window_start)
        window_start_4 = game_date - pd.Timedelta(days=4)
        games_last_4 = sum(1 for d in state.recent_dates if d > window_start_4)

        return {
            "rest_days": min(max(rest_days, 0.0), 7.0),
            "is_b2b": 1.0 if rest_days <= 1.0 else 0.0,
            "games_last_7": float(games_last_7),
            "three_in_four": 1.0 if games_last_4 >= 2 else 0.0,
            "travel_km": travel,
            "tz_shift": tz_shift,
            "is_home": 1.0 if (is_home and not neutral) else 0.0,
            "road_trip": 0.0 if state.last_was_home is None else float(
                (not is_home) and (state.last_was_home is False)
            ),
        }

    def _h2h_features(self, home_uid: str, away_uid: str,
                      game_date: pd.Timestamp) -> dict[str, float]:
        key = tuple(sorted((home_uid, away_uid)))
        meetings = self.h2h.get(key, [])
        recent = [m for m in meetings if (game_date - m["date"]).days <= 900]
        if not recent:
            # No prior meetings: report the absence explicitly instead of
            # inventing a 0.50 win rate the model would treat as evidence.
            return {"h2h_games": 0.0, "h2h_home_win_pct": np.nan, "h2h_avg_margin": np.nan}
        wins = [1.0 if m["winner"] == home_uid else 0.0 for m in recent]
        margins = [m["margin"] if m["home"] == home_uid else -m["margin"] for m in recent]
        return {
            "h2h_games": float(len(recent)),
            "h2h_home_win_pct": float(np.mean(wins)),
            "h2h_avg_margin": float(np.mean(margins)),
        }

    # ---------------------------------------------------------------- update

    def _apply_result(self, home_row: pd.Series, away_row: pd.Series) -> None:
        if pd.isna(home_row.get("pts")) or pd.isna(away_row.get("pts")):
            return

        home_metrics = compute_game_metrics(home_row, away_row)
        away_metrics = compute_game_metrics(away_row, home_row)
        home_uid, away_uid = home_row["team_uid"], away_row["team_uid"]
        game_date = pd.Timestamp(home_row["game_date"])
        neutral = bool(home_row.get("neutral_site", 0))

        self.elo.update(home_uid, away_uid, home_row["pts"], away_row["pts"], neutral=neutral)
        self.adjusted.update_game(home_uid, away_uid,
                                  home_metrics["ortg"], away_metrics["ortg"])

        venue_team = NBA_BY_ABBR.get(home_uid.split(":")[-1])
        venue = (venue_team.lat, venue_team.lon) if venue_team else None
        venue_tz = venue_team.tz if venue_team else None

        for uid, metrics, is_home in ((home_uid, home_metrics, True),
                                      (away_uid, away_metrics, False)):
            state = self.states[uid]
            state.apply(metrics, self.config.ewma_halflife)
            state.last_game_date = game_date
            state.last_venue = venue
            state.last_tz = venue_tz
            state.last_was_home = is_home
            state.recent_dates.append(game_date)

        key = tuple(sorted((home_uid, away_uid)))
        self.h2h[key].append(
            {
                "date": game_date,
                "home": home_uid,
                "winner": home_uid if home_row["pts"] > away_row["pts"] else away_uid,
                "margin": float(home_row["pts"] - away_row["pts"]),
            }
        )

    # ------------------------------------------------------------ prediction

    def upcoming_features(self, home_uid: str, away_uid: str, game_date: pd.Timestamp,
                          season: str, *, neutral: bool = False) -> dict[str, Any]:
        """Features for a game that has not been played.

        Uses exactly the same code path as training, so a live prediction can
        never be built from a different definition than the model was fitted on.
        """
        home_row = pd.Series({
            "game_uid": f"pending:{home_uid}:{away_uid}:{game_date:%Y-%m-%d}",
            "game_date": game_date, "season": season, "team_uid": home_uid,
            "is_home": 1, "neutral_site": int(neutral), "pts": np.nan,
        })
        away_row = pd.Series({
            "game_uid": home_row["game_uid"], "game_date": game_date, "season": season,
            "team_uid": away_uid, "is_home": 0, "neutral_site": int(neutral), "pts": np.nan,
        })
        self.elo.start_season(season)
        self.adjusted.start_season(season)
        for uid in (home_uid, away_uid):
            self.states[uid].rollover(season)
        features = self._emit_features(home_row, away_row, season)
        features.pop("_eligible", None)
        return features


def _diff(home_value: float | None, away_value: float | None) -> float:
    if home_value is None or away_value is None:
        return np.nan
    return float(home_value) - float(away_value)


def build_nba_dataset(team_games: pd.DataFrame, config: NbaFeatureConfig | None = None
                      ) -> tuple[pd.DataFrame, NbaFeatureBuilder]:
    """Convenience wrapper returning both the dataset and the fitted state."""
    builder = NbaFeatureBuilder(config)
    dataset = builder.build(team_games)
    log.info("built NBA feature dataset",
             extra={"rows": len(dataset), "features": len(dataset.columns),
                    "version": (config or NbaFeatureConfig()).version})
    return dataset, builder
