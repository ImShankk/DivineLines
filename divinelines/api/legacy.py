"""Backwards-compatible v1 matchup endpoint.

The original ``POST /api/predict`` took ``{home, away}`` team abbreviations and
returned a win probability, a quant-edge block and rolling-average metrics.
That contract is preserved exactly — existing clients keep working — while the
numbers behind it are now produced by the calibrated ensemble, priced against
de-vigged multi-book consensus odds, and returned with the model version that
produced them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..betting.ev import expected_value
from ..betting.kelly import recommend_stake
from ..betting.odds_math import build_consensus, decimal_to_american
from ..config import nba_season_for_date
from ..db.connection import query_df
from ..db.repository import latest_odds, load_nba_team_games
from ..logging_setup import get_logger
from ..models.registry import latest_model_id, load_artifact

log = get_logger(__name__)

ROLLING_WINDOW = 10


def _team_rolling_stats(team_games: pd.DataFrame, team_uid: str) -> dict[str, float]:
    """Rolling averages for the display panel (last ``ROLLING_WINDOW`` games)."""
    rows = team_games[team_games["team_uid"] == team_uid].tail(ROLLING_WINDOW)
    if rows.empty:
        return {}

    opponents = team_games[team_games["game_uid"].isin(rows["game_uid"])]
    opponent_rows = opponents[opponents["team_uid"] != team_uid]

    possessions = (rows["fga"] - rows["oreb"] + rows["tov"] + 0.44 * rows["fta"]).clip(lower=1)
    opponent_possessions = (
        opponent_rows["fga"] - opponent_rows["oreb"] + opponent_rows["tov"]
        + 0.44 * opponent_rows["fta"]
    ).clip(lower=1)

    ortg = float((100 * rows["pts"] / possessions).mean())
    drtg = float((100 * opponent_rows["pts"].to_numpy() /
                  opponent_possessions.to_numpy()).mean()) if len(opponent_rows) else 0.0
    return {
        "pts": float(rows["pts"].mean()),
        "opp_pts": float(opponent_rows["pts"].mean()) if len(opponent_rows) else 0.0,
        "fg3m": float(rows["fg3m"].mean()),
        "reb": float(rows["reb"].mean()),
        "ast": float(rows["ast"].mean()),
        "tov": float(rows["tov"].mean()),
        "ortg": ortg,
        "drtg": drtg,
        "net_rating": ortg - drtg,
        "pace": float(possessions.mean()),
    }


def _head_to_head(team_games: pd.DataFrame, home_uid: str, away_uid: str
                  ) -> tuple[str, dict[str, Any] | None]:
    meetings = query_df(
        """
        SELECT game_uid, game_date, home_team_uid, away_team_uid, home_score, away_score
        FROM games
        WHERE sport='nba' AND status='final'
          AND ((home_team_uid = ? AND away_team_uid = ?) OR (home_team_uid = ? AND away_team_uid = ?))
        ORDER BY game_date DESC LIMIT 1
        """,
        (home_uid, away_uid, away_uid, home_uid),
    )
    if meetings.empty:
        return "No previous meetings on record", None

    meeting = meetings.iloc[0]
    box = team_games[team_games["game_uid"] == meeting["game_uid"]]
    if len(box) != 2:
        return "No previous meetings on record", None

    home_box = box[box["team_uid"] == home_uid]
    away_box = box[box["team_uid"] == away_uid]
    if home_box.empty or away_box.empty:
        return "No previous meetings on record", None
    home_box, away_box = home_box.iloc[0], away_box.iloc[0]

    date_str = pd.Timestamp(meeting["game_date"]).strftime("%b %d, %Y")
    home_abbr = home_uid.split(":")[-1]
    away_abbr = away_uid.split(":")[-1]
    if home_box["pts"] > away_box["pts"]:
        summary = (f"{home_abbr} {int(home_box['pts'])} - {int(away_box['pts'])} "
                   f"{away_abbr} ({date_str})")
    else:
        summary = (f"{away_abbr} {int(away_box['pts'])} - {int(home_box['pts'])} "
                   f"{home_abbr} ({date_str})")

    def side(row: pd.Series, opponent: pd.Series) -> dict[str, int]:
        return {
            "pts": int(row["pts"]), "opp_pts": int(opponent["pts"]), "reb": int(row["reb"]),
            "ast": int(row["ast"]), "fg3m": int(row["fg3m"]), "tov": int(row["tov"]),
            "pf": int(row["pf"]),
        }

    return summary, {"home": side(home_box, away_box), "away": side(away_box, home_box)}


def _quant_edge(home_uid: str, away_uid: str, home_probability: float) -> dict[str, Any] | None:
    """Price the matchup off stored multi-book snapshots, if any exist."""
    today = datetime.now(timezone.utc).date()
    upcoming = query_df(
        """
        SELECT game_uid FROM games
        WHERE sport='nba' AND home_team_uid = ? AND away_team_uid = ?
          AND game_date >= ? ORDER BY game_date LIMIT 1
        """,
        (home_uid, away_uid, str(today - timedelta(days=1))),
    )
    if upcoming.empty:
        return None

    odds = latest_odds(str(upcoming["game_uid"].iloc[0]), "h2h")
    if odds.empty:
        return None

    quotes: dict[str, dict[str, float]] = {}
    for _, row in odds.iterrows():
        quotes.setdefault(row["bookmaker"], {})[row["selection"]] = float(row["price_decimal"])
    complete = {b: p for b, p in quotes.items() if {"home", "away"} <= set(p)}
    if not complete:
        return None

    consensus = build_consensus(complete)
    home_price = consensus.best_price.get("home")
    away_price = consensus.best_price.get("away")
    if not home_price or not away_price:
        return None

    home_ev = expected_value(home_probability, home_price,
                             consensus.fair_probabilities.get("home"))
    away_ev = expected_value(1 - home_probability, away_price,
                             consensus.fair_probabilities.get("away"))
    home_stake = recommend_stake(model_probability=home_probability, price_decimal=home_price,
                                 market_probability=consensus.fair_probabilities.get("home"))
    away_stake = recommend_stake(model_probability=1 - home_probability,
                                 price_decimal=away_price,
                                 market_probability=consensus.fair_probabilities.get("away"))

    return {
        # v1 fields, unchanged in name and meaning.
        "bookmaker": consensus.best_bookmaker.get("home", "consensus"),
        "home_odds": decimal_to_american(home_price),
        "away_odds": decimal_to_american(away_price),
        "home_ev": round(home_ev.ev_per_unit * 100, 2),
        "away_ev": round(away_ev.ev_per_unit * 100, 2),
        "home_kelly": round(home_stake.kelly_used * 100, 2),
        "away_kelly": round(away_stake.kelly_used * 100, 2),
        # v2 additions (extra keys are safe for existing clients).
        "home_decimal": round(home_price, 3),
        "away_decimal": round(away_price, 3),
        "market_probability_home": round(consensus.fair_probabilities.get("home", 0.0), 4),
        "market_probability_away": round(consensus.fair_probabilities.get("away", 0.0), 4),
        "overround": round(consensus.overround, 4),
        "n_bookmakers": consensus.n_bookmakers,
    }


def legacy_matchup_response(home_abbr: str, away_abbr: str) -> dict[str, Any]:
    model_id = latest_model_id("nba", "ensemble")
    if not model_id:
        raise RuntimeError("no trained NBA model — run `divinelines train --sport nba`")
    bundle = load_artifact(model_id)
    model, builder = bundle["model"], bundle["feature_builder"]

    home_uid, away_uid = f"nba:{home_abbr}", f"nba:{away_abbr}"
    features = builder.upcoming_features(
        home_uid, away_uid, pd.Timestamp(datetime.now(timezone.utc).date()),
        nba_season_for_date(),
    )
    frame = pd.DataFrame([features])
    for column in model.features:
        if column not in frame.columns:
            frame[column] = np.nan

    detail = model.predict_detail(frame)
    home_probability = float(detail["probability"][0])
    win_pct = round(home_probability * 100, 2)
    favourite = home_abbr if win_pct >= 50 else away_abbr
    favourite_pct = win_pct if win_pct >= 50 else 100 - win_pct

    team_games = load_nba_team_games()
    last_h2h, h2h_stats = _head_to_head(team_games, home_uid, away_uid)

    return {
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_win_probability": win_pct,
        "message": (f"The model favours {favourite} with a {favourite_pct:.1f}% "
                    f"probability of winning."),
        "quant_edge": _quant_edge(home_uid, away_uid, home_probability),
        "metrics": {
            "last_h2h": last_h2h,
            "h2h_stats": h2h_stats,
            "home_stats": _team_rolling_stats(team_games, home_uid),
            "away_stats": _team_rolling_stats(team_games, away_uid),
        },
        # v2 additions.
        "model_id": model_id,
        "model_version": model.model_version,
        "components": {
            "xgboost": round(float(detail["xgboost"][0]), 4),
            "logistic": round(float(detail["logistic"][0]), 4),
            "elo": round(float(detail["elo"][0]), 4),
        },
        "model_agreement": round(float(detail["agreement"][0]), 4),
    }
