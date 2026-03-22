from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import xgboost as xgb
import os

app = FastAPI(title="DivineLines V4 Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Matchup(BaseModel):
    home: str
    away: str


# Load the model once when the server starts (saves memory and time)
MODEL_PATH = os.path.join("..", "data", "processed", "divinelines_v3_optimized.json")
try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_PATH)
    print("[SUCCESS] XGBoost Brain Loaded.")
except Exception as e:
    print(f"[!] Warning: Could not load model. Ensure path is correct. {e}")


@app.get("/")
def read_root():
    return {"status": "DivineLines V4 API is Online and Listening."}


@app.post("/api/predict")
def get_prediction(matchup: Matchup):
    try:
        db_path = os.path.join("..", "data", "processed", "nba_data.db")
        conn = sqlite3.connect(db_path)

        # 1. Pull the raw logs to get current stats
        query = "SELECT * FROM game_logs ORDER BY GAME_DATE ASC"
        raw_df = pd.read_sql_query(query, conn)
        conn.close()

        raw_df["GAME_DATE"] = pd.to_datetime(raw_df["GAME_DATE"])
        raw_df.sort_values(by=["GAME_ID", "TEAM_ID"], inplace=True)

        # 2. Recreate the V4 Math Environment
        raw_df["POSS"] = (
            raw_df["FGA"] - raw_df["OREB"] + raw_df["TOV"] + (0.44 * raw_df["FTA"])
        )
        raw_df["ORTG"] = (raw_df["PTS"] / raw_df["POSS"]) * 100
        raw_df["POINT_DIFF"] = raw_df["PLUS_MINUS"]

        raw_df["OPP_PTS"] = raw_df.groupby("GAME_ID")["PTS"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_POSS"] = raw_df.groupby("GAME_ID")["POSS"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_FG3A"] = raw_df.groupby("GAME_ID")["FG3A"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_FG3M"] = raw_df.groupby("GAME_ID")["FG3M"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_FTA"] = raw_df.groupby("GAME_ID")["FTA"].transform(
            lambda x: x.iloc[::-1].values
        )

        raw_df["DRTG"] = (raw_df["OPP_PTS"] / raw_df["OPP_POSS"]) * 100
        raw_df["NET_RATING"] = raw_df["ORTG"] - raw_df["DRTG"]
        raw_df["OPP_FG3_PCT"] = raw_df.apply(
            lambda row: row["OPP_FG3M"] / row["OPP_FG3A"] if row["OPP_FG3A"] > 0 else 0,
            axis=1,
        )
        raw_df["WIN_BIN"] = raw_df["WL"].apply(lambda x: 1 if x == "W" else 0)

        # Helper to get a team's last 10 games
        def get_latest_stats(team_abbr):
            team_data = (
                raw_df[raw_df["TEAM_ABBREVIATION"] == team_abbr]
                .sort_values(by="GAME_DATE")
                .tail(10)
            )
            return team_data.mean(numeric_only=True)

        home_stats = get_latest_stats(matchup.home)
        away_stats = get_latest_stats(matchup.away)

        # 3. Calculate the Matchup Differentials
        features = {}
        features["DIFF_PACE"] = home_stats["POSS"] - away_stats["POSS"]
        features["DIFF_NET_RATING"] = (
            home_stats["NET_RATING"] - away_stats["NET_RATING"]
        )
        features["DIFF_POINT_MARGIN"] = (
            home_stats["POINT_DIFF"] - away_stats["POINT_DIFF"]
        )

        features["HOME_3PT_ADVANTAGE"] = home_stats["FG3A"] * away_stats["OPP_FG3_PCT"]
        features["AWAY_3PT_ADVANTAGE"] = away_stats["FG3A"] * home_stats["OPP_FG3_PCT"]
        features["HOME_FT_ADVANTAGE"] = home_stats["FTA"] - away_stats["OPP_FTA"]
        features["AWAY_FT_ADVANTAGE"] = away_stats["FTA"] - home_stats["OPP_FTA"]

        # Standard diffs
        for stat in ["ORTG", "DRTG", "REB", "AST", "TOV", "WIN_BIN"]:
            features[f"DIFF_ROLL_10_{stat}"] = home_stats[stat] - away_stats[stat]

        features["H2H_WIN_PCT"] = 0.50

        # 4. Format for XGBoost
        pred_df = pd.DataFrame([features])

        # This prevents the API from crashing if a feature name slightly mismatches
        expected_cols = xgb_model.get_booster().feature_names
        for col in expected_cols:
            if col not in pred_df.columns:
                pred_df[col] = 0  # Fill missing context (like B2B) with 0
        pred_df = pred_df[expected_cols]

        # 5. Execute Prediction
        prob = float(xgb_model.predict_proba(pred_df)[0][1])  # forgot to make float
        # Probability of Class 1 (Home Win)
        win_pct = round(prob * 100, 2)

        # Determine the favorite
        favorite = matchup.home if win_pct >= 50 else matchup.away
        fav_pct = win_pct if win_pct >= 50 else (100 - win_pct)

        # My idea of futmob like stats on the side.
        h2h_games = raw_df[
            (
                (raw_df["TEAM_ABBREVIATION"] == matchup.home)
                & (raw_df["MATCHUP"].str.contains(matchup.away))
            )
            | (
                (raw_df["TEAM_ABBREVIATION"] == matchup.away)
                & (raw_df["MATCHUP"].str.contains(matchup.home))
            )
        ]

        last_h2h_str = "No matchups yet this season"
        h2h_stats = None

        if not h2h_games.empty:
            last_game_id = h2h_games.iloc[-1]["GAME_ID"]
            game_rows = raw_df[raw_df["GAME_ID"] == last_game_id]

            if len(game_rows) == 2:
                t_away = game_rows[game_rows["TEAM_ABBREVIATION"] == matchup.away].iloc[
                    0
                ]
                t_home = game_rows[game_rows["TEAM_ABBREVIATION"] == matchup.home].iloc[
                    0
                ]
                date_str = pd.to_datetime(t_away["GAME_DATE"]).strftime("%b %d, %Y")
                # Format: "WINNER 110 - 105 LOSER (Date)"
                if t_home["PTS"] > t_away["PTS"]:
                    last_h2h_str = f"{t_home['TEAM_ABBREVIATION']} {int(t_home['PTS'])} - {int(t_away['PTS'])} {t_away['TEAM_ABBREVIATION']} ({date_str})"
                else:
                    last_h2h_str = f"{t_away['TEAM_ABBREVIATION']} {int(t_away['PTS'])} - {int(t_home['PTS'])} {t_home  ['TEAM_ABBREVIATION']} ({date_str})"

                # Kind of like box scores
                h2h_stats = {
                    "away": {
                        "pts": int(t_away["PTS"]),
                        "opp_pts": int(t_home["PTS"]),
                        "reb": int(t_away["REB"]),
                        "ast": int(t_away["AST"]),
                        "fg3m": int(t_away["FG3M"]),
                        "tov": int(t_away["TOV"]),
                        "pf": int(t_away["PF"]),
                    },
                    "home": {
                        "pts": int(t_home["PTS"]),
                        "opp_pts": int(t_away["PTS"]),
                        "reb": int(t_home["REB"]),
                        "ast": int(t_home["AST"]),
                        "fg3m": int(t_home["FG3M"]),
                        "tov": int(t_home["TOV"]),
                        "pf": int(t_home["PF"]),
                    },
                }

        # better representation of the data (if possible trying to do players and injurires too)
        return {
            "home_team": matchup.home,
            "away_team": matchup.away,
            "home_win_probability": win_pct,
            "message": f"The model heavily favors {favorite} with a {fav_pct:.1f}% probability of winning.",
            "metrics": {
                "last_h2h": last_h2h_str,
                "h2h_stats": h2h_stats,
                "away_stats": {
                    "pts": float(away_stats.get("PTS", 0)),
                    "opp_pts": float(away_stats.get("OPP_PTS", 0)),
                    "fg3m": float(away_stats.get("FG3M", 0)),
                    "reb": float(away_stats.get("REB", 0)),
                    "ast": float(away_stats.get("AST", 0)),
                    "tov": float(away_stats.get("TOV", 0)),
                    "ortg": float(away_stats.get("ORTG", 0)),
                    "drtg": float(away_stats.get("DRTG", 0)),
                    "net_rating": float(away_stats.get("NET_RATING", 0)),
                    "pace": float(away_stats.get("POSS", 0)),
                },
                "home_stats": {
                    "pts": float(home_stats.get("PTS", 0)),
                    "opp_pts": float(home_stats.get("OPP_PTS", 0)),
                    "fg3m": float(home_stats.get("FG3M", 0)),
                    "reb": float(home_stats.get("REB", 0)),
                    "ast": float(home_stats.get("AST", 0)),
                    "tov": float(home_stats.get("TOV", 0)),
                    "ortg": float(home_stats.get("ORTG", 0)),
                    "drtg": float(home_stats.get("DRTG", 0)),
                    "net_rating": float(home_stats.get("NET_RATING", 0)),
                    "pace": float(home_stats.get("POSS", 0)),
                },
            },
        }

    except Exception as e:
        return {
            "home_team": matchup.home,
            "away_team": matchup.away,
            "message": f"Engine Error: {str(e)}",
        }


if __name__ == "__main__":
    print("Starting DivineLines Brain on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
