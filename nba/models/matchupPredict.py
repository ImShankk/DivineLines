import pandas as pd
import xgboost as xgb
import os
import sys
import sqlite3


def get_latest_team_stats(db_path: str, team_abbr: str):
    """Calculates the current rolling averages for a team based on their last 10 games."""
    # We query the raw database directly to get their absolute latest performance
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM game_logs WHERE TEAM_ABBREVIATION = '{team_abbr}' ORDER BY GAME_DATE ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN_BIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    # Grab the last 10 games to calculate current momentum
    last_10 = df.tail(10)

    stats = {}
    stats["ROLL_10_PTS"] = last_10["PTS"].mean()
    stats["ROLL_10_PLUS_MINUS"] = last_10["PLUS_MINUS"].mean()
    stats["ROLL_10_FG_PCT"] = last_10["FG_PCT"].mean()
    stats["ROLL_10_REB"] = last_10["REB"].mean()
    stats["ROLL_10_WIN_PCT"] = last_10["WIN_BIN"].mean()

    # Calculate Days Rest from their last played game to today
    last_game_date = df.iloc[-1]["GAME_DATE"]
    days_since_last_game = (pd.Timestamp.today() - last_game_date).days
    stats["DAYS_REST"] = min(max(days_since_last_game, 0), 4)

    return stats


def predict_matchup(home: str, away: str) -> None:
    print(f"\nDivineLines: {home} vs {away}")
    print("-" * 50)

    # 1. Load the Data and the Model
    data_path = os.path.join("..", "data", "processed", "nba_data.db")
    model_path = os.path.join("..", "data", "processed", "divinelines_v2.json")

    if not os.path.exists(model_path):
        print("[ERROR] Could not find the saved model. Run train_model.py first!")
        return

    home_stats = get_latest_team_stats(data_path, home)
    away_stats = get_latest_team_stats(data_path, away)

    if home_stats is None or away_stats is None:
        print(
            "[ERROR] One or both teams not found. Check abbreviations (e.g., 'BOS', 'DAL')."
        )
        return

    # Use XGBoost Booster class to load the model, since we saved it in JSON format
    model = xgb.Booster()
    model.load_model(model_path)

    diff_features = {
        "DIFF_ROLL_10_PTS": home_stats["ROLL_10_PTS"] - away_stats["ROLL_10_PTS"],
        "DIFF_ROLL_10_PLUS_MINUS": home_stats["ROLL_10_PLUS_MINUS"]
        - away_stats["ROLL_10_PLUS_MINUS"],
        "DIFF_ROLL_10_FG_PCT": home_stats["ROLL_10_FG_PCT"]
        - away_stats["ROLL_10_FG_PCT"],
        "DIFF_ROLL_10_REB": home_stats["ROLL_10_REB"] - away_stats["ROLL_10_REB"],
        "DIFF_ROLL_10_WIN_PCT": home_stats["ROLL_10_WIN_PCT"]
        - away_stats["ROLL_10_WIN_PCT"],
        "DIFF_DAYS_REST": home_stats["DAYS_REST"] - away_stats["DAYS_REST"],
    }

    # 2. Get the most recent stats for both teams
    home_stats = get_latest_team_stats(data_path, home)
    away_stats = get_latest_team_stats(data_path, away)

    if home_stats is None or away_stats is None:
        print(
            "[ERROR] One or both teams not found in the database. Check abbreviations (e.g., 'BOS', 'LAL')."
        )
        return

    matchup_df = pd.DataFrame([diff_features])

    model = xgb.Booster()
    model.load_model(model_path)
    dmatrix = xgb.DMatrix(matchup_df)

    homewin_prob = model.predict(dmatrix)[0] * 100
    awaywin_prob = 100 - homewin_prob

    print(f"{home} Win Probability: {homewin_prob:.1f}%")
    print(f"{away} Win Probability: {awaywin_prob:.1f}%")

    if homewin_prob > 50:
        print(
            f"PREDICTION: {home} (HOME TEAM) wins with {homewin_prob:.1f}% confidence."
        )
    else:
        print(
            f"PREDICTION: {away} (AWAY TEAM) wins with {awaywin_prob:.1f}% confidence."
        )
    print("-" * 40)


if __name__ == "__main__":
    # Test for now
    HOME = "WAS"
    AWAY = "BOS"

    predict_matchup(HOME, AWAY)
