import sqlite3
import pandas as pd
import numpy as np
import os


def build_historical_dataset():
    print("--- INITIATING V4 HISTORICAL FEATURE ENGINEERING ---")
    db_path = os.path.join("..", "data", "processed", "nba_data.db")

    if not os.path.exists(db_path):
        print(f"[!] Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)

    print("1. Extracting Raw Game Logs...")
    query = "SELECT * FROM game_logs ORDER BY GAME_DATE ASC"
    df = pd.read_sql_query(query, conn)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.sort_values(by=["GAME_DATE", "GAME_ID"], inplace=True)

    print("2. Calculating V4 Advanced Metrics...")
    df["POSS"] = df["FGA"] - df["OREB"] + df["TOV"] + (0.44 * df["FTA"])
    df["ORTG"] = (df["PTS"] / df["POSS"]) * 100
    df["POINT_DIFF"] = df["PLUS_MINUS"]

    # grab opponent stats by reversing the groupby order
    df["OPP_PTS"] = df.groupby("GAME_ID")["PTS"].transform(
        lambda x: x.iloc[::-1].values
    )
    df["OPP_POSS"] = df.groupby("GAME_ID")["POSS"].transform(
        lambda x: x.iloc[::-1].values
    )
    df["OPP_FG3A"] = df.groupby("GAME_ID")["FG3A"].transform(
        lambda x: x.iloc[::-1].values
    )
    df["OPP_FG3M"] = df.groupby("GAME_ID")["FG3M"].transform(
        lambda x: x.iloc[::-1].values
    )
    df["OPP_FTA"] = df.groupby("GAME_ID")["FTA"].transform(
        lambda x: x.iloc[::-1].values
    )

    df["DRTG"] = (df["OPP_PTS"] / df["OPP_POSS"]) * 100
    df["NET_RATING"] = df["ORTG"] - df["DRTG"]
    df["OPP_FG3_PCT"] = df.apply(
        lambda row: row["OPP_FG3M"] / row["OPP_FG3A"] if row["OPP_FG3A"] > 0 else 0,
        axis=1,
    )
    df["WIN_BIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    print("3. Generating Rolling 10-Game Windows (Strictly Pre-Game)...")
    # strictly shift by 1 to prevent data leakage (can't predict a game using its own stats)
    rolling_cols = [
        "POSS",
        "NET_RATING",
        "POINT_DIFF",
        "FG3A",
        "OPP_FG3_PCT",
        "FTA",
        "OPP_FTA",
        "ORTG",
        "DRTG",
        "REB",
        "AST",
        "TOV",
        "WIN_BIN",
    ]

    df_rolling = (
        df.groupby("TEAM_ABBREVIATION")[rolling_cols]
        .apply(lambda x: x.shift(1).rolling(10, min_periods=10).mean())
        .reset_index(level=0, drop=True)
    )

    # slap rolling stats back onto the main df
    df = df.join(df_rolling, rsuffix="_ROLL10")

    # scrub the first 10 games of the season (too much noise)
    df.dropna(subset=["POSS_ROLL10"], inplace=True)

    print("4. Pairing Matchups and Calculating Differentials...")
    # split into home and away subsets
    home_df = df[df["MATCHUP"].str.contains("vs.")].copy()
    away_df = df[df["MATCHUP"].str.contains("@")].copy()

    # join them back on the game id
    games = pd.merge(home_df, away_df, on="GAME_ID", suffixes=("_HOME", "_AWAY"))

    features = pd.DataFrame()
    features["GAME_ID"] = games["GAME_ID"]
    features["GAME_DATE"] = games["GAME_DATE_HOME"]
    features["HOME_TEAM"] = games["TEAM_ABBREVIATION_HOME"]
    features["AWAY_TEAM"] = games["TEAM_ABBREVIATION_AWAY"]

    # v4 custom diffs
    features["DIFF_PACE"] = games["POSS_ROLL10_HOME"] - games["POSS_ROLL10_AWAY"]
    features["DIFF_NET_RATING"] = (
        games["NET_RATING_ROLL10_HOME"] - games["NET_RATING_ROLL10_AWAY"]
    )
    features["DIFF_POINT_MARGIN"] = (
        games["POINT_DIFF_ROLL10_HOME"] - games["POINT_DIFF_ROLL10_AWAY"]
    )

    features["HOME_3PT_ADVANTAGE"] = (
        games["FG3A_ROLL10_HOME"] * games["OPP_FG3_PCT_ROLL10_AWAY"]
    )
    features["AWAY_3PT_ADVANTAGE"] = (
        games["FG3A_ROLL10_AWAY"] * games["OPP_FG3_PCT_ROLL10_HOME"]
    )
    features["HOME_FT_ADVANTAGE"] = (
        games["FTA_ROLL10_HOME"] - games["OPP_FTA_ROLL10_AWAY"]
    )
    features["AWAY_FT_ADVANTAGE"] = (
        games["FTA_ROLL10_AWAY"] - games["OPP_FTA_ROLL10_HOME"]
    )

    # generic 10-game diffs
    for stat in ["ORTG", "DRTG", "REB", "AST", "TOV", "WIN_BIN"]:
        features[f"DIFF_ROLL_10_{stat}"] = (
            games[f"{stat}_ROLL10_HOME"] - games[f"{stat}_ROLL10_AWAY"]
        )

    features["H2H_WIN_PCT"] = 0.50  # to-do: replace placeholder

    # target var
    features["ACTUAL_WINNER"] = games["WIN_BIN_HOME"]

    print("5. Loading into Database...")
    features.to_sql("historical_features", conn, if_exists="replace", index=False)
    conn.close()

    print(f"[SUCCESS] {len(features)} historical matchups generated and saved.")


if __name__ == "__main__":
    build_historical_dataset()
