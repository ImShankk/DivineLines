import sqlite3
import pandas as pd
import os


def engineer_ml_features(db_path: str, output_path: str) -> None:
    """
    The Ultimate Feature Engine: Creates time-sensitive, multi-window
    snapshots for XGBoost training and drops invalid empty rows.
    """
    print("DivineLines: Initializing Machine Learning Feature Engine...")
    conn = sqlite3.connect(db_path)

    try:
        # 1. Pull the raw logs ordered by time so we can create accurate rolling features
        query = "SELECT * FROM game_logs ORDER BY TEAM_ID, GAME_DATE ASC"
        raw_df = pd.read_sql_query(query, conn)
        raw_df["GAME_DATE"] = pd.to_datetime(raw_df["GAME_DATE"])

        raw_df.sort_values(by=["GAME_ID", "TEAM_ID"], inplace=True)

        print("Calculating Pace, Offensive Rating, and Defensive Profiles...")

        # Calculate Base Possessions
        raw_df["POSS"] = (
            raw_df["FGA"] - raw_df["OREB"] + raw_df["TOV"] + (0.44 * raw_df["FTA"])
        )
        raw_df["ORTG"] = (raw_df["PTS"] / raw_df["POSS"]) * 100

        # MAGIC PANDAS TRICK: Find the opponent's stats for that specific game
        # We flip the rows inside each game to see what the OTHER team did
        raw_df["OPP_PTS"] = raw_df.groupby("GAME_ID")["PTS"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_POSS"] = raw_df.groupby("GAME_ID")["POSS"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_FG3A"] = raw_df.groupby("GAME_ID")["FG3A"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_FTA"] = raw_df.groupby("GAME_ID")["FTA"].transform(
            lambda x: x.iloc[::-1].values
        )

        # Calculate Defensive Rating (How many points they allowed per 100 possessions)
        raw_df["DRTG"] = (raw_df["OPP_PTS"] / raw_df["OPP_POSS"]) * 100
        raw_df["NET_RATING"] = raw_df["ORTG"] - raw_df["DRTG"]

        # 2. The Target Variable (1 for Win, 0 for Loss)
        team_df = raw_df.sort_values(by=["TEAM_ID", "GAME_DATE"]).copy()
        team_df["WIN_BIN"] = team_df["WL"].apply(lambda x: 1 if x == "W" else 0)

        # 3. Create Multi-Window Rolling Features (Identity vs. Trend)
        # stats = [
        #     "PTS",
        #     "PLUS_MINUS",
        #     "FG_PCT",
        #     "FG3_PCT",
        #     "FT_PCT",
        #     "REB",
        #     "OREB",
        #     "AST",
        #     "TOV",
        #     "STL",
        #     "BLK",
        #     "PF",
        #     "WIN_BIN",
        # ]

        # Trying new way
        # As a NBA wacther these are the stats I care about most when analyzing a team:
        stats = [
            "ORTG",
            "DRTG",
            "NET_RATING",
            "POSS",
            "FG3A",
            "OPP_FG3A",
            "FTA",
            "OPP_FTA",
            "REB",
            "AST",
            "TOV",
            "WIN_BIN",
        ]

        for stat in stats:
            team_df[f"ROLL_10_{stat}"] = team_df.groupby("TEAM_ID")[stat].transform(
                lambda x: x.shift(1).rolling(window=10).mean()
            )

        # 4. Momentum Feature: Rolling Win Percentage
        # team_df["ROLL_10_WIN_PCT"] = team_df.groupby("TEAM_ID")["WIN_BIN"].transform(
        #     lambda x: x.shift(1).rolling(window=10).mean()
        # )
        team_df["DAYS_REST"] = (
            team_df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(3).clip(0, 4)
        )

        # 5. Drop rows with NaN values (the first few games for each team won't have full rolling stats)
        cols_to_keep = [
            "GAME_ID",
            "TEAM_ID",
            "MATCHUP",
            "GAME_DATE",
            "WIN_BIN",
            "DAYS_REST",
        ] + [col for col in team_df.columns if "ROLL_10_" in col]
        team_df = team_df[cols_to_keep].dropna()

        # We separate the data into Home Teams and Away Teams based on the Matchup string.
        home_teams = team_df[team_df["MATCHUP"].str.contains(" vs. ")].copy()
        away_teams = team_df[team_df["MATCHUP"].str.contains(" @ ")].copy()

        # Glue the Home and Away stats together onto the exact same row using GAME_ID.
        # It automatically adds "_HOME" and "_AWAY" to the column names so we don't mix them up.
        matchups_df = pd.merge(
            home_teams,
            away_teams,
            on=["GAME_ID", "GAME_DATE"],
            suffixes=("_HOME", "_AWAY"),
        )

        # Calculate the spread between them.
        print("Calculating Matchup Differentials...")
        # Pace Difference (Who dictates the speed of the game?)
        matchups_df["DIFF_PACE"] = (
            matchups_df["ROLL_10_POSS_HOME"] - matchups_df["ROLL_10_POSS_AWAY"]
        )

        # Net Rating Differential
        matchups_df["DIFF_NET_RATING"] = (
            matchups_df["ROLL_10_NET_RATING_HOME"]
            - matchups_df["ROLL_10_NET_RATING_AWAY"]
        )

        # 3-Point Clash (Team's 3s taken vs Opponent's 3s allowed)
        matchups_df["HOME_3PT_ADVANTAGE"] = (
            matchups_df["ROLL_10_FG3A_HOME"] - matchups_df["ROLL_10_OPP_FG3A_AWAY"]
        )
        matchups_df["AWAY_3PT_ADVANTAGE"] = (
            matchups_df["ROLL_10_FG3A_AWAY"] - matchups_df["ROLL_10_OPP_FG3A_HOME"]
        )

        # Foul Clash (Team's Free Throws taken vs Opponent's Fouls committed)
        matchups_df["HOME_FT_ADVANTAGE"] = (
            matchups_df["ROLL_10_FTA_HOME"] - matchups_df["ROLL_10_OPP_FTA_AWAY"]
        )
        matchups_df["AWAY_FT_ADVANTAGE"] = (
            matchups_df["ROLL_10_FTA_AWAY"] - matchups_df["ROLL_10_OPP_FTA_HOME"]
        )

        # Standard Differentials for the rest
        standard_diffs = [
            "ROLL_10_ORTG",
            "ROLL_10_DRTG",
            "ROLL_10_REB",
            "ROLL_10_AST",
            "ROLL_10_TOV",
            "ROLL_10_WIN_BIN",
            "DAYS_REST",
        ]
        for feature in standard_diffs:
            matchups_df[f"DIFF_{feature}"] = (
                matchups_df[f"{feature}_HOME"] - matchups_df[f"{feature}_AWAY"]
            )

        matchups_df["H2H_WIN_PCT"] = 0.50

        # THE NEW TARGET
        # Now predicting "Did the HOME team win?"
        matchups_df["HOME_WIN_TARGET"] = matchups_df["WIN_BIN_HOME"]

        # 7. Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        matchups_df.to_csv(output_path, index=False)

        print(f"[SUCCESS] Engineered {len(matchups_df)} perfect ML snapshots.")
        print(f"[SUCCESS] Feature set saved to {output_path}")

    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    DB = os.path.join("..", "data", "processed", "nba_data.db")
    OUT = os.path.join("..", "data", "processed", "engineered_features.csv")
    engineer_ml_features(DB, OUT)
