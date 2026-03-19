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

        # 2. The Target Variable (1 for Win, 0 for Loss)
        team_df = raw_df.sort_values(by=["TEAM_ID", "GAME_DATE"]).copy()
        team_df["WIN_BIN"] = team_df["WL"].apply(lambda x: 1 if x == "W" else 0)

        # 3. Create Multi-Window Rolling Features (Identity vs. Trend)
        stats = ["PTS", "PLUS_MINUS", "FG_PCT", "TOV", "REB", "AST"]

        for stat in stats:
            # Slow Window (The Team's Identity - Last 10 Games)
            team_df[f"ROLL_10_{stat}"] = team_df.groupby("TEAM_ID")[stat].transform(
                lambda x: x.shift(1).rolling(window=10).mean()
            )
            # Fast Window (The Team's Recent Form - Last 3 Games)
            team_df[f"ROLL_3_{stat}"] = team_df.groupby("TEAM_ID")[stat].transform(
                lambda x: x.shift(1).rolling(window=3).mean()
            )
            # The Trend (Momentum: Fast - Slow)
            # Positive number = Team is performing better recently than their average
            team_df[f"TREND_{stat}"] = (
                team_df[f"ROLL_3_{stat}"] - team_df[f"ROLL_10_{stat}"]
            )

        # 4. Momentum Feature: Rolling Win Percentage
        team_df["ROLL_10_WIN_PCT"] = team_df.groupby("TEAM_ID")["WIN_BIN"].transform(
            lambda x: x.shift(1).rolling(window=10).mean()
        )
        team_df["DAYS_REST"] = (
            team_df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(3).clip(0, 4)
        )

        # 5. Drop rows with NaN values (the first few games for each team won't have full rolling stats)
        cols_to_keep = [
            "GAME_ID",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "GAME_DATE",
            "MATCHUP",
            "WIN_BIN",
            "DAYS_REST",
        ] + [col for col in team_df.columns if "ROLL_" in col or "TREND_" in col]

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
        features_to_diff = [
            "ROLL_10_PTS",
            "ROLL_10_PLUS_MINUS",
            "ROLL_10_FG_PCT",
            "ROLL_10_REB",
            "ROLL_10_WIN_PCT",
            "DAYS_REST",
        ]

        for feature in features_to_diff:
            matchups_df[f"DIFF_{feature}"] = (
                matchups_df[f"{feature}_HOME"] - matchups_df[f"{feature}_AWAY"]
            )

        # THE NEW TARGET
        # No longer predicting "Did this team win?".
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
