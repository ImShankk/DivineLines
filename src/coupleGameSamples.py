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
        df = pd.read_sql_query(query, conn)

        # 2. The Target Variable (1 for Win, 0 for Loss)
        df["WIN_TARGET"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

        # 3. Create Multi-Window Rolling Features (Identity vs. Trend)
        stats = ["PTS", "PLUS_MINUS", "FG_PCT", "TOV", "REB", "AST"]

        for stat in stats:
            # Slow Window (The Team's Identity - Last 10 Games)
            df[f"ROLL_10_{stat}"] = df.groupby("TEAM_ID")[stat].transform(
                lambda x: x.shift(1).rolling(window=10).mean()
            )
            # Fast Window (The Team's Recent Form - Last 3 Games)
            df[f"ROLL_3_{stat}"] = df.groupby("TEAM_ID")[stat].transform(
                lambda x: x.shift(1).rolling(window=3).mean()
            )
            # The Trend (Momentum: Fast - Slow)
            # Positive number = Team is performing better recently than their average
            df[f"TREND_{stat}"] = df[f"ROLL_3_{stat}"] - df[f"ROLL_10_{stat}"]

        # 4. Momentum Feature: Rolling Win Percentage
        df["ROLL_10_WIN_PCT"] = df.groupby("TEAM_ID")["WIN_TARGET"].transform(
            lambda x: x.shift(1).rolling(window=10).mean()
        )

        # 5. Fatigue Feature: Days of Rest
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["DAYS_REST"] = df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
        df["DAYS_REST"] = df["DAYS_REST"].fillna(3).clip(0, 4)

        # 6. Data Cleaning for XGBoost (The "Drop NaN" Rule)
        # The first 10 games of every season for every team will be blank (NaN).
        # XGBoost hates blanks. We MUST drop them so the model doesn't crash.
        clean_df = df.dropna(subset=["ROLL_10_PTS"]).copy()

        # 7. Save to CSV for the Brain to eat
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        clean_df.to_csv(output_path, index=False)

        print(f"[SUCCESS] Engineered {len(clean_df)} perfect ML snapshots.")
        print(f"[SUCCESS] Feature set saved to {output_path}")

    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    DB = os.path.join("..", "data", "processed", "nba_data.db")
    OUT = os.path.join("..", "data", "processed", "engineered_features.csv")
    engineer_ml_features(DB, OUT)
