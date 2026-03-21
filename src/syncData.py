import sqlite3
import pandas as pd
import os
from nba_api.stats.endpoints import leaguegamelog


def sync_database():
    """Fetches the latest 2025-26 game logs from the NBA API and updates the local database."""
    print("DivineLines: Initiating connection to NBA servers...")

    try:
        # 1. Fetch the live data for the current season
        gamelog = leaguegamelog.LeagueGameLog(
            season="2025-26", season_type_all_star="Regular Season"
        )

        # Convert the raw API JSON into a clean Pandas DataFrame
        df = gamelog.get_data_frames()[0]

        # 2. Connect to local database
        db_path = os.path.join("..", "data", "processed", "nba_data.db")

        # Make sure the folder exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)

        # 3. Overwrite the old table with the completely up-to-date data
        df.to_sql("game_logs", conn, if_exists="replace", index=False)
        conn.close()

        print(
            f"[SUCCESS] Database synced. Currently holding {len(df)} total game records."
        )

    except Exception as e:
        print(
            f"[!] ERROR: Failed to sync database. Make sure you have internet access."
        )
        print(f"Details: {e}")


if __name__ == "__main__":
    sync_database()
