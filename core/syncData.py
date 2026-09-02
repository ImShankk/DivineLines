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

        df.to_sql("game_logs", conn, if_exists="append", index=False)

        # Remove duplicates
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM game_logs 
            WHERE rowid NOT IN (
                SELECT MIN(rowid) 
                FROM game_logs 
                GROUP BY GAME_ID, TEAM_ID
            )
        """
        )
        conn.commit()

        # 3. Verify the total size
        cursor.execute("SELECT COUNT(*) FROM game_logs")
        total_rows = cursor.fetchone()[0]

        print(
            f"[SUCCESS] Database synced. Currently holding {total_rows} total game records."
        )

    except Exception as e:
        print(
            f"[!] ERROR: Failed to sync database. Make sure you have internet access."
        )
        print(f"Details: {e}")


if __name__ == "__main__":
    sync_database()
