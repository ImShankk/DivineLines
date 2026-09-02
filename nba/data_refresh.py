import sqlite3
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from datetime import datetime
import os


def refresh_nba_data():
    db_path = os.path.join("..", "data", "processed", "nba_data.db")
    conn = sqlite3.connect(db_path)

    # 1. Find the most recent game in your database
    print("[*] Checking database for last update...")
    try:
        last_date_query = "SELECT MAX(GAME_DATE) FROM game_logs"
        last_date_str = pd.read_sql_query(last_date_query, conn).iloc[0, 0]
        # Date conversion for the API
        last_date_obj = datetime.strptime(last_date_str, "%Y-%m-%d")
        start_date = last_date_obj.strftime("%m/%d/%Y")
        print(f"[*] Last game found: {last_date_str}. Fetching new data since then...")
    except Exception as e:
        print(
            f"[!] Database empty or error: {e}. Defaulting to start of 2025-26 season."
        )
        start_date = "10/20/2025"

    # 2. Fetch new logs from NBA API
    try:
        # Fetch for the current season (2025-26)
        log = leaguegamelog.LeagueGameLog(
            season="2025-26", date_from_nullable=start_date
        )
        new_logs = log.get_data_frames()[0]

        if new_logs.empty:
            print("[SUCCESS] Database is already up to date. No new games found.")
            return

        # 3. Data Cleaning
        # Filter out games we already have (since the API date_from is inclusive)
        new_logs["GAME_DATE"] = pd.to_datetime(new_logs["GAME_DATE"]).dt.strftime(
            "%Y-%m-%d"
        )
        new_logs = new_logs[new_logs["GAME_DATE"] > last_date_str]

        if new_logs.empty:
            print("[SUCCESS] No new unique games to add.")
            return

        print(f"[*] Found {len(new_logs)} new team logs. Appending to database...")

        # 4. Append to SQLite
        new_logs.to_sql("game_logs", conn, if_exists="append", index=False)
        print(f"[SUCCESS] Database updated. Your model now sees the latest NBA action.")

    except Exception as e:
        print(f"[!] API Error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    refresh_nba_data()
