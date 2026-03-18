import os
import glob
import pandas as pd
import sqlite3

import os
import glob
import sqlite3
import pandas as pd


def load_csv_to_db(
    db_path: str = "../data/processed/nba_data.db", data_dir: str = "../data/raw"
) -> None:
    """
    Reads all raw CSV files and loads them into the SQLite database.
    """
    print("DivineLines: Initializing Data Loading Protocol...")

    conn = sqlite3.connect(db_path)

    try:
        # 1. Load Teams & Players (Use 'REPLACE' so they stay updated but don't duplicate)
        teams_df = pd.read_csv(os.path.join(data_dir, "nba_teams.csv"))
        teams_df.to_sql("teams", conn, if_exists="replace", index=False)

        players_df = pd.read_csv(os.path.join(data_dir, "nba_players.csv"))
        players_df.to_sql("players", conn, if_exists="replace", index=False)

        # 2. Load Game Logs using an "INSERT OR IGNORE" strategy
        game_files = glob.glob(os.path.join(data_dir, "nba_games_*.csv"))

        for file in game_files:
            print(f"-> Processing {os.path.basename(file)}...")
            df = pd.read_csv(file)

            # Create a temporary table to hold the new data
            df.to_sql("temp_game_logs", conn, if_exists="replace", index=False)

            # Use SQL to move data from TEMP to the MAIN table, skipping duplicates
            conn.execute(
                """
                INSERT OR IGNORE INTO game_logs 
                SELECT * FROM temp_game_logs
            """
            )

            # Drop the temp table
            conn.execute("DROP TABLE temp_game_logs")
            conn.commit()

        print("[SUCCESS] Database synchronized. No duplicates created.")

    except Exception as e:
        print(f"[ERROR] Load failed: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    load_csv_to_db()
