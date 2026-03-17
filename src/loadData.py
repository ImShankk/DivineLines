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

    # Connect to the database
    conn = sqlite3.connect(db_path)

    try:
        # --- 1. Load Teams ---
        print("-> Loading Teams...")
        teams_df = pd.read_csv(os.path.join(data_dir, "nba_teams.csv"))
        # if_exists="append" adds the data to our existing schema without overwriting it
        teams_df.to_sql("teams", conn, if_exists="append", index=False)
        print(f"   [SUCCESS] {len(teams_df)} teams loaded.")

        # --- 2. Load Players ---
        print("-> Loading Players...")
        players_df = pd.read_csv(os.path.join(data_dir, "nba_players.csv"))
        players_df.to_sql("players", conn, if_exists="append", index=False)
        print(f"   [SUCCESS] {len(players_df)} players loaded.")

        # --- 3. Load All Game Logs ---
        print("-> Loading Game Logs...")
        # Find every CSV file that starts with "nba_games_"
        game_files = glob.glob(os.path.join(data_dir, "nba_games_*.csv"))

        total_games = 0
        for file in game_files:
            season_name = (
                os.path.basename(file).replace("nba_games_", "").replace(".csv", "")
            )
            print(f"   Loading season: {season_name}...")

            games_df = pd.read_csv(file)
            games_df.to_sql("game_logs", conn, if_exists="append", index=False)
            total_games += len(games_df)

        print(f"   [SUCCESS] {total_games} total game records loaded.")

    except sqlite3.IntegrityError as e:
        print(f"\n[WARNING] Data Integrity Error: {e}")
        print(
            "Tip: This usually means the data is already in the database. "
            "The Primary Keys prevent you from loading the same game twice!"
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline Failed: {e}")
    finally:
        conn.close()
        print("\nDivineLines: Database loading complete.")


if __name__ == "__main__":
    load_csv_to_db()
