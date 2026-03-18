import sqlite3
import os


def reset_tables():
    # 1. Get the absolute path to the directory this script is in (src)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Build the absolute path to the database
    # This goes UP one level to DivineLines, then into data/processed
    db_path = os.path.abspath(
        os.path.join(script_dir, "..", "data", "processed", "nba_data.db")
    )

    print(f"DivineLines: Attempting to connect to {db_path}")

    # 3. Double check if the folder exists before trying to open the DB
    if not os.path.exists(os.path.dirname(db_path)):
        print(f"[ERROR] The directory {os.path.dirname(db_path)} does not exist!")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("-> Wiping tables...")
        cursor.execute("DELETE FROM game_logs")
        cursor.execute("DELETE FROM teams")
        cursor.execute("DELETE FROM players")

        conn.commit()
        conn.close()
        print("[SUCCESS] Database wiped clean. Ready for a fresh load.")

    except sqlite3.OperationalError as e:
        print(f"[ERROR] Could not open database: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    reset_tables()
