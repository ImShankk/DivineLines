import sqlite3
import os


def create_database(db_path: str = "../data/processed/nba_data.db") -> None:
    """
    Creates a SQLite database for storing NBA data.

    Args:
        db_path (str): The full path where the SQLite database will be created.
    """
    print("DivineLines: Creating SQLite database...")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to the SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"Success! Database created at {db_path}")

    # Create the necessary tables for our data model

    # TABLE 1: TEAMS (One row per team, with static info about the team)
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY,
        full_name TEXT,
        abbreviation TEXT,
        nickname TEXT,
        city TEXT,
        state TEXT,
        year_founded INTEGER
    )
    """
    )

    # TABLE 2: PLAYERS (One row per player, with static info about the player)
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY,
        full_name TEXT,
        first_name TEXT,
        last_name TEXT,
        is_active BOOLEAN
    )
    """
    )

    #  TABLE 3: GAME LOGS (One row per team per game, so each game has 2 rows)
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS game_logs (
        SEASON_ID TEXT,
        TEAM_ID INTEGER,
        TEAM_ABBREVIATION TEXT,
        TEAM_NAME TEXT,
        GAME_ID TEXT,
        GAME_DATE DATE,
        MATCHUP TEXT,
        WL TEXT,
        MIN INTEGER,
        FGM INTEGER,
        FGA INTEGER,
        FG_PCT REAL,
        FG3M INTEGER,
        FG3A INTEGER,
        FG3_PCT REAL,
        FTM INTEGER,
        FTA INTEGER,
        FT_PCT REAL,
        OREB INTEGER,
        DREB INTEGER,
        REB INTEGER,
        AST INTEGER,
        STL INTEGER,
        BLK INTEGER,
        TOV INTEGER,
        PF INTEGER,
        PTS INTEGER,
        PLUS_MINUS INTEGER,
        VIDEO_AVAILABLE INTEGER,
        PRIMARY KEY (TEAM_ID, GAME_ID),
        FOREIGN KEY (TEAM_ID) REFERENCES teams(id)
    )
    """
    )

    conn.commit()
    # Close the connection
    conn.close()

    print(f"Success! Schema built successfully at {db_path}")


if __name__ == "__main__":
    create_database()
