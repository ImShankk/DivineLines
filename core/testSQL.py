import sqlite3
import pandas as pd
import os


def run_diagnostic_query(db_path: str, query: str) -> pd.DataFrame:
    """
    Connects to the SQLite database, runs the provided SQL query,
    and returns the results as a DataFrame.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)

    try:
        # Execute the query and fetch results into a DataFrame
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        # Ensure the connection is closed even if an error occurs
        conn.close()


if __name__ == "__main__":
    # Define the path to the database (use os.path.join for cross-platform compatibility)
    DB_PATH = os.path.join("..", "data", "processed", "nba_data.db")

    # Checks if the WL (Win/Loss) column and TEAM_NAME are mapped correctly
    test_query = """
    SELECT 
        TEAM_NAME, 
        COUNT(GAME_ID) AS Games_Analyzed,
        SUM(CASE WHEN WL = 'W' THEN 1 ELSE 0 END) AS Total_Wins,
        ROUND(AVG(CASE WHEN WL = 'W' THEN 1.0 ELSE 0.0 END) * 100, 2) AS Win_Percentage
    FROM game_logs
    GROUP BY TEAM_NAME
    HAVING Games_Analyzed > 100
    ORDER BY Win_Percentage DESC
    LIMIT 10;
    """

    print(f"DivineLines: Querying {DB_PATH}...\n")

    # Execute the function
    results = run_diagnostic_query(DB_PATH, test_query)

    # Display the results
    if not results.empty:
        print("--- HISTORICAL PERFORMANCE LEADERBOARD ---")
        print(results.to_string(index=False))
        print("\n[SUCCESS]")
    else:
        print("[ERROR] No data found.")
